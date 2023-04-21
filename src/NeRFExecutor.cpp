#include "NeRFExecutor.h"

NeRFExecutor :: NeRFExecutor(
	const int net_depth /*= 8*/,
	const int net_width /*= 256*/,
	const int multires /*= 10*/,
	const bool use_viewdirs /*= true*/,
	const int multires_views /*= 4*/,
	const int n_importance /*= 0*/,
	const int net_depth_fine /*= 8*/,
	const int net_width_fine /*= 256*/,
	torch::Device device /*= torch::kCUDA*/,
	const float learning_rate /*= 5e-4*/,
	std::filesystem::path ft_path /*= ""*/
) : Device(device), NImportance(n_importance), UseViewDirs(use_viewdirs), LearningRate(learning_rate)
{
	ExecutorEmbedder = Embedder("embedder", multires);
	int input_ch = ExecutorEmbedder->GetOutputDims(),
		input_ch_views = 0;
	if (use_viewdirs)
	{
		ExecutorEmbeddirs = Embedder("embeddirs", multires_views);
		input_ch_views = ExecutorEmbeddirs->GetOutputDims();
	}

	int output_ch = (n_importance > 0) ? 5 : 4;
	const std::set<int> skips = std::set<int>{ 4 };

	Model = NeRF(net_depth, net_width, input_ch, input_ch_views, output_ch, skips, use_viewdirs);
	Model->to(device);
	GradVars = Model->parameters();

	if (n_importance > 0)
	{
		ModelFine = NeRF(net_depth_fine, net_width_fine, input_ch, input_ch_views, output_ch, skips, use_viewdirs);
		ModelFine->to(device);
		auto temp = ModelFine->parameters();
		GradVars.insert(GradVars.end(), std::make_move_iterator(temp.begin()), std::make_move_iterator(temp.end()));
	}

	//network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
	//	embed_fn = embed_fn,
	//	embeddirs_fn = embeddirs_fn,
	//	netchunk = args.netchunk)

	Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(learning_rate)/*.weight_decay(0.001)*/.betas(std::make_tuple(0.9, 0.999)));

	if (/*Проверить наличие файлов*/
		std::filesystem::exists(ft_path / "start_checkpoint.pt") &&
		std::filesystem::exists(ft_path / "optimizer_checkpoint.pt") &&
		std::filesystem::exists(ft_path / "model_checkpoint.pt") &&
		(ModelFine.is_empty() || (!ModelFine.is_empty() && std::filesystem::exists(ft_path / "model_fine_checkpoint.pt")))
	){
		std::cout << "restoring parameters from checkpoint..." << std::endl;
		torch::Tensor temp;
		torch::load(temp, (ft_path / "start_checkpoint.pt").string());
		Start = temp.item<float>();
		torch::load(*Optimizer.get(), (ft_path / "optimizer_checkpoint.pt").string());
		torch::load(Model, (ft_path / "model_checkpoint.pt").string());
		if (!ModelFine.is_empty())
			torch::load(ModelFine, (ft_path / "model_fine_checkpoint.pt").string());
	}	else {
		Trainable::Initialize(Model);
		if (n_importance > 0)
			Trainable::Initialize(ModelFine);
	}
}			//NeRFExecutor :: NeRFExecutor


std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> NeRFExecutor :: RenderPath(
	const std::vector <torch::Tensor> &render_poses,
	int h,
	int w,
	float focal,
	torch::Tensor k,
	const int n_samples,
	const int chunk /*= 1024 * 32*/,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
	const bool return_raw /*= false*/,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp /*= false*/,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb /*= 0.f*/,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance /*= 0*/,							///Number of additional times to sample along each ray.
	const bool white_bkgr /*= false*/,					///If True, assume a white background.
	const float raw_noise_std /*= 0.*/,
	std::pair<torch::Tensor, torch::Tensor> rays /*= { torch::Tensor(), torch::Tensor() }*/,			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
	const bool ndc /*= true*/,						///If True, represent ray origin, direction in NDC coordinates.
	const float near /*= 0.*/,						///float or array of shape[batch_size].Nearest distance for a ray.
	const float far /*= 1.*/,							///float or array of shape[batch_size].Farthest distance for a ray.
	const bool use_viewdirs /*= false*/,	///If True, use viewing direction of a point in space in model.
	torch::Tensor c2w_staticcam /*= torch::Tensor()*/,			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
	//torch::Tensor gt_imgs = torch::Tensor(),
	const std::filesystem::path savedir /*= ""*/,
	const float render_factor /*= 0*/
) {
	std::vector<torch::Tensor> rgbs,
		disps;

	if (render_factor != 0)
	{
		//Render downsampled for speed
		h = h / render_factor;
		w = w / render_factor;
		focal = focal / render_factor;
	}

	for (auto& c2w : render_poses) //(int i = 0; i < render_poses.size(); i++)
	{
		RenderResult render_result = Render(h, w, k,
			Model,
			ExecutorEmbedder,
			ExecutorEmbeddirs,
			ModelFine,
			n_samples,
			chunk,
			return_raw,
			lin_disp,
			perturb,
			n_importance,
			white_bkgr,
			raw_noise_std,
			rays,
			c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
			ndc,
			near,
			far,
			use_viewdirs,
			c2w_staticcam
		);
		rgbs.push_back(render_result.Outputs1.RGBMap.cpu());
		disps.push_back(render_result.Outputs1.DispMap.cpu());

		if (!savedir.empty())
		{
			cv::imwrite((savedir / (std::to_string(rgbs.size() - 1) + ".png")).string(), TorchTensorToCVMat(rgbs.back()));
		}
	}

	//rgbs = np.stack(rgbs, 0)
	//disps = np.stack(disps, 0)
	return std::make_pair(rgbs, disps);
}			//NeRFExecutor :: RenderPath
