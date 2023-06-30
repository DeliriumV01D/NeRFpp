#pragma once

#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "NeRF.h"
#include "BaseNeRFRenderer.h"
#include "TRandomInt.h"

#include <set>
#include <filesystem>
#include <chrono>

enum class DatasetType { LLFF, BLENDER, LINEMOD, DEEP_VOXELS };

struct NeRFExecutorTrainParams {
	DatasetType DatasetType = DatasetType::BLENDER;
	std::filesystem::path DataDir,	///input data directory
		BaseDir;											///where to store ckpts and logs
	bool HalfRes{ false },					///load blender synthetic data at 400x400 instead of 800x800
		TestSkip{ false },						///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
		WhiteBkgr{ false },						///set to render synthetic data on a white bkgd (always use for dvoxels)
		RenderOnly{ false },					///do not optimize, reload weights and render out render_poses path
		Ndc{ true },									///use normalized device coordinates (set for non-forward facing scenes)
		LinDisp{ false },							///sampling linearly in disparity rather than depth
		NoBatching{ true };						///only take random rays from 1 image at a time
	int Chunk{ 1024 * 32 },					///number of rays processed in parallel, decrease if running out of memory
		NetChunk{ 1024 * 64 },				///number of pts sent through network in parallel, decrease if running out of memory
		NSamples{ 64 },								///number of coarse samples per ray
		NRand{ 32 * 32 * 4 },					///batch size (number of random rays per gradient step) must be < H * W
		PrecorpIters{ 0 },						///number of steps to train on central crops
		NIters {50000},
		LRateDecay{ 250 },						///exponential learning rate decay (in 1000 steps)
		//logging / saving options
		IPrint{ 100 },			///frequency of console printout and metric loggin
		IImg{ 500 },				///frequency of tensorboard image logging
		IWeights{ 10000 },	///frequency of weight ckpt saving
		ITestset{ 50000 },	///frequency of testset saving
		IVideo{ 50000 };		///frequency of render_poses video saving
	bool ReturnRaw{ false };
	float RenderFactor{ 0 },
		PrecorpFrac{ 0.5f };
};


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
class NeRFExecutor {
protected:
	TEmbedder ExecutorEmbedder = nullptr;
	TEmbedDirs ExecutorEmbeddirs = nullptr;
	TNeRF Model = nullptr,
		ModelFine = nullptr;
	std::vector<torch::Tensor> GradVars;
	std::unique_ptr<torch::optim::Adam> Optimizer;
	int Start = 0,
		NImportance{ 0 };
	float LearningRate;
	torch::Device Device;
	bool UseViewDirs;									//use full 5D input instead of 3D
public:
	NeRFExecutor(
		const int net_depth = 8,				//layers in network 8 for classic NeRF, 2 for HashNeRF
		const int net_width = 256,			//channels per layer 256 for classic NeRF, 64 for HashNeRF
		const int multires = 10,
		const bool use_viewdirs = true,	//use full 5D input instead of 3D. Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		const int multires_views = 4,		//log2 of max freq for positional encoding (2D direction)
		const int n_importance = 0,			//number of additional fine samples per ray
		const int net_depth_fine = 8,		//layers in fine network 8 for classic NeRF, 2 for HashNeRF
		const int net_width_fine = 256,	//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
		const int num_layers_color = 4,				//for color part of the HashNeRF
		const int hidden_dim_color = 64,			//for color part of the HashNeRF
		const int num_layers_color_fine = 4,	//for color part of the HashNeRF
		const int hidden_dim_color_fine = 64,	//for color part of the HashNeRF
		torch::Tensor bounding_box = torch::tensor({ 0.f, 0.f, 0.f, 1.f, 1.f, 1.f }),
		const int n_levels = 16,
		const int n_features_per_level = 2,
		const int log2_hashmap_size = 19,
		const int base_resolution = 16,
		const int finest_resolution = 512,
		torch::Device device = torch::kCUDA,
		const float learning_rate = 5e-4,
		std::filesystem::path ft_path = ""
	);

	///
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> RenderPath(
		const std::vector <torch::Tensor>& render_poses,
		int h,
		int w,
		float focal,
		torch::Tensor k,
		const int n_samples,
		const int chunk = 1024 * 32,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
		const int net_chunk = 1024 * 64,				///number of pts sent through network in parallel, decrease if running out of memory
		const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
		const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
		const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		const int n_importance = 0,							///Number of additional times to sample along each ray.
		const bool white_bkgr = false,					///If True, assume a white background.
		const float raw_noise_std = 0.,
		std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		const bool ndc = true,						///If True, represent ray origin, direction in NDC coordinates.
		const float near = 0.,						///float or array of shape[batch_size].Nearest distance for a ray.
		const float far = 1.,							///float or array of shape[batch_size].Farthest distance for a ray.
		const bool use_viewdirs = false,	///If True, use viewing direction of a point in space in model.
		torch::Tensor c2w_staticcam = torch::Tensor(),			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
		//torch::Tensor gt_imgs = torch::Tensor(),
		const std::filesystem::path savedir = "",
		const float render_factor = 0
	);

	///
	void Train(NeRFExecutorTrainParams &params);
};

///if using hashed for xyz, use SH for views
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF> :: NeRFExecutor(
	const int net_depth /*= 8, num_layers = 2*/,
	const int net_width /*= 256, hidden_dim = 64*/,
	const int multires /*= 10, -*/,
	const bool use_viewdirs /*= true*/,
	const int multires_views /*= 4*/,
	const int n_importance /*= 0*/,
	const int net_depth_fine /*= 8, num_layers_fine = 2*/,
	const int net_width_fine /*= 256, hidden_dim_fine = 64*/,
	const int num_layers_color /*= 4*/,
	const int hidden_dim_color /*= 64*/,
	const int num_layers_color_fine /*= 4*/,
	const int hidden_dim_color_fine /*= 64*/,
	torch::Tensor bounding_box,
	const int n_levels /*= 16*/,
	const int n_features_per_level /*= 2*/,
	const int log2_hashmap_size /*= 19*/,
	const int base_resolution /*= 16*/,
	const int finest_resolution /*= 512*/,
	torch::Device device /*= torch::kCUDA*/,
	const float learning_rate /*= 5e-4*/,
	std::filesystem::path ft_path /*= ""*/
) : Device(device), NImportance(n_importance), UseViewDirs(use_viewdirs), LearningRate(learning_rate)
{
	int input_ch,
		input_ch_views = 0;
	if constexpr (std::is_same_v<TEmbedder, Embedder>)
		ExecutorEmbedder = Embedder("embedder", multires);
	if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
		ExecutorEmbedder = HashEmbedder("embedder", bounding_box.to(Device), n_levels, n_features_per_level, log2_hashmap_size, base_resolution, finest_resolution);

	ExecutorEmbedder->to(device);
	input_ch = ExecutorEmbedder->GetOutputDims();
	auto embp = ExecutorEmbedder->parameters();
	GradVars.insert(GradVars.end(), std::make_move_iterator(embp.begin()), std::make_move_iterator(embp.end()));		//!!!
	
	if (use_viewdirs)
	{
		if constexpr (std::is_same_v<TEmbedDirs, Embedder>)
			ExecutorEmbeddirs = Embedder("embeddirs", multires_views);
		if constexpr (std::is_same_v<TEmbedDirs, HashEmbedder>)   //!!!if using hashed for xyz, use SH for views
			ExecutorEmbeddirs = SHEncoder("embeddirs", 3, 4);
		if constexpr (std::is_same_v<TEmbedDirs, SHEncoder>)
			ExecutorEmbeddirs = SHEncoder("embeddirs", 3, 4);
		input_ch_views = ExecutorEmbeddirs->GetOutputDims();
		ExecutorEmbeddirs->to(device);
	}

	int output_ch = (n_importance > 0) ? 5 : 4;
	const std::set<int> skips = std::set<int>{ 4 };

	if constexpr (std::is_same_v<TNeRF, NeRF>)
		Model = NeRF(net_depth, net_width, input_ch, input_ch_views, output_ch, skips, use_viewdirs, "model");

	if constexpr (std::is_same_v<TNeRF, NeRFSmall>)
	{
		int geo_feat_dim = 15;
		Model = NeRFSmall(net_depth, net_width, geo_feat_dim, num_layers_color, hidden_dim_color, input_ch, input_ch_views, "model");
	}

	Model->to(device);
	auto mp = Model->parameters();
	GradVars.insert(GradVars.end(), std::make_move_iterator(mp.begin()), std::make_move_iterator(mp.end()));		//!!!

	for (auto &k : Model->named_parameters())
		std::cout << k.key() << std::endl;
	std::cout << "Model params count: " << Trainable::ParamsCount(Model) << std::endl;

	if (n_importance > 0)
	{
		if constexpr (std::is_same_v<TNeRF, NeRF>)
			ModelFine = NeRF(net_depth_fine, net_width_fine, input_ch, input_ch_views, output_ch, skips, use_viewdirs, "model_fine");
		
		if constexpr (std::is_same_v<TNeRF, NeRFSmall>)
		{
			int geo_feat_dim = 15;
			ModelFine = NeRFSmall(net_depth_fine, net_width_fine, geo_feat_dim, num_layers_color_fine, hidden_dim_color_fine, input_ch, input_ch_views, "model_fine");
		}

		ModelFine->to(device);
		auto temp = ModelFine->parameters();
		GradVars.insert(GradVars.end(), std::make_move_iterator(temp.begin()), std::make_move_iterator(temp.end()));

		for (auto& k : ModelFine->named_parameters())
			std::cout << k.key() << std::endl;
		std::cout << "ModelFine params count: " << Trainable::ParamsCount(ModelFine) << std::endl;
	}

	if constexpr (std::is_same_v<TNeRF, NeRF>)
		Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(learning_rate).eps(1e-7)/*.weight_decay(0.001)*/.betas(std::make_tuple(0.9, 0.999)));

	if constexpr (std::is_same_v<TNeRF, NeRFSmall>)	//!!!RAdam
		//torch::optim::SGD generator_optimizer(generator->parameters(), torch::optim::SGDOptions(1e-4).weight_decay(0.001));
		//Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(learning_rate).eps(1e-8)/*.weight_decay(1e-6)*/.betas(std::make_tuple(0.9, 0.999)));
		Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(learning_rate).eps(1e-15).betas(std::make_tuple(0.9, 0.99)));

	if (/*Проверить наличие файлов*/
		std::filesystem::exists(ft_path / "start_checkpoint.pt") &&
		std::filesystem::exists(ft_path / "optimizer_checkpoint.pt") &&
		std::filesystem::exists(ft_path / "model_checkpoint.pt") &&
		(ModelFine.is_empty() || (!ModelFine.is_empty() && std::filesystem::exists(ft_path / "model_fine_checkpoint.pt")))
	) {
		if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
			if (std::filesystem::exists(ft_path / "embedder_checkpoint.pt"))
				torch::load(ExecutorEmbedder, (ft_path / "embedder_checkpoint.pt").string());

		std::cout << "restoring parameters from checkpoint..." << std::endl;
		torch::Tensor temp;
		torch::load(temp, (ft_path / "start_checkpoint.pt").string());
		Start = temp.item<float>();
		torch::load(*Optimizer.get(), (ft_path / "optimizer_checkpoint.pt").string());
		torch::load(Model, (ft_path / "model_checkpoint.pt").string());
		if (!ModelFine.is_empty())
			torch::load(ModelFine, (ft_path / "model_fine_checkpoint.pt").string());
	} else {
		if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
			ExecutorEmbedder->Initialize();//Trainable::Initialize(ExecutorEmbedder);//ExecutorEmbedder->Initialize();
		Trainable::Initialize(Model);
		if (n_importance > 0)
			Trainable::Initialize(ModelFine);
	}
}			//NeRFExecutor :: NeRFExecutor


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF> :: RenderPath(
	const std::vector <torch::Tensor> &render_poses,
	int h,
	int w,
	float focal,
	torch::Tensor k,
	const int n_samples,
	const int chunk /*= 1024 * 32*/,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
	const int net_chunk /*= 1024 * 64*/,				///number of pts sent through network in parallel, decrease if running out of memory
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

	for (auto &c2w : render_poses) //(int i = 0; i < render_poses.size(); i++)
	{
		RenderResult render_result = Render(h, w, k,
			Model,
			ExecutorEmbedder,
			ExecutorEmbeddirs,
			ModelFine,
			n_samples,
			chunk,
			net_chunk,
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
		//normalize depth to[0, 1]
		render_result.Outputs1.DepthMap = (render_result.Outputs1.DepthMap - near) / (far - near);

		if (!savedir.empty())
		{
			cv::imwrite((savedir / (std::to_string(rgbs.size() - 1) + ".png")).string(), TorchTensorToCVMat(rgbs.back()));
			cv::imwrite((savedir / ("disp_" + std::to_string(disps.size() - 1) + ".png")).string(), TorchTensorToCVMat(disps.back()));
			cv::imwrite((savedir / ("depth_" + std::to_string(disps.size() - 1) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DepthMap));
		}
	}
	return std::make_pair(rgbs, disps);
}			//NeRFExecutor :: RenderPath

///
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF> :: Train(NeRFExecutorTrainParams &params)
{
	torch::Tensor k;
	CompactData data;

	//Загрузить данные
	if (params.DatasetType == DatasetType::BLENDER)
	{
		data = load_blender_data(params.DataDir, params.HalfRes, params.TestSkip);
		std::cout << "Loaded blender " << data.Imgs.size() << " " << data.RenderPoses.size() << " " << data.H << " " << data.W << " " << data.Focal << " " << params.DataDir;
		std::cout << "data.Imgs[0]: " << data.Imgs[0].sizes() << " " << data.Imgs[0].device() << " " << data.Imgs[0].type() << std::endl;
		//i_train, i_val, i_test = i_split;

		for (auto &img : data.Imgs)
		{
			if (params.WhiteBkgr)
				img = img.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) }) * img.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }) + (1. - img.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }));
			else
				img = img.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) });
		}
	}

	//Задать калибровки камеры
	if (!k.defined())
	{
		float kdata[] = { data.Focal, 0, 0.5f * data.W,
			0, data.Focal, 0.5f * data.H,
			0, 0, 1 };
		k = torch::from_blob(kdata, { 3, 3 });
	}

	//NDC only good for LLFF - style forward facing data
	if (params.DatasetType != DatasetType::LLFF)
	{
		std::cout << "Not ndc!" << std::endl;
		params.Ndc = false;
	}

	////Если требуется то будем рендерить в тестовых позициях см еще код ниже
	//if (params.RenderTest)
	//	render_poses = np.array(poses[i_test взять из data.SplitsIdx])
	//Create log dir and copy the config file ....

	int global_step = Start;

	//Move testing data to GPU
	for (auto &it : data.RenderPoses)
		it = it.to(Device);

	//Short circuit if only rendering out from trained model
	if (params.RenderOnly)
	{
		std::cout << "RENDER ONLY" << std::endl;
		torch::NoGradGuard no_grad;
		//if (params.RenderTest)
		//{ 
		//	//render_test switches to test poses
		//	data.Imgs = data.Imgs[i_test];
		//}	else {
			//Default is smoother render_poses path
		for (auto &it : data.Imgs)
			it = torch::Tensor();
		//}

		auto test_save_dir = params.BaseDir / (std::string("renderonly_") + /*params.RenderTest ? "test_" :*/ "path_" + std::to_string(Start));
		if (!std::filesystem::exists(test_save_dir))	std::filesystem::create_directories(test_save_dir);
		std::cout << "test_poses_shape: " << data.RenderPoses.size() << std::endl;

		//test args: perturb = False, raw_noise_std = 0.
		auto [rgbs, disps] = RenderPath(data.RenderPoses, data.H, data.W, data.Focal, k, params.NSamples, params.Chunk, params.NetChunk,
			params.ReturnRaw, params.LinDisp, 0, NImportance, params.WhiteBkgr, 0., { torch::Tensor(), torch::Tensor() },
			params.Ndc, data.Near, data.Far, UseViewDirs, /*c2w_staticcam*/torch::Tensor(), /*data.Imgs,*/ test_save_dir, params.RenderFactor
		);

		cv::VideoWriter video((test_save_dir / "video.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(data.W, data.H), true);
		for (auto &img : rgbs)
			video.write(TorchTensorToCVMat(img));
		video.release();

		std::cout << "Done rendering " << test_save_dir << std::endl;
		return;
	}		//if (params.RenderOnly)

	//Prepare raybatch tensor if batching random rays
	int i_batch;
	torch::Tensor rays_rgb;
	if (!params.NoBatching)
	{
		std::cout << "get rays" << std::endl;
		std::vector<torch::Tensor> rays;

		for (auto pose : data.Poses)
		{
			auto [rays_o, rays_d] = GetRays(data.H, data.W, k, pose.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }));
			rays.emplace_back(torch::cat({ rays_o, rays_d }, 0/*?*/));	//[N, ro + rd, H, W, 3]
		}
		rays_rgb = torch::stack(rays, 0);								//[N, ro + rd, H, W, 3]
		std::cout << rays_rgb.sizes() << std::endl;
		rays_rgb = torch::cat({ rays_rgb, torch::stack(data.Imgs, 0) }, 1);				//[N, ro + rd + rgb, H, W, 3]
		std::cout << rays_rgb.sizes() << std::endl;

		//numpy.transpose is same as torch.permute
		rays_rgb = rays_rgb.permute({ 0, 2, 3, 1, 4 });			//[N, H, W, ro + rd + rgb, 3]

		rays_rgb = rays_rgb.index({ torch::indexing::Slice(torch::indexing::None, data.SplitsIdx[0]) });		//train images only
		//A single dimension may be - 1, in which case it’s inferred from the remaining dimensionsand the number of elements in input.
		rays_rgb = torch::reshape(rays_rgb, { -1/*?*/, 3, 3 });		//[(N - 1) * H * W, ro + rd + rgb, 3]
		std::cout << "shuffle rays" << std::endl;
		rays_rgb = rays_rgb.index({ torch::randperm(rays_rgb.sizes()[0]) });
		std::cout << "done " << rays_rgb.sizes() << std::endl;
		i_batch = 0;
	}	//if (!params.NoBatching)

	//Move training data to GPU
	torch::Tensor images;
	if (!params.NoBatching)
	{
		images = torch::stack(data.Imgs, 0).to(Device);
		rays_rgb = rays_rgb.to(Device);
	}
	torch::Tensor poses = torch::stack(data.Poses, 0).to(Device);

	int n_iters = params.NIters + 1;
	std::cout << "Begin" << std::endl;
	for (auto it : data.Splits)
		std::cout << it << std::endl;
	for (auto it : data.SplitsIdx)
		std::cout << it << std::endl;

	for (int i = this->Start + 1; i < n_iters; i++)
	{
		auto time0 = std::chrono::steady_clock::now();

		std::pair<torch::Tensor, torch::Tensor> batch_rays;		//{rays_o, rays_d}
		torch::Tensor batch,
			target_s;
		if (!params.NoBatching)
		{
			//Random over all images
			batch = rays_rgb.index({ torch::indexing::Slice(i_batch, i_batch + params.NRand) });
			batch = torch::transpose(batch, 0, 1);
			batch_rays.first = batch.index({ torch::indexing::Slice(torch::indexing::None, 2) });
			auto temp = torch::split(batch_rays.first, int(batch_rays.first.size(0) / 2), /*dim*/0);
			batch_rays.first = temp[0];
			batch_rays.second = temp[1];
			target_s = batch.index({ 2 });
			i_batch += params.NRand;
			if (i_batch > rays_rgb.sizes()[0])
			{
				std::cout << "Shuffle data after an epoch!" << std::endl;
				rays_rgb = rays_rgb.index({ torch::randperm(rays_rgb.sizes()[0]) });
				i_batch = 0;
			}
		}
		else {
			//Random from one image
			int img_i = RandomInt() % (data.SplitsIdx[0] /* + 1 ? */);
			auto target = data.Imgs[img_i].squeeze(0).to(Device);		//1, 800, 800, 4 -> 800, 800, 4
			auto pose = poses.index({ img_i, torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) });
			torch::Tensor coords;
			if (params.NRand != 0)
			{
				auto [rays_o, rays_d] = GetRays(data.H, data.W, k, pose);

				if (i < params.PrecorpIters)
				{
					int dh = int(data.H / 2 * params.PrecorpFrac);
					int dw = int(data.W / 2 * params.PrecorpFrac);

					coords = torch::stack(
						torch::meshgrid({
							torch::linspace(data.H / 2 - dh, data.H / 2 + dh - 1, 2 * dh, torch::kLong),
							torch::linspace(data.W / 2 - dw, data.W / 2 + dw - 1, 2 * dw, torch::kLong)
							}), -1);
					if (i == this->Start + 1)
						std::cout << "[Config] Center cropping of size " << 2 * dh << "x" << 2 * dw << " is enabled until iter " << params.PrecorpIters << std::endl;
				} else {
					coords = torch::stack(torch::meshgrid({ torch::linspace(0, data.H - 1, data.H, torch::kLong), torch::linspace(0, data.W - 1, data.W, torch::kLong) }), -1);  //(H, W, 2)
				}

				coords = torch::reshape(coords, { -1, 2 });		//(H * W, 2)

				torch::Tensor select_inds = torch::randperm(coords.size(0)/*, torch::kLong*/).slice(0, 0, params.NRand);		// (N_rand,)
				torch::Tensor select_coords = coords.index({ select_inds }).to(Device);		///!!!Вот так можно вытащить данные из по индексам записанным в другой тензор, здесь по нулевому измерению, но построить индекс можно по любому набору измерений
				rays_o = rays_o.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({ torch::indexing::Slice(), 1 }) });		// (N_rand, 3)
				rays_d = rays_d.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });		// (N_rand, 3)
				//batch_rays = torch::stack({ rays_o, rays_d }, /*dim=*/0);
				batch_rays.first = rays_o;
				batch_rays.second = rays_d;
				target_s = target.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });  // (N_rand, 3)
			}
		}		// else (!params.NoBatching)

		//  Core optimization loop  

		RenderResult rgb_disp_acc_extras = Render(
			data.H, data.W, k, Model, ExecutorEmbedder, ExecutorEmbeddirs, ModelFine,
			params.NSamples, params.Chunk, params.NetChunk, params.ReturnRaw, params.LinDisp,
			0,		//0. or 1. If non - zero, each ray is sampled at stratified random points in time.
			NImportance,
			params.WhiteBkgr,
			0.,
			batch_rays,  //либо rays либо pose c2w
			torch::Tensor(),
			params.Ndc,
			data.Near, data.Far, UseViewDirs,
			torch::Tensor()
		);

		Optimizer->zero_grad();
		auto img_loss = torch::mse_loss(rgb_disp_acc_extras.Outputs1.RGBMap, target_s);
		if (params.ReturnRaw)
			auto trans = rgb_disp_acc_extras.Raw.index({ "...", -1 });		//последний элемент последнего измерения тензора
		auto loss = img_loss;
		auto psnr = -10. * torch::log(img_loss) / torch::log(torch::full({ 1/*img_loss.sizes()*/ }, /*value=*/10.f)).to(Device);

		torch::Tensor img_loss0,
			psnr0;
		if (rgb_disp_acc_extras.Outputs0.RGBMap.defined() && (rgb_disp_acc_extras.Outputs0.RGBMap.numel() != 0))
		{
			img_loss0 = torch::mse_loss(rgb_disp_acc_extras.Outputs0.RGBMap, target_s);
			loss = loss + img_loss0;
			psnr0 = -10. * torch::log(img_loss0) / torch::log(torch::full({ 1/*img_loss.sizes()*/ }, /*value=*/10.f)).to(Device);
		}

		//add Total Variation loss
		if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
			if (i < 2000)
			{
				const float tv_loss_weight = 1e-6;
				for (int level = 0; level < ExecutorEmbedder->GetNLevels(); level++)
				{
					loss += tv_loss_weight * TotalVariationLoss(
						ExecutorEmbedder,
						Device,
						ExecutorEmbedder->GetBaseResolution(),
						ExecutorEmbedder->GetFinestResolution(),
						level,
						ExecutorEmbedder->GetLog2HashmapSize(),
						ExecutorEmbedder->GetNLevels()
					).to(loss.device());
				}
			}

		loss.backward();
		Optimizer->step();

		//Update learning rate
		float decay_rate = 0.1f;
		int decay_steps = params.LRateDecay * 1000;
		auto new_lrate = LearningRate * powf(decay_rate, (float)global_step / decay_steps);
		for (auto param_group : Optimizer->param_groups())
			param_group.options().set_lr(new_lrate);
		//std::cout << "Step: " << global_step << " of "<< n_iters << " Loss: " << loss << " Time : " << std::chrono::duration <double, std::milli>(std::chrono::steady_clock::now() - time0).count() << " ms" << std::endl;

		//Rest is logging
		if (i % params.IWeights == 0)
		{
			auto path = params.BaseDir; /* / (std::to_string(i) + ".tar") */;
			if (!std::filesystem::exists(path))	std::filesystem::create_directories(path);
			if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
				torch::save(ExecutorEmbedder, (path / "embedder_checkpoint.pt").string());
			torch::save(torch::full({ 1 }, /*value=*/global_step), (path / "start_checkpoint.pt").string());
			torch::save(*Optimizer.get(), (path / "optimizer_checkpoint.pt").string());
			torch::save(Model, (path / "model_checkpoint.pt").string());
			if (!ModelFine.is_empty())
				torch::save(ModelFine, (path / "model_fine_checkpoint.pt").string());
			std::cout << "Saved checkpoints at " << path.string() << std::endl;
		}

		if ((i % params.IVideo == 0) && i > 0)
		{
			torch::NoGradGuard no_grad;

			//test args: perturb = False, raw_noise_std = 0.
			auto [rgbs, disps] = RenderPath(data.RenderPoses, data.H, data.W, data.Focal, k, params.NSamples, params.Chunk, params.NetChunk,
				params.ReturnRaw, params.LinDisp, 0, NImportance, params.WhiteBkgr, 0., { torch::Tensor(), torch::Tensor() },
				params.Ndc, data.Near, data.Far, UseViewDirs, /*c2w_staticcam*/torch::Tensor()/*, data.Imgs,*/ /*params.TestSaveDir, params.RenderFactor*/
			);
			std::cout << "Done, saving " << rgbs.size() << " " << rgbs[0].sizes() << " " << disps.size() << " " << disps[0].sizes() << std::endl;
			auto path = params.BaseDir; /* / ("spiral_" + std::to_string(i)) */;
			if (!std::filesystem::exists(path))	std::filesystem::create_directories(path);

			cv::VideoWriter video((params.BaseDir / "rgb.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(data.W, data.H), true);
			for (auto &img : rgbs)
				video.write(TorchTensorToCVMat(img));
			video.release();

			cv::VideoWriter video2((params.BaseDir / "disp.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(data.W, data.H), true);
			for (auto &img : disps)
			{
				float min_val = img.min().item<float>();
				float max_val = img.max().item<float>();
				// Нормализуем тензор
				img = (img - min_val) / (max_val - min_val);
				video2.write(TorchTensorToCVMat(img));
			}
			video2.release();
		}

		if ((i % params.ITestset == 0) && (i > 0) && !params.TestSkip)		//если TestSkip то просто не подгружены тестовые данные 
		{
			auto test_save_dir = params.BaseDir; /* / ("testset_" + std::to_string(i)) */;
			if (!std::filesystem::exists(test_save_dir))	std::filesystem::create_directories(test_save_dir);
			//Есть torch::Tensor poses и есть std::vector<torch::Tensor> data.Poses
			std::cout << "poses shape " << poses.sizes() << "; test poses count " << data.SplitsIdx[2] << std::endl;   //	std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			std::cout << "poses shape " << data.Poses[0].sizes() << "; test poses count " << data.SplitsIdx[2] << std::endl;   //	std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			torch::NoGradGuard no_grad;

			std::vector<torch::Tensor> test_poses,
				test_imgs;
			for (int k = data.SplitsIdx[0] + data.SplitsIdx[1]; k < data.SplitsIdx[0] + data.SplitsIdx[1] + data.SplitsIdx[2]; k++)		 //	std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			{
				//test_imgs.push_back(data.Imgs[k]);
				test_poses.push_back(data.Poses[k].to(Device));
			}
			auto [rgbs, disps] = RenderPath(test_poses/*poses[i_test]).to(device)*/, data.H, data.W, data.Focal, k, params.NSamples, params.Chunk, params.NetChunk,
				params.ReturnRaw, params.LinDisp, 0, NImportance, params.WhiteBkgr, 0., { torch::Tensor(), torch::Tensor() },
				params.Ndc, data.Near, data.Far, UseViewDirs, /*c2w_staticcam*/torch::Tensor(), /*test_imgs,*/ test_save_dir, params.RenderFactor
			);
			std::cout << "Saved test set" << std::endl;
		}

		if (i % params.IPrint == 0)
			std::cout << "[TRAIN] Iter: " << i << " of " << n_iters << " lr = " << new_lrate << " Loss: " << loss.item() << " PSNR: " << psnr.item() << std::endl;

		global_step++;
	}	//for (int i = this->Start + 1; i < n_iters; i++)
}			//NeRFExecutor :: Train