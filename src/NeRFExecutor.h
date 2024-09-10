#pragma once

#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "CuSHEncoder.h"
#include "CuHashEmbedder.h"
#include "PyramidEmbedder.h"
#include "NeRF.h"
#include "LeRF.h"
#include "Sampler.h"
#include "BaseNeRFRenderer.h"
#include "TRandomInt.h"

#include <set>
#include <filesystem>
#include <chrono>
#include <memory>

#include <torch/script.h>

enum class DatasetType { LLFF, BLENDER, LINEMOD, DEEP_VOXELS };

struct NeRFExecutorParams {
	int net_depth{ 8 },				//layers in network 8 for classic NeRF, 2 for HashNeRF
		net_width{ 256 },				//channels per layer 256 for classic NeRF, 64 for HashNeRF
		multires{ 10 },
		multires_views{ 4 },		//log2 of max freq for positional encoding (2D direction)
		n_importance{ 0 },			//number of additional fine samples per ray
		net_depth_fine{ 8 },		//layers in fine network 8 for classic NeRF, 2 for HashNeRF
		net_width_fine{ 256 },	//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
		num_layers_color{ 4 },				//for color part of the HashNeRF
		hidden_dim_color{ 64 },				//for color part of the HashNeRF
		num_layers_color_fine{ 4 },		//for color part of the HashNeRF
		hidden_dim_color_fine{ 64 },	//for color part of the HashNeRF
		num_layers_normals{ 3 },
		hidden_dim_normals{ 64 },
		geo_feat_dim{ 15 };
	//torch::Tensor bounding_box = torch::tensor({ -4.f, -4.f, -4.f, 4.f, 4.f, 4.f });
	bool use_viewdirs{ true },	//use full 5D input instead of 3D. Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		calculate_normals{ false },
		use_pred_normal{ true },	//whether to use predicted normals
		use_lerf{ true };
	int n_levels{ 16 },											//for color density embedder
		n_features_per_level{ 2 },						//for color density embedder
		log2_hashmap_size{ 19 },							//for color density embedder
		base_resolution{ 16 },								//for color density embedder
		finest_resolution{ 512 },							//for color density embedder
		n_levels_le = n_levels - 2,																		//for language embedder
		n_features_per_level_le = n_features_per_level,								//for language embedder
		log2_hashmap_size_le = log2_hashmap_size - 3,									//for language embedder
		base_resolution_le = base_resolution,													//for language embedder
		finest_resolution_le = finest_resolution/4,										//for language embedder
		clip_input_img_size{ 336 },	//Input RuClip model size
		num_layers_le{ 3 },					//Language embedder head params
		hidden_dim_le{ 64 },				//Language embedder head params
		lang_embed_dim{ 768 };			//Language embedder head params
	torch::Device device{ torch::kCUDA };
	float learning_rate{ 5e-4 };
	std::filesystem::path ft_path,
		path_to_clip,			//Path to RuClip model
		path_to_bpe;			//Path to tokenizer
	std::string lerf_positives;
	std::vector<std::string> lerf_negatives;
};

struct NeRFExecutorTrainParams {
	std::filesystem::path //DataDir,	///input data directory
		PyramidClipEmbeddingSaveDir,
		BaseDir;											///where to store ckpts and logs
	bool TestSkip{ false },						///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
		RenderOnly{ false },					///do not optimize, reload weights and render out render_poses path
		Ndc{ true },									///use normalized device coordinates (set for non-forward facing scenes)
		LinDisp{ false },							///sampling linearly in disparity rather than depth
		NoBatching{ true };						///only take random rays from 1 image at a time
	int Chunk{ 1024 * 32 },					///number of rays processed in parallel, decrease if running out of memory не влияет на качество
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
	NeRFExecutorParams Params;
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
	CLIP Clip = nullptr;
	std::shared_ptr<RuCLIPProcessor> ClipProcessor = nullptr;
	PyramidEmbedding PyramidClipEmbedding;
public:
	NeRFExecutor(const NeRFExecutorParams &params) : Params(params), Device(params.device), NImportance(params.n_importance), UseViewDirs(params.use_viewdirs), LearningRate(params.learning_rate){};
	void Initialize(const NeRFExecutorParams &params, torch::Tensor bounding_box);
	
	CompactData LoadData (
		const std::filesystem::path &basedir,
		DatasetType dataset_type = DatasetType::BLENDER,
		bool half_res = false,				///load blender synthetic data at 400x400 instead of 800x800
		bool test_skip = false,				///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
		bool white_bkgr = false				///set to render synthetic data on a white bkgd (always use for dvoxels)
	)	{
		CompactData data;
		//Загрузить данные
		if (dataset_type == DatasetType::BLENDER)
		{
			data = load_blender_data(basedir, half_res, test_skip);
			//!!!Тут K and BoundingBox поломано, поэтому ниже все повторно заполняется
			std::cout << "Loaded blender " << data.Imgs.size() << " " << data.RenderPoses.size() << " " << data.H << " " << data.W << " " << data.Focal << " " << basedir;
			std::cout << "data.Imgs[0]: " << data.Imgs[0].sizes() << " " << data.Imgs[0].device() << " " << data.Imgs[0].type() << std::endl;
			//i_train, i_val, i_test = i_split;

			for (auto &img : data.Imgs)
			{
				if (white_bkgr)
					img = img.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) }) * img.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }) + (1. - img.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }));
				else
					img = img.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) });
			}
		}

		///!!!Сделать нормальное копирование, так как эти тензоры заполняются внутри
		float kdata[] = { data.Focal, 0, 0.5f * data.W,
			0, data.Focal, 0.5f * data.H,
			0, 0, 1 };
		data.K = torch::from_blob(kdata, { 3, 3 }, torch::kFloat32);
		//data.K = GetCalibrationMatrix(data.Focal, data.W, data.H).clone().detach();
		data.BoundingBox = GetBbox3dForObj(data).clone().detach();
		
		//Move testing data to GPU
		for (auto &it : data.RenderPoses)
			it = it.to(Device);

		return data;
	}


	///
	RenderResult RenderView(
		const torch::Tensor render_pose,	//rays?  	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() };			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		int w,
		int h,
		const torch::Tensor k,
		const RenderParams &rparams
	);

	///
	void RenderPath(
		const std::vector <torch::Tensor> &render_poses,
		int h,
		int w,
		float focal,
		torch::Tensor k,
		const RenderParams &rparams,
		std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		torch::Tensor c2w_staticcam = torch::Tensor(),			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
		//torch::Tensor gt_imgs = torch::Tensor(),
		const std::filesystem::path savedir = ""
	);

	///
	void Train(const CompactData &data, NeRFExecutorTrainParams &params);
	NeRFExecutorParams GetParams(){return Params;};
	torch::Tensor GetEmbedderBoundingBox(){return ExecutorEmbedder->GetBoundingBox();};
};				//NeRFExecutor


///if using hashed for xyz, use SH for views
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF> :: Initialize(const NeRFExecutorParams &params, torch::Tensor bounding_box)
{
	int input_ch,
		input_ch_views = 0;
	if constexpr (std::is_same_v<TEmbedder, Embedder>)
		ExecutorEmbedder = Embedder("embedder", params.multires);
	if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
		ExecutorEmbedder = HashEmbedder("embedder", bounding_box.to(Device), params.n_levels, params.n_features_per_level, params.log2_hashmap_size, params.base_resolution, params.finest_resolution);
	if constexpr (std::is_same_v<TEmbedder, CuHashEmbedder>)
		ExecutorEmbedder = CuHashEmbedder("embedder", bounding_box.to(Device), params.n_levels, params.n_features_per_level, params.log2_hashmap_size, params.base_resolution, params.finest_resolution);
	if constexpr (std::is_same_v<TEmbedder, LeRFEmbedder<CuHashEmbedder>>)
		ExecutorEmbedder = LeRFEmbedder<CuHashEmbedder>("embedder", bounding_box.to(Device), params.n_levels, params.n_features_per_level, params.log2_hashmap_size, params.base_resolution, params.finest_resolution,
			params.n_levels_le, params.n_features_per_level_le, params.log2_hashmap_size_le, params.base_resolution_le, params.finest_resolution_le);

	ExecutorEmbedder->to(params.device);
	input_ch = ExecutorEmbedder->GetOutputDims();
	auto embp = ExecutorEmbedder->parameters();
	GradVars.insert(GradVars.end(), std::make_move_iterator(embp.begin()), std::make_move_iterator(embp.end()));		//!!!
	
	for (auto &k : ExecutorEmbedder->named_parameters())
		std::cout << k.key() << std::endl;
	std::cout << "ExecutorEmbedder params count: " << Trainable::ParamsCount(ExecutorEmbedder) << std::endl;

	if (params.use_viewdirs)
	{
		if constexpr (std::is_same_v<TEmbedDirs, Embedder>)
			ExecutorEmbeddirs = Embedder("embeddirs", params.multires_views);
		if constexpr (std::is_same_v<TEmbedDirs, HashEmbedder>)   //!!!if using hashed for xyz, use SH for views
			ExecutorEmbeddirs = SHEncoder("embeddirs", 3, 4);
		if constexpr (std::is_same_v<TEmbedDirs, SHEncoder>)
			ExecutorEmbeddirs = SHEncoder("embeddirs", 3, 4);
		if constexpr (std::is_same_v<TEmbedDirs, CuSHEncoder>)
			ExecutorEmbeddirs = CuSHEncoder("embeddirs", 3, params.multires_views);
		input_ch_views = ExecutorEmbeddirs->GetOutputDims();
		ExecutorEmbeddirs->to(params.device);
	}

	int output_ch = (params.n_importance > 0) ? 5 : 4;
	const std::set<int> skips = std::set<int>{ 4 };

	if constexpr (std::is_same_v<TNeRF, NeRF>)
		Model = NeRF(params.net_depth, params.net_width, input_ch, input_ch_views, output_ch, skips, params.use_viewdirs, "model");

	if constexpr (std::is_same_v<TNeRF, NeRFSmall>)
	{
		Model = NeRFSmall(
			params.net_depth,
			params.net_width,
			params.geo_feat_dim,
			params.num_layers_color,
			params.hidden_dim_color,
			(params.n_importance == 0) && params.use_pred_normal,		//В грубой сети не учим нормали, поэтому проверим нет ли тонкой сети
			params.num_layers_normals,
			params.hidden_dim_normals,
			input_ch,
			input_ch_views,
			"model"
		);
	}

	if constexpr (std::is_same_v<TNeRF, LeRF>)
	{
		Model = LeRF(		
			params.net_depth,
			params.net_width,
			params.geo_feat_dim,
			params.num_layers_color,
			params.hidden_dim_color,
			(params.n_importance == 0) && params.use_pred_normal,		//В грубой сети не учим нормали, поэтому проверим нет ли тонкой сети
			params.num_layers_normals,
			params.hidden_dim_normals,
			input_ch,
			input_ch_views,
			(params.n_importance == 0)?params.num_layers_le:0,
			(params.n_importance == 0)?params.hidden_dim_le:0,
			(params.n_importance == 0)?params.lang_embed_dim:0,
			ExecutorEmbedder->GetOutputDimsLE(),/*input_ch_le,*/
			"model"
		);
	}

	Model->to(params.device);
	auto mp = Model->parameters();
	GradVars.insert(GradVars.end(), std::make_move_iterator(mp.begin()), std::make_move_iterator(mp.end()));		//!!!

	for (auto &k : Model->named_parameters())
		std::cout << k.key() << std::endl;
	std::cout << "Model params count: " << Trainable::ParamsCount(Model) << std::endl;

	if (params.n_importance > 0)
	{
		if constexpr (std::is_same_v<TNeRF, NeRF>)
			ModelFine = NeRF(params.net_depth_fine, params.net_width_fine, input_ch, input_ch_views, output_ch, skips, params.use_viewdirs, "model_fine");
		
		if constexpr (std::is_same_v<TNeRF, NeRFSmall>)
		{
			ModelFine = NeRFSmall(
				params.net_depth_fine,
				params.net_width_fine,
				params.geo_feat_dim,
				params.num_layers_color_fine,
				params.hidden_dim_color_fine,
				params.use_pred_normal,
				params.num_layers_normals,
				params.hidden_dim_normals,
				input_ch,
				input_ch_views,
				"model_fine"
			);
		}

		if constexpr (std::is_same_v<TNeRF, LeRF>)
		{
			ModelFine = LeRF(
				params.net_depth_fine,
				params.net_width_fine,
				params.geo_feat_dim,
				params.num_layers_color_fine,
				params.hidden_dim_color_fine,
				params.use_pred_normal,
				params.num_layers_normals,
				params.hidden_dim_normals,
				input_ch,
				input_ch_views,
				params.num_layers_le,
				params.hidden_dim_le,
				params.lang_embed_dim,
				ExecutorEmbedder->GetOutputDimsLE(),/*input_ch_le,*/
				"model_fine"
			);
		}

		ModelFine->to(params.device);
		auto temp = ModelFine->parameters();
		GradVars.insert(GradVars.end(), std::make_move_iterator(temp.begin()), std::make_move_iterator(temp.end()));

		for (auto& k : ModelFine->named_parameters())
			std::cout << k.key() << std::endl;
		std::cout << "ModelFine params count: " << Trainable::ParamsCount(ModelFine) << std::endl;
	}

	if constexpr (std::is_same_v<TNeRF, NeRF>)
		Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-7)/*.weight_decay(0.001)*/.betas(std::make_tuple(0.9, 0.999)));

	if constexpr (std::is_same_v<TNeRF, NeRFSmall> && std::is_same_v<TEmbedder, HashEmbedder>)	//!!!RAdam
		//torch::optim::SGD generator_optimizer(generator->parameters(), torch::optim::SGDOptions(1e-4).weight_decay(0.001));
		//Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(learning_rate).eps(1e-8)/*.weight_decay(1e-6)*/.betas(std::make_tuple(0.9, 0.999)));
		Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-15).betas(std::make_tuple(0.9, 0.99)));

	if constexpr (std::is_same_v<TNeRF, NeRFSmall> && std::is_same_v<TEmbedder, CuHashEmbedder>)
		Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-15).betas(std::make_tuple(0.9, 0.99)));

	if constexpr (std::is_same_v<TNeRF, LeRF>)//RAdam в оригинале LeRF
		Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-15).betas(std::make_tuple(0.9, 0.99)));
	
	if (/*Проверить наличие файлов*/
		std::filesystem::exists(params.ft_path / "start_checkpoint.pt") &&
		std::filesystem::exists(params.ft_path / "optimizer_checkpoint.pt") &&
		std::filesystem::exists(params.ft_path / "model_checkpoint.pt") &&
		(ModelFine.is_empty() || (!ModelFine.is_empty() && std::filesystem::exists(params.ft_path / "model_fine_checkpoint.pt")))
	) {
		//if constexpr (std::is_same_v<TEmbedder, HashEmbedder> || std::is_same_v<TEmbedder, CuHashEmbedder> || std::is_same_v<TEmbedder, LeRFEmbedder>)
		if (std::filesystem::exists(params.ft_path / "embedder_checkpoint.pt"))
			torch::load(ExecutorEmbedder, (params.ft_path / "embedder_checkpoint.pt").string());

		std::cout << "restoring parameters from checkpoint..." << std::endl;
		torch::Tensor temp;
		torch::load(temp, (params.ft_path / "start_checkpoint.pt").string());
		Start = temp.item<float>();
		torch::load(*Optimizer.get(), (params.ft_path / "optimizer_checkpoint.pt").string());
		torch::load(Model, (params.ft_path / "model_checkpoint.pt").string());
		if (!ModelFine.is_empty())
			torch::load(ModelFine, (params.ft_path / "model_fine_checkpoint.pt").string());
	} else {
		ExecutorEmbedder->Initialize();//Trainable::Initialize(ExecutorEmbedder);//ExecutorEmbedder->Initialize();
		Trainable::Initialize(Model);
		if (params.n_importance > 0)
			Trainable::Initialize(ModelFine);
	}

	if (Params.use_lerf)
	{
		std::cout << "Load clip from: " << params.path_to_clip << std::endl;
		Clip = FromPretrained(params.path_to_clip);
		Clip->to(params.device);
		
		std::cout << "Load processor from: " << params.path_to_bpe << std::endl;
		ClipProcessor = std::make_shared<RuCLIPProcessor>(
			params.path_to_bpe,
			params.clip_input_img_size,
			77,
			std::vector<double>({ 0.48145466, 0.4578275, 0.40821073 }),
			std::vector<double>({ 0.26862954, 0.26130258, 0.27577711 })
		);
	}
}			//NeRFExecutor :: Initialize


	///
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
RenderResult NeRFExecutor<TEmbedder, TEmbedDirs, TNeRF> :: RenderView(
	const torch::Tensor render_pose,	//rays?  	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() };			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	int w,
	int h,
	const torch::Tensor k,
	const RenderParams &rparams
){
	torch::Tensor k1 = k.clone().detach();
	if (rparams.RenderFactor != 0)
	{
		//Render downsampled for speed
		h = h / rparams.RenderFactor;
		w = w / rparams.RenderFactor;
		k1[0][0] = k1[0][0]/rparams.RenderFactor;
		k1[1][1] = k1[1][1]/rparams.RenderFactor;
		k1[0][2] = k1[0][2]/rparams.RenderFactor;
		k1[1][2] = k1[1][2]/rparams.RenderFactor;
	}

	return Render(h, w, k1,
		Model,
		ExecutorEmbedder,
		ExecutorEmbeddirs,
		ModelFine,
		rparams,
		/*Rays*/ {torch::Tensor(), torch::Tensor()},
		render_pose.to(Device).index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
		/*rparams.C2wStaticCam*/ torch::Tensor()
	);
};		//NeRFExecutor :: RenderView


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF> :: RenderPath(
	const std::vector <torch::Tensor> &render_poses,
	int h,
	int w,
	float focal,
	torch::Tensor k,
	const RenderParams &rparams,
	std::pair<torch::Tensor, torch::Tensor> rays /*= { torch::Tensor(), torch::Tensor() }*/,			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
	torch::Tensor c2w_staticcam /*= torch::Tensor()*/,			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
	//torch::Tensor gt_imgs = torch::Tensor(),
	const std::filesystem::path savedir /*= ""*/
) {
	//std::vector<torch::Tensor> rgbs,
	//	disps;

	if (rparams.RenderFactor != 0)
	{
		//Render downsampled for speed
		h = h / rparams.RenderFactor;
		w = w / rparams.RenderFactor;
		focal = focal / rparams.RenderFactor;
	}

	for (size_t i = 0; i < render_poses.size(); i++)		//(auto &c2w : render_poses) //
	{
		RenderResult render_result = std::move(Render(h, w, k,
			Model,
			ExecutorEmbedder,
			ExecutorEmbeddirs,
			ModelFine,
			rparams,
			rays,
			render_poses[i].index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
			c2w_staticcam
		));
		//rgbs.push_back(render_result.Outputs1.RGBMap.cpu());
		//disps.push_back(render_result.Outputs1.DispMap.cpu());
		//normalize depth to[0, 1]
		render_result.Outputs1.DepthMap = (render_result.Outputs1.DepthMap - rparams.Near) / (rparams.Far - rparams.Near);

		//torch::Tensor normals_from_depth = NormalMapFromDepthMap(render_result.Outputs1.DepthMap.detach().cpu());
		//if (!savedir.empty())
		//	cv::imwrite((savedir / ("normals_from_depth_" + std::to_string(disps.size() - 1) + ".png")).string(), TorchTensorToCVMat(normals_from_depth));

		if (!savedir.empty())
		{
			cv::imwrite((savedir / (std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RGBMap.cpu()));
			cv::imwrite((savedir / ("disp_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DispMap));
			cv::imwrite((savedir / ("depth_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DepthMap));
			if (rparams.CalculateNormals)
			{
				cv::imwrite((savedir / ("rendered_norm_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RenderedNormals));
			}
			if (rparams.UsePredNormal)
			{
				cv::imwrite((savedir / ("pred_rendered_norm_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RenderedPredNormals));
			}
			if (rparams.UseLeRF)
			{
				//!!!визуализацию similarity с каким-нибудь словом относительно случайного набора (см статью и код)
				cv::Mat relevancy_img(h, w, CV_8UC1/*CV_32FC1*/);
				#pragma omp parallel for
				for (int i = 0; i < w; i++)
					for (int j = 0; j < h; j++)
					{
						torch::Tensor image_features = render_result.Outputs1.RenderedLangEmbedding.index({j,i}).to(torch::kCPU).unsqueeze(0);
						////Уже нормировано при вычислении пирамиды normalize features???
						//image_features = image_features / image_features.norm(2/*L2*/, -1, true);
				
						////cosine similarity as logits
						//auto logits = /*scale * */torch::mm(image_features, text_features.t());
						//float lv = (logits[0,0].item<float>()/*/scale*/ + 1)/2;
						//test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>(lv/*(lv>0.5?lv:0) */* 255);	//[-1..1] -> [0..255]

						//relevancy
						torch::Tensor rel = Relevancy(image_features, rparams.LerfPositives, rparams.LerfNegatives);
						float lv = rel.index({0,0}).item<float>();
						relevancy_img.at<uchar>(j, i) = cv::saturate_cast<uchar>((lv>0.5?lv:0) * 255);	//[-1..1] -> [0..255]
						//relevancy_img.at<float>(j, i) = /*cv::saturate_cast<uchar>(*/lv/*(lv>0.5?lv:0)*/ /** 255*/;	//[-1..1] -> [0..255]
					}
				
				//cv::normalize(relevancy_img, relevancy_img, 0, 255, cv::NORM_MINMAX);
				//relevancy_img.convertTo(relevancy_img, CV_8UC1);
				cv::imwrite((savedir / ("rendered_lang_embedding" + std::to_string(i) + ".png")).string(), relevancy_img);
			}
		}
	}
	//return std::make_pair(rgbs, disps);
}			//NeRFExecutor :: RenderPath


///
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF> :: Train(const CompactData &data, NeRFExecutorTrainParams &params)
{
	torch::Tensor LerfPositives,
		LerfNegatives;

	Initialize(this->Params, data.BoundingBox);

	if (Params.use_lerf)
	{
		PyramidEmbedderProperties pyramid_embedder_properties; 
		pyramid_embedder_properties.ImgSize = {Params.clip_input_img_size, Params.clip_input_img_size};	//Входной размер изображения сети
		pyramid_embedder_properties.Overlap = 0.5f;										///Доля перекрытия
		///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0 , 1, 2...
		pyramid_embedder_properties.MaxZoomOut = std::min(log2f(data.W/Params.clip_input_img_size), log2f(data.H/Params.clip_input_img_size));
		PyramidEmbedder PyramidClipEmbedder(Clip, ClipProcessor, pyramid_embedder_properties);

		if (!std::filesystem::exists(params.PyramidClipEmbeddingSaveDir/"pyramid_embeddings.pt"))
		{
			std::cout << "calculating pyramid embeddings..." << std::endl;
			///Разбить на патчи с перекрытием  +  парочку масштабов (zoomout) и кэшировать эмбеддинги от них
			PyramidClipEmbedding = PyramidClipEmbedder(data);
			PyramidClipEmbedding.Save(params.PyramidClipEmbeddingSaveDir/"pyramid_embeddings.pt");
		} else {
			std::cout << "loading pyramid embeddings..." << std::endl;
			PyramidClipEmbedding.Load(params.PyramidClipEmbeddingSaveDir/"pyramid_embeddings.pt");
		}

		std::vector<torch::Tensor> canon_texts_tensors;
		for (auto &it : Params.lerf_negatives)
			canon_texts_tensors.push_back(ClipProcessor->EncodeText(it));
		LerfNegatives = Clip->EncodeText(torch::stack(canon_texts_tensors).to(Device)).to(torch::kCPU); ///[3, 768]
		//LerfNegatives = LerfNegatives / LerfNegatives.norm(2/*L2*/, -1, true);
		//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));

		auto input = ClipProcessor->EncodeText(Params.lerf_positives);
		LerfPositives = Clip->EncodeText(input.unsqueeze(0).to(Device)).to(torch::kCPU);		///[1, 768]
		//LerfPositives = LerfPositives / LerfPositives.norm(2/*L2*/, -1, true);
	}

	//if (Params.use_lerf) {
	//	std::cout<<"lang embedding visualization test..."<<std::endl;
	//	//test
	//	PyramidEmbedderProperties pyramid_embedder_properties; 
	//	pyramid_embedder_properties.ImgSize = {Params.clip_input_img_size, Params.clip_input_img_size};	//Входной размер изображения сети
	//	pyramid_embedder_properties.Overlap = 0.5f;										///Доля перекрытия
	//	///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0 , 1, 2...
	//	pyramid_embedder_properties.MaxZoomOut = std::min(log2f(data.W/Params.clip_input_img_size), log2f(data.H/Params.clip_input_img_size));
	//	//auto scale = Clip->GetLogitScale().exp().to(torch::kCPU);

	//	int img_id = 50;

	//	//!!!В этом месте можно распараллелить через выполнение батчами на GPU
	//	//!Объединив несколько cat({image_features1..image_featuresТ})
	//	cv::Mat test_img(data.H, data.W, CV_8UC1);
	//	#pragma omp parallel for
	//	for (int i = 0; i < data.W; i++)
	//		for (int j = 0; j < data.H; j++)
	//		{
	//			torch::Tensor image_features = PyramidClipEmbedding.GetPixelValue(i,j,0.5f,img_id,pyramid_embedder_properties,cv::Size(data.W, data.H)).to(torch::kCPU);
	//			//Уже нормировано при вычислении пирамиды normalize features???
	//			//image_features = image_features / image_features.norm(2/*L2*/, -1, true);
	//			//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
	//			
	//			////cosine similarity as logits
	//			//auto logits = /*scale * */torch::mm(image_features, LerfPositives.t());
	//			//float lv = (logits[0,0].item<float>()/*/scale*/ + 1)/2;
	//			//test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>((lv>0.5?lv:0) * 255);	//[-1..1] -> [0..255]

	//			//relevancy
	//			torch::Tensor rel = Relevancy(image_features, LerfPositives, LerfNegatives);
	//			float lv = rel.index({0,0}).item<float>();
	//			test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>((lv>0.5?lv:0) * 255);	//[-1..1] -> [0..255]
	//		}
	//	cv::imshow("img", TorchTensorToCVMat(data.Imgs[img_id]));
	//	cv::imshow("test_img", test_img);
	//	cv::waitKey(0);
	//}

	////NDC only good for LLFF - style forward facing data
	//if (params.DatasetType != DatasetType::LLFF)
	//{
	//	std::cout << "Not ndc!" << std::endl;
	//	params.Ndc = false;
	//}

	////Если требуется то будем рендерить в тестовых позициях см еще код ниже
	//if (params.RenderTest)
	//	render_poses = np.array(poses[i_test взять из data.SplitsIdx])
	//Create log dir and copy the config file ....

	int global_step = Start;

	//Short circuit if only rendering out from trained model
	if (params.RenderOnly)
	{
		std::cout << "RENDER ONLY" << std::endl;
		torch::NoGradGuard no_grad;

		auto test_save_dir = params.BaseDir / (std::string("renderonly_") + /*params.RenderTest ? "test_" :*/ "path_" + std::to_string(Start));
		if (!std::filesystem::exists(test_save_dir))	std::filesystem::create_directories(test_save_dir);
		std::cout << "test_poses_shape: " << data.RenderPoses.size() << std::endl;

		//test args: perturb = False, raw_noise_std = 0., calculate_normals = false
		RenderParams render_params;
		render_params.NSamples = params.NSamples;
		render_params.NImportance = NImportance;
		render_params.Chunk = params.Chunk;
		render_params.ReturnRaw = params.ReturnRaw;
		render_params.LinDisp = params.LinDisp;
		render_params.Perturb = 0.f;
		//render_params.WhiteBkgr = params.WhiteBkgr;
		render_params.RawNoiseStd = 0.;
		render_params.Ndc = params.Ndc;
		render_params.Near = data.Near;
		render_params.Far = data.Far;
		render_params.UseViewdirs = UseViewDirs;
		render_params.CalculateNormals = false;
		render_params.UsePredNormal = Params.use_pred_normal;
		render_params.ReturnWeights = false;
		render_params.UseLeRF = Params.use_lerf;
		render_params.LangEmbedDim = Params.lang_embed_dim;
		render_params.RenderFactor = params.RenderFactor;
		render_params.LerfPositives = LerfPositives;
		render_params.LerfNegatives = LerfNegatives;
		/*auto [rgbs, disps] = */RenderPath(data.RenderPoses, data.H, data.W, data.Focal, data.K, render_params, 
			{ torch::Tensor(), torch::Tensor() }, /*c2w_staticcam*/torch::Tensor(), /*data.Imgs,*/ test_save_dir
		);

		//cv::VideoWriter video((test_save_dir / "video.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(data.W, data.H), true);
		//for (auto &img : rgbs)
		//	video.write(TorchTensorToCVMat(img));
		//video.release();

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
			auto [rays_o, rays_d] = GetRays(data.H, data.W, data.K, pose.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }));
			rays.emplace_back(torch::cat({ rays_o, rays_d }, 0/*?*/));	//[N, ro + rd, H, W, 3]
		}
		rays_rgb = torch::stack(rays, 0);								//[N, ro + rd, H, W, 3]
		rays_rgb = torch::cat({ rays_rgb, torch::stack(data.Imgs, 0) }, 1);				//[N, ro + rd + rgb, H, W, 3]

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
		torch::Tensor target_lang_embedding;
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
		} else {
			//Random from one image
			int img_i = RandomInt() % (data.SplitsIdx[0] /* + 1 ? */);
			auto target = data.Imgs[img_i].squeeze(0).to(Device);		//1, 800, 800, 4 -> 800, 800, 4
			auto pose = poses.index({ img_i, torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) });
			torch::Tensor coords;
			if (params.NRand != 0)
			{
				///Выбор случайных точек для батча и получение для них параметров луча
				auto [rays_o, rays_d] = GetRays(data.H, data.W, data.K, pose);

				if (i < params.PrecorpIters)
				{
					int dh = int(data.H / 2 * params.PrecorpFrac);
					int dw = int(data.W / 2 * params.PrecorpFrac);

					coords = torch::stack(
						torch::meshgrid({
							torch::linspace(data.H / 2 - dh, data.H / 2 + dh - 1, 2 * dh, torch::TensorOptions().dtype(torch::kLong).device(Device)),
							torch::linspace(data.W / 2 - dw, data.W / 2 + dw - 1, 2 * dw, torch::TensorOptions().dtype(torch::kLong).device(Device))
							}), -1);
					if (i == this->Start + 1)
						std::cout << "[Config] Center cropping of size " << 2 * dh << "x" << 2 * dw << " is enabled until iter " << params.PrecorpIters << std::endl;
				} else {
					coords = torch::stack(torch::meshgrid({ torch::linspace(0, data.H - 1, data.H, torch::TensorOptions().dtype(torch::kLong).device(Device)), torch::linspace(0, data.W - 1, data.W, torch::TensorOptions().dtype(torch::kLong).device(Device)) }), -1);  //(H, W, 2)
				}

				coords = torch::reshape(coords, { -1, 2 });		//(H * W, 2)

				torch::Tensor select_inds = torch::randperm(coords.size(0)/*, torch::kLong*/).slice(0, 0, params.NRand);		// (N_rand,)
				torch::Tensor select_coords = coords.index({ select_inds }).to(Device);		///!!!Вот так можно вытащить данные из по индексам записанным в другой тензор, здесь по нулевому измерению, но построить индекс можно по любому набору измерений
				rays_o = rays_o.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({ torch::indexing::Slice(), 1}) });		// (N_rand, 3)
				rays_d = rays_d.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });		// (N_rand, 3)
				//batch_rays = torch::stack({ rays_o, rays_d }, /*dim=*/0);
				batch_rays.first = rays_o;
				batch_rays.second = rays_d;
				target_s = target.index({ select_coords.index({torch::indexing::Slice(), 0}), select_coords.index({torch::indexing::Slice(), 1}) });  // (N_rand, 3)

				///Вычислим CLIP эмбеддинги в точках изображения которые попали в батч
				if (Params.use_lerf)
				{
					PyramidEmbedderProperties pyramid_embedder_properties; 
					pyramid_embedder_properties.ImgSize = {Params.clip_input_img_size, Params.clip_input_img_size};	//Входной размер изображения сети
					pyramid_embedder_properties.Overlap = 0.5f;										///Доля перекрытия
					///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0, 1, 2...
					pyramid_embedder_properties.MaxZoomOut = std::min(log2f(data.W/Params.clip_input_img_size), log2f(data.H/Params.clip_input_img_size));
					auto select_coords_cpu = select_coords.to(torch::kCPU).to(torch::kFloat);		//!!!.item<long>()почему то не находит поэтому преобразуем во float
					target_lang_embedding = torch::ones({select_coords_cpu.size(0)/*NRand*/, Params.lang_embed_dim}, torch::kFloat32);
					#pragma omp parallel for
					for (int idx = 0; idx < select_coords_cpu.size(0)/*NRand*/; idx++)
					{
						target_lang_embedding.index_put_({idx/*, torch::indexing::Slice()*/}, PyramidClipEmbedding.GetPixelValue(
							select_coords_cpu.index({ idx, 0 }).item<float>(),
							select_coords_cpu.index({ idx, 1 }).item<float>(),
							0.5f,		///!!!ПРИВЯЗАТЬСЯ К МАСШТАБУ для этого перенести в RunNetwork по аналогии с calculated_normals			//Get zoom_out_idx from scale //-1, 0, 1, 2 ... <- 1/2, 1, 2, 4 ...
							img_i,
							pyramid_embedder_properties,
							cv::Size(data.W, data.H)
						));
					}
				}		//if use_lerf
			}		//if (params.NRand != 0)
		}		// else (!params.NoBatching)

		//Core optimization loop
		Optimizer->zero_grad();
		
		RenderParams render_params;
		render_params.NSamples = params.NSamples;
		render_params.NImportance = NImportance;
		render_params.Chunk = params.Chunk;
		render_params.ReturnRaw = params.ReturnRaw;
		render_params.LinDisp = params.LinDisp;
		render_params.Perturb = 0.f;
		//render_params.WhiteBkgr = params.WhiteBkgr;
		render_params.RawNoiseStd = 0.f;
		render_params.Ndc = params.Ndc;
		render_params.Near = data.Near;
		render_params.Far = data.Far;
		render_params.UseViewdirs = UseViewDirs;
		render_params.CalculateNormals = Params.calculate_normals || Params.use_pred_normal;
		render_params.UsePredNormal = Params.use_pred_normal;
		render_params.ReturnWeights = true;
		render_params.UseLeRF = Params.use_lerf;
		render_params.LangEmbedDim = Params.lang_embed_dim;
		render_params.RenderFactor = 0;
		render_params.LerfPositives = LerfPositives;
		render_params.LerfNegatives = LerfNegatives;

		RenderResult rgb_disp_acc_extras = std::move(Render(data.H, data.W, data.K,
			Model, ExecutorEmbedder, ExecutorEmbeddirs, ModelFine,
			render_params, batch_rays, torch::Tensor(), torch::Tensor()				/*либо rays либо pose c2w*/
		));

		auto mse_loss = torch::mse_loss(rgb_disp_acc_extras.Outputs1.RGBMap, target_s.detach());
		auto img_loss = torch::nn::functional::huber_loss(
					rgb_disp_acc_extras.Outputs1.RGBMap,
					target_s.detach()
			);
		if (params.ReturnRaw)
			auto trans = rgb_disp_acc_extras.Raw.index({ "...", -1 });		//последний элемент последнего измерения тензора
		auto loss = img_loss;
		auto psnr = -10. * torch::log(mse_loss) / torch::log(torch::full({ 1/*img_loss.sizes()*/ }, /*value=*/10.f)).to(Device);

		torch::Tensor img_loss0,
			psnr0;
		if (rgb_disp_acc_extras.Outputs0.RGBMap.defined() && (rgb_disp_acc_extras.Outputs0.RGBMap.numel() != 0))
		{
			auto mse_loss0 = torch::mse_loss(rgb_disp_acc_extras.Outputs0.RGBMap, target_s.detach());
			img_loss0 = torch::nn::functional::huber_loss(
					rgb_disp_acc_extras.Outputs0.RGBMap,
					target_s.detach()
				);
			loss = loss + img_loss0;
			psnr0 = -10. * torch::log(mse_loss0) / torch::log(torch::full({ 1/*img_loss.sizes()*/ }, /*value=*/10.f)).to(Device);
		}

		//add Total Variation loss
		if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
			if (i < n_iters/2)
			{
				const float tv_loss_weight = 1e-6;
				for (int level = 0; level < ExecutorEmbedder->GetNLevels(); level++)
				{
					loss = loss + tv_loss_weight * TotalVariationLoss(
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

		////add Total Variation loss
		//if constexpr (std::is_same_v<TEmbedder, CuHashEmbedder>)
		//	if (i < n_iters/2)
		//	{
		//		const float tv_loss_weight = 1e-6;
		//		loss += tv_loss_weight * TotalVariationLoss(ExecutorEmbedder).to(loss.device());
		//	}

		if (rgb_disp_acc_extras.Outputs1.Normals.defined() && rgb_disp_acc_extras.Outputs1.Normals.numel() != 0)
		{
			const float orientation_loss_weight = 1.f;//1e-4;
			//Loss that encourages that all visible normals are facing towards the camera.
			auto orientation_loss = OrientationLoss(rgb_disp_acc_extras.Outputs1.Weights.unsqueeze(-1).detach(), rgb_disp_acc_extras.Outputs1.Normals, batch_rays.second/*directions*/);
			loss = loss + torch::mean(orientation_loss) * orientation_loss_weight;
		}

		//
		if (Params.use_pred_normal)
		{
			const float pred_normal_loss_weight = 1.f;//1e-3;
			//Loss between normals calculated from density and normals from prediction network.
			auto pred_normal_loss = PredNormalLoss(
				rgb_disp_acc_extras.Outputs1.Weights.unsqueeze(-1).detach(),
				rgb_disp_acc_extras.Outputs1.Normals.detach(),	//Градиент не течет сюда потому что хотим оптимизировать предсказанные нормали за счет вычисленных
				rgb_disp_acc_extras.Outputs1.PredNormals
			);
			loss = loss + torch::mean(pred_normal_loss) * pred_normal_loss_weight;
		}

		torch::Tensor lang_loss;
		const float lang_loss_weight = 1e-1f;//1e-2f;
		if (Params.use_lerf)
		{
			lang_loss = torch::nn::functional::huber_loss(
					rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()),
					target_lang_embedding.detach().to(loss.device()),	//target clip embeddings provided by PyramidEmbedder for a given set of pixels
					torch::nn::functional::HuberLossFuncOptions().reduction(torch::kNone).delta(1.25)
				).sum(-1).nanmean();
			//auto lang_loss = torch::mse_loss(rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()), target_lang_embedding.to(loss.device()));
				loss = loss + lang_loss * lang_loss_weight;		
		}

		try {
			loss.backward();
			Optimizer->step();
		} catch (c10::Error &error){
			std::cout<<error.msg()<<" "<<error.what()<<std::endl;
		}

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
			torch::save(ExecutorEmbedder, (path / "embedder_checkpoint.pt").string());
			torch::save(torch::full({ 1 }, /*value=*/global_step), (path / "start_checkpoint.pt").string());
			torch::save(*Optimizer.get(), (path / "optimizer_checkpoint.pt").string());
			torch::save(Model, (path / "model_checkpoint.pt").string());
			if (!ModelFine.is_empty())
				torch::save(ModelFine, (path / "model_fine_checkpoint.pt").string());
			std::cout << "Saved checkpoints at " << path.string() << std::endl;
		}

		if ((i % params.ITestset == 0) && (i > 0) && !params.TestSkip)		//если TestSkip то просто не подгружены тестовые данные 
		{
			auto test_save_dir = params.BaseDir; /* / ("testset_" + std::to_string(i)) */;
			if (!std::filesystem::exists(test_save_dir))	std::filesystem::create_directories(test_save_dir);
			//Есть torch::Tensor poses и есть std::vector<torch::Tensor> data.Poses
			std::cout << "poses shape " << poses.sizes() << "; test poses count " << data.SplitsIdx[2] << std::endl;   //	std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			torch::NoGradGuard no_grad;		///Закоментировал для вычисления нормалей

			std::vector<torch::Tensor> test_poses,
				test_imgs;
			//std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			if (data.SplitsIdx[2] == data.SplitsIdx[1] || data.SplitsIdx[2] == 0)
			{
				for (int k = 0; k < data.SplitsIdx[0]; k++)
				{
					test_poses.push_back(data.Poses[k].to(Device));
				}
			} else {
				for (int k = data.SplitsIdx[0] + data.SplitsIdx[1]; k < data.SplitsIdx[0] + data.SplitsIdx[1] + data.SplitsIdx[2]; k++)
				{
					test_poses.push_back(data.Poses[k].to(Device));
				}
			}
			RenderParams render_params;
			render_params.NSamples = params.NSamples;
			render_params.NImportance = NImportance;
			render_params.Chunk = params.Chunk;
			render_params.ReturnRaw = params.ReturnRaw;
			render_params.LinDisp = params.LinDisp;
			render_params.Perturb = 0.f;
			//render_params.WhiteBkgr = params.WhiteBkgr;
			render_params.RawNoiseStd = 0.f;
			render_params.Ndc = params.Ndc;
			render_params.Near = data.Near;
			render_params.Far = data.Far;
			render_params.UseViewdirs = UseViewDirs;
			render_params.CalculateNormals = false;
			render_params.UsePredNormal = Params.use_pred_normal;
			render_params.ReturnWeights = false;
			render_params.UseLeRF = Params.use_lerf;
			render_params.LangEmbedDim = Params.lang_embed_dim;
			render_params.RenderFactor = 0;
			render_params.LerfPositives = LerfPositives;
			render_params.LerfNegatives = LerfNegatives;
			RenderPath(test_poses/*poses[i_test]).to(device)*/, data.H, data.W, data.Focal, data.K, render_params,
				{ torch::Tensor(), torch::Tensor() }, /*c2w_staticcam*/torch::Tensor(), /*test_imgs,*/ test_save_dir
			);
			std::cout << "Saved test set" << std::endl;
		}

		if ((i % params.IVideo == 0) && i > 0)
		{
			torch::NoGradGuard no_grad;
			auto dir = params.BaseDir / "render";
			if (!std::filesystem::exists(dir))	std::filesystem::create_directories(dir);
			//test args: perturb = False, raw_noise_std = 0.
			RenderParams render_params;
			render_params.NSamples = params.NSamples;
			render_params.NImportance = NImportance;
			render_params.Chunk = params.Chunk;
			render_params.ReturnRaw = params.ReturnRaw;
			render_params.LinDisp = params.LinDisp;
			render_params.Perturb = 0.f;
			//render_params.WhiteBkgr = params.WhiteBkgr;
			render_params.RawNoiseStd = 0.f;
			render_params.Ndc = params.Ndc;
			render_params.Near = data.Near;
			render_params.Far = data.Far;
			render_params.UseViewdirs = UseViewDirs;
			render_params.CalculateNormals = false;
			render_params.UsePredNormal = Params.use_pred_normal;
			render_params.ReturnWeights = false;
			render_params.UseLeRF = Params.use_lerf;
			render_params.LangEmbedDim = Params.lang_embed_dim;
			render_params.RenderFactor = 0;
			render_params.LerfPositives = LerfPositives;
			render_params.LerfNegatives = LerfNegatives;
			RenderPath(data.RenderPoses, data.H, data.W, data.Focal, data.K, render_params, { torch::Tensor(), torch::Tensor() },
				/*c2w_staticcam*/torch::Tensor(),/*, data.Imgs,*/  dir/*, params.RenderFactor*/
			);
			std::cout << "Done, saving " << std::endl;
			auto path = params.BaseDir; /* / ("spiral_" + std::to_string(i)) */;
			if (!std::filesystem::exists(path))	std::filesystem::create_directories(path);

			//cv::VideoWriter video((params.BaseDir / "rgb.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(data.W, data.H), true);
			//for (auto &img : rgbs)
			//	video.write(TorchTensorToCVMat(img));
			//video.release();

			//cv::VideoWriter video2((params.BaseDir / "disp.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(data.W, data.H), true);
			//for (auto &img : disps)
			//{
			//	float min_val = img.min().item<float>();
			//	float max_val = img.max().item<float>();
			//	// Нормализуем тензор
			//	img = (img - min_val) / (max_val - min_val);
			//	video2.write(TorchTensorToCVMat(img));
			//}
			//video2.release();
		}

		if (i % params.IPrint == 0)
			std::cout << "[TRAIN] Iter: " << i << " of " << n_iters << " lr = " << new_lrate << " Loss: " << loss.item() << " PSNR: " << psnr.item() << std::endl;

		global_step++;
	}	//for (int i = this->Start + 1; i < n_iters; i++)
}			//NeRFExecutor :: Train