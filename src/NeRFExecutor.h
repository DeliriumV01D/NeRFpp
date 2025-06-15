#pragma once

#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "CuSHEncoder.h"
#include "CuHashEmbedder.h"
#include "PyramidEmbedder.h"
#include "NeRF.h"
#include "LeRF.h"
#include "LeRFRenderer.h"
#include "NeRFactor.h"
#include "NeRFactorRenderer.h"
#include "Sampler.h"
#include "NeRFRenderer.h"
#include "TRandomInt.h"

#include "ColmapReconstruction.h"

#include <set>
#include <filesystem>
#include <chrono>
#include <memory>

//#include <torch/script.h>

#include "json.hpp"

enum class DatasetType { LLFF, BLENDER, LINEMOD, DEEP_VOXELS, COLMAP };

struct NeRFExecutorParams 
{
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
	bool use_nerf{ true },
		use_viewdirs{ true },	//use full 5D input instead of 3D. Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		calculate_normals{ false },
		use_pred_normal{ false },	//whether to use predicted normals
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
		lang_embed_dim{ 768 },			//Language embedder head params
		geo_feat_dim_le{ 32 };			//Language embedder head params
	torch::Device device{ torch::kCUDA };
	float learning_rate{ 5e-4 },
		pyr_embedder_overlap{0.75};
	std::filesystem::path ft_path,
		path_to_clip,			//Path to RuClip model
		path_to_bpe;			//Path to tokenizer
	std::string lerf_positives;
	std::vector<std::string> lerf_negatives;

	nlohmann::json ToJson() const
	{
		nlohmann::json j;
		j["net_depth"] = net_depth;
		j["net_width"] = net_width;
		j["multires"] = multires;
		j["multires_views"] = multires_views;
		j["n_importance"] = n_importance;
		j["net_depth_fine"] = net_depth_fine;
		j["net_width_fine"] = net_width_fine;
		j["num_layers_color"] = num_layers_color;
		j["hidden_dim_color"] = hidden_dim_color;
		j["num_layers_color_fine"] = num_layers_color_fine;
		j["hidden_dim_color_fine"] = hidden_dim_color_fine;
		j["num_layers_normals"] = num_layers_normals;
		j["hidden_dim_normals"] = hidden_dim_normals;
		j["geo_feat_dim"] = geo_feat_dim;
		j["use_nerf"] = use_nerf;
		j["use_viewdirs"] = use_viewdirs;
		j["calculate_normals"] = calculate_normals;
		j["use_pred_normal"] = use_pred_normal;
		j["use_lerf"] = use_lerf;
		j["n_levels"] = n_levels;
		j["n_features_per_level"] = n_features_per_level;
		j["log2_hashmap_size"] = log2_hashmap_size;
		j["base_resolution"] = base_resolution;
		j["finest_resolution"] = finest_resolution;
		j["n_levels_le"] = n_levels_le;
		j["n_features_per_level_le"] = n_features_per_level_le;
		j["log2_hashmap_size_le"] = log2_hashmap_size_le;
		j["base_resolution_le"] = base_resolution_le;
		j["finest_resolution_le"] = finest_resolution_le;
		j["clip_input_img_size"] = clip_input_img_size;
		j["num_layers_le"] = num_layers_le;
		j["hidden_dim_le"] = hidden_dim_le;
		j["lang_embed_dim"] = lang_embed_dim;
		j["geo_feat_dim_le"] = geo_feat_dim_le;
		j["device"] = device.str();			//
		j["learning_rate"] = learning_rate;
		j["pyr_embedder_overlap"] = pyr_embedder_overlap;
		j["ft_path"] = ft_path.string();
		j["path_to_clip"] = path_to_clip.string();
		j["path_to_bpe"] = path_to_bpe.string();
		j["lerf_positives"] = lerf_positives;
		j["lerf_negatives"] = lerf_negatives;
		return j;
	}		//NeRFExecutorParams::ToJson

	void FromJson(const nlohmann::json &j) 
	{
		j.at("net_depth").get_to(net_depth);
		j.at("net_width").get_to(net_width);
		j.at("multires").get_to(multires);
		j.at("multires_views").get_to(multires_views);
		j.at("n_importance").get_to(n_importance);
		j.at("net_depth_fine").get_to(net_depth_fine);
		j.at("net_width_fine").get_to(net_width_fine);
		j.at("num_layers_color").get_to(num_layers_color);
		j.at("hidden_dim_color").get_to(hidden_dim_color);
		j.at("num_layers_color_fine").get_to(num_layers_color_fine);
		j.at("hidden_dim_color_fine").get_to(hidden_dim_color_fine);
		j.at("num_layers_normals").get_to(num_layers_normals);
		j.at("hidden_dim_normals").get_to(hidden_dim_normals);
		j.at("geo_feat_dim").get_to(geo_feat_dim);
		j.at("use_nerf").get_to(use_nerf);
		j.at("use_viewdirs").get_to(use_viewdirs);
		j.at("calculate_normals").get_to(calculate_normals);
		j.at("use_pred_normal").get_to(use_pred_normal);
		j.at("use_lerf").get_to(use_lerf);
		j.at("n_levels").get_to(n_levels);
		j.at("n_features_per_level").get_to(n_features_per_level);
		j.at("log2_hashmap_size").get_to(log2_hashmap_size);
		j.at("base_resolution").get_to(base_resolution);
		j.at("finest_resolution").get_to(finest_resolution);
		j.at("n_levels_le").get_to(n_levels_le);
		j.at("n_features_per_level_le").get_to(n_features_per_level_le);
		j.at("log2_hashmap_size_le").get_to(log2_hashmap_size_le);
		j.at("base_resolution_le").get_to(base_resolution_le);
		j.at("finest_resolution_le").get_to(finest_resolution_le);
		j.at("clip_input_img_size").get_to(clip_input_img_size);
		j.at("num_layers_le").get_to(num_layers_le);
		j.at("hidden_dim_le").get_to(hidden_dim_le);
		j.at("lang_embed_dim").get_to(lang_embed_dim);
		j.at("geo_feat_dim_le").get_to(geo_feat_dim_le);
		device = torch::Device(j.at("device").get<std::string>());					// Приводим device к правильному типу
		learning_rate = j.at("learning_rate").get<float>();
		pyr_embedder_overlap = j.at("pyr_embedder_overlap").get<float>();
		ft_path = j.at("ft_path").get<std::string>();
		path_to_clip = j.at("path_to_clip").get<std::string>();
		path_to_bpe = j.at("path_to_bpe").get<std::string>();
		j.at("lerf_positives").get_to(lerf_positives);
		j.at("lerf_negatives").get_to(lerf_negatives);
	}		//NeRFExecutorParams::FromJson

	void LoadFromFile(const std::filesystem::path &file_path)
	{
		std::ifstream fs(file_path);
		nlohmann::json j;
		fs >> j;
		FromJson(j);
	}

	void SaveToFile(const std::filesystem::path &file_path)
	{
		std::ofstream fs(file_path);
		fs << ToJson() << std::endl;
	}
};		//NeRFExecutorParams

struct NeRFExecutorTrainParams {
	std::filesystem::path //DataDir,	///input data directory
		PyramidClipEmbeddingSaveDir,
		BaseDir;											///where to store ckpts and logs
	bool TestSkip{ false },						///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
		RenderOnly{ false },					///do not optimize, reload weights and render out render_poses path
		Ndc{ false },									///use normalized device coordinates (set for non-forward facing scenes)
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

	nlohmann::json ToJson() const
	{
		nlohmann::json j;
		j["PyramidClipEmbeddingSaveDir"] = PyramidClipEmbeddingSaveDir.string();
		j["BaseDir"] = BaseDir.string();
		j["TestSkip"] = TestSkip;
		j["RenderOnly"] = RenderOnly;
		j["Ndc"] = Ndc;
		j["LinDisp"] = LinDisp;
		j["NoBatching"] = NoBatching;
		j["Chunk"] = Chunk;
		j["NSamples"] = NSamples;
		j["NRand"] = NRand;
		j["PrecorpIters"] = PrecorpIters;
		j["NIters"] = NIters;
		j["LRateDecay"] = LRateDecay;
		j["IPrint"] = IPrint;
		j["IImg"] = IImg;
		j["IWeights"] = IWeights;
		j["ITestset"] = ITestset;
		j["IVideo"] = IVideo;
		j["ReturnRaw"] = ReturnRaw;
		j["RenderFactor"] = RenderFactor;
		j["PrecorpFrac"] = PrecorpFrac;
		return j;
	}

	void FromJson(const nlohmann::json &j)
	{
		j.at("PyramidClipEmbeddingSaveDir").get_to(PyramidClipEmbeddingSaveDir);
		j.at("BaseDir").get_to(BaseDir);
		j.at("TestSkip").get_to(TestSkip);
		j.at("RenderOnly").get_to(RenderOnly);
		j.at("Ndc").get_to(Ndc);
		j.at("LinDisp").get_to(LinDisp);
		j.at("NoBatching").get_to(NoBatching);
		j.at("Chunk").get_to(Chunk);
		j.at("NSamples").get_to(NSamples);
		j.at("NRand").get_to(NRand);
		j.at("PrecorpIters").get_to(PrecorpIters);
		j.at("NIters").get_to(NIters);
		j.at("LRateDecay").get_to(LRateDecay);
		j.at("IPrint").get_to(IPrint);
		j.at("IImg").get_to(IImg);
		j.at("IWeights").get_to(IWeights);
		j.at("ITestset").get_to(ITestset);
		j.at("IVideo").get_to(IVideo);
		j.at("ReturnRaw").get_to(ReturnRaw);
		j.at("RenderFactor").get_to(RenderFactor);
		j.at("PrecorpFrac").get_to(PrecorpFrac);
	}

	void LoadFromFile(const std::filesystem::path &file_path)
	{
		std::ifstream fs(file_path);
		nlohmann::json j;
		fs >> j;
		FromJson(j);
	}

	void SaveToFile(const std::filesystem::path &file_path)
	{
		std::ofstream fs(file_path);
		fs << ToJson() << std::endl;
	}
};		//NeRFExecutorTrainParams


inline CompactData LoadData (
	const std::filesystem::path &basedir,
	const torch::Device &device,
	DatasetType dataset_type = DatasetType::BLENDER,
	bool half_res = false,				///load blender synthetic data at 400x400 instead of 800x800
	bool test_skip = false,				///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
	bool white_bkgr = false				///set to render synthetic data on a white bkgd (always use for dvoxels)
)	{
	std::string dataset_type_str;
	CompactData data;
	//Загрузить данные
	if (dataset_type == DatasetType::BLENDER)
	{
		dataset_type_str = "BLENDER";
		data = load_blender_data(basedir, half_res, test_skip);
		//!!!Тут K and BoundingBox поломано, поэтому ниже все повторно заполняется
	}

	if (dataset_type == DatasetType::COLMAP)
	{
		dataset_type_str = "COLMAP";
		data = LoadFromColmapReconstruction(basedir.string());
	}

	std::cout << "Loaded " << dataset_type_str << data.Imgs.size() << " " << data.RenderPoses.size() << " " << data.H << " " << data.W << " " << data.Focal << " " << basedir << std::endl;
	if (!data.Imgs.empty())
		std::cout << "data.Imgs[0]: " << data.Imgs[0].sizes() << " " << data.Imgs[0].device() << " " << data.Imgs[0].type() << std::endl;
	//i_train, i_val, i_test = i_split;

	for (auto &img : data.Imgs)
	{
		if (white_bkgr)
			img = img.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) }) * img.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }) + (1. - img.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }));
		else
			img = img.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) });
	}

	if (dataset_type == DatasetType::BLENDER)
	{
		///!!!Сделать нормальное копирование, так как эти тензоры заполняются внутри
		float kdata[] = { data.Focal, 0, 0.5f * data.W,
			0, data.Focal, 0.5f * data.H,
			0, 0, 1 };
		data.K = torch::from_blob(kdata, { 3, 3 }, torch::kFloat32);
		//data.K = GetCalibrationMatrix(data.Focal, data.W, data.H).clone().detach();
		data.BoundingBox = GetBbox3dForObj(data).clone().detach();
		
		//Move testing data to GPU
		for (auto &it : data.RenderPoses)
			it = it.to(device);
	}

	return data;
}


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
class NeRFExecutor {
protected:
	NeRFExecutorParams Params;
	TEmbedder ExecutorEmbedder = nullptr;
	TEmbedDirs ExecutorEmbeddirs = nullptr;
	TLeRFEmbedder LangEmbedder = nullptr;
	TNeRF Model = nullptr,
		ModelFine = nullptr;
	TLeRF LangModel = nullptr,
		LangModelFine = nullptr;
	std::unique_ptr<TNeRFRenderer> NeRFRenderer = nullptr;
	std::unique_ptr<TLeRFRenderer> LeRFRenderer = nullptr;
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
	//torch::Tensor LerfPositives,
	//	LerfNegatives;
public:
	NeRFExecutor(const NeRFExecutorParams &params) : Params(params), Device(params.device), NImportance(params.n_importance), UseViewDirs(params.use_viewdirs), LearningRate(params.learning_rate){};
	void Initialize(const NeRFExecutorParams &params, torch::Tensor bounding_box);
	
	///
	std::tuple <NeRFRenderResult, LeRFRenderResult> RenderView(
		const torch::Tensor render_pose,	//rays?  	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() };			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		int w,
		int h,
		const torch::Tensor k,
		const NeRFRenderParams &rparams
	);

	///
	void RenderPath(
		const std::vector <torch::Tensor> &render_poses,
		int h,
		int w,
		float focal,
		torch::Tensor k,
		const NeRFRenderParams &rparams,
		std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		torch::Tensor c2w_staticcam = torch::Tensor(),			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
		//torch::Tensor gt_imgs = torch::Tensor(),
		const std::filesystem::path savedir = ""
	);

	///
	NeRFRenderParams * FillRenderParams(
		NeRFExecutorParams executor_params,
		NeRFExecutorTrainParams executor_train_params,
		const float Near,
		const float Far,
		const bool returm_weights = false,
		const bool calculate_normals = false
	);

	///
	void SetLeRFPrompts(const std::string &lerf_positives, const std::vector<std::string> &lerf_negatives);
	void SetLeRFPrompts(torch::Tensor lerf_positives, torch::Tensor lerf_negatives);
	std::tuple<torch::Tensor, torch::Tensor> GetLeRFPrompts();

	void InitializePyramidClipEmbedding(const NeRFExecutorTrainParams &params, const CompactData &data, bool test = false);

	///
	void Train(const CompactData &data, NeRFExecutorTrainParams &params);
	NeRFExecutorParams GetParams(){return Params;};
	torch::Tensor GetEmbedderBoundingBox(){return ExecutorEmbedder->GetBoundingBox();};
	torch::Tensor GetLangEmbedderBoundingBox(){return LangEmbedder->GetBoundingBox();};

};				//NeRFExecutor


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
NeRFRenderParams * NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: FillRenderParams(
	NeRFExecutorParams executor_params,
	NeRFExecutorTrainParams executor_train_params,
	const float Near,
	const float Far,
	const bool returm_weights /*= false*/,
	const bool calculate_normals /*= false*/
){
	NeRFRenderParams * render_params;

	//if constexpr (std::is_same_v<TNeRF, NeRFactor>)
	//{
	// 	render_params.CalculateNormals = false;
	//	render_params.UsePredNormal = executor_params.use_pred_normal;
	//}
	render_params = new NeRFRenderParams();
	
	render_params->NSamples = executor_train_params.NSamples;
	render_params->NImportance = NImportance;
	render_params->Chunk = executor_train_params.Chunk;
	render_params->ReturnRaw = executor_train_params.ReturnRaw;
	render_params->LinDisp = executor_train_params.LinDisp;
	render_params->Perturb = 0.f;
	//render_params.WhiteBkgr = params.WhiteBkgr;
	render_params->RawNoiseStd = 0.;
	render_params->Ndc = executor_train_params.Ndc;
	render_params->Near = Near;
	render_params->Far = Far;
	render_params->UseViewdirs = UseViewDirs;
	render_params->ReturnWeights = returm_weights;
	render_params->RenderFactor = executor_train_params.RenderFactor;

	return render_params;
}


///if using hashed for xyz, use SH for views
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: Initialize(const NeRFExecutorParams &params, torch::Tensor bounding_box)
{
	int input_ch,
		input_ch_views = 0;
	if (params.use_nerf)
	{
		if constexpr (std::is_same_v<TEmbedder, Embedder>)
			ExecutorEmbedder = Embedder("embedder", params.multires);
		if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
			ExecutorEmbedder = HashEmbedder("embedder", bounding_box.to(Device), params.n_levels, params.n_features_per_level, params.log2_hashmap_size, params.base_resolution, params.finest_resolution);
		if constexpr (std::is_same_v<TEmbedder, CuHashEmbedder>)
			ExecutorEmbedder = CuHashEmbedder("embedder", bounding_box.to(Device), params.n_levels, params.n_features_per_level, params.log2_hashmap_size, params.base_resolution, params.finest_resolution);

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
	}	//if use_nerf

	if (params.use_lerf)
	{
		if constexpr (std::is_same_v<TLeRFEmbedder, CuHashEmbedder>)
			LangEmbedder = CuHashEmbedder("lang_embedder", bounding_box.to(Device), params.n_levels_le, params.n_features_per_level_le, params.log2_hashmap_size_le, params.base_resolution_le, params.finest_resolution_le);

		LangEmbedder->to(params.device);
		auto embp = LangEmbedder->parameters();
		GradVars.insert(GradVars.end(), std::make_move_iterator(embp.begin()), std::make_move_iterator(embp.end()));		//!!!
	
		for (auto &k : LangEmbedder->named_parameters())
			std::cout << k.key() << std::endl;
		std::cout << "LangEmbedder params count: " << Trainable::ParamsCount(LangEmbedder) << std::endl;
	}	//if use_lerf

	if (params.use_nerf)
	{
		int output_ch = (params.n_importance > 0) ? 5 : 4;
		const std::set<int> skips = std::set<int>{ 4 };

		if constexpr (std::is_same_v<TNeRF, NeRF>)
			Model = NeRF(params.net_depth, params.net_width, input_ch, input_ch_views, output_ch, skips, params.use_viewdirs, "model");

		if constexpr (std::is_same_v<TNeRF, NeRFSmall>)
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

			ModelFine->to(params.device);
			auto temp = ModelFine->parameters();
			GradVars.insert(GradVars.end(), std::make_move_iterator(temp.begin()), std::make_move_iterator(temp.end()));

			for (auto& k : ModelFine->named_parameters())
				std::cout << k.key() << std::endl;
			std::cout << "ModelFine params count: " << Trainable::ParamsCount(ModelFine) << std::endl;
		}		//if n_importance > 0
	}	//if use_nerf

	if (params.use_lerf)
	{
		//if constexpr (std::is_same_v<TLeRF, LeRF>)
			LangModel = TLeRF(		
				params.geo_feat_dim_le,
				params.num_layers_le,
				params.hidden_dim_le,
				params.lang_embed_dim,
				LangEmbedder->GetOutputDims(),/*input_ch_le,*/
				"lang_model"
			);

		LangModel->to(params.device);
		auto mp = LangModel->parameters();
		GradVars.insert(GradVars.end(), std::make_move_iterator(mp.begin()), std::make_move_iterator(mp.end()));		//!!!

		for (auto &k : LangModel->named_parameters())
			std::cout << k.key() << std::endl;
		std::cout << "LangModel params count: " << Trainable::ParamsCount(LangModel) << std::endl;
	}	//if use_lerf

	if (params.use_nerf)
	{
		//if constexpr (std::is_same_v<TNeRFRenderer, NeRFRenderer<TEmbedder, TEmbedDirs, TNeRF>>)
			NeRFRenderer = std::make_unique<TNeRFRenderer> (ExecutorEmbedder, ExecutorEmbeddirs, Model, ModelFine);
	}

	if (params.use_lerf)
	{
		//if constexpr (std::is_same_v<TLeRFRenderer, LeRFRenderer>)
			LeRFRenderer = std::make_unique<TLeRFRenderer> (LangEmbedder, LangModel, LangModelFine);
	}

	//if constexpr (std::is_same_v<TNeRF, NeRF>)
	//	Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-7)/*.weight_decay(0.001)*/.betas(std::make_tuple(0.9, 0.999)));
	Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-15).betas(std::make_tuple(0.9, 0.99)));
	
	if (/*Проверить наличие файлов*/
		std::filesystem::exists(params.ft_path / "start_checkpoint.pt") &&
		std::filesystem::exists(params.ft_path / "optimizer_checkpoint.pt") &&
		((params.use_nerf && std::filesystem::exists(params.ft_path / "model_checkpoint.pt")) || !params.use_nerf) &&
		((params.use_nerf && params.n_importance > 0 && std::filesystem::exists(params.ft_path / "model_fine_checkpoint.pt")) || !(params.use_nerf && params.n_importance > 0)) &&
		((params.use_lerf && std::filesystem::exists(params.ft_path / "lang_embedder_checkpoint.pt")) || !params.use_lerf)
	) {
		std::cout << "restoring parameters from checkpoint..." << std::endl;

		if (params.use_nerf)
		{
			if (std::filesystem::exists(params.ft_path / "embedder_checkpoint.pt"))
				torch::load(ExecutorEmbedder, (params.ft_path / "embedder_checkpoint.pt").string());
			torch::load(Model, (params.ft_path / "model_checkpoint.pt").string());
			if (!ModelFine.is_empty())
				torch::load(ModelFine, (params.ft_path / "model_fine_checkpoint.pt").string());
		}

		if (params.use_lerf)
		{
			if (std::filesystem::exists(params.ft_path / "lang_embedder_checkpoint.pt"))
				torch::load(LangEmbedder, (params.ft_path / "lang_embedder_checkpoint.pt").string());
			torch::load(LangModel, (params.ft_path / "lang_model_checkpoint.pt").string());
			if (!LangModelFine.is_empty())
				torch::load(LangModelFine, (params.ft_path / "lang_model_fine_checkpoint.pt").string());
		}

		torch::Tensor temp;
		torch::load(temp, (params.ft_path / "start_checkpoint.pt").string());
		Start = temp.item<float>();
		torch::load(*Optimizer.get(), (params.ft_path / "optimizer_checkpoint.pt").string());
	} else {
		if (params.use_nerf)
		{
			ExecutorEmbedder->Initialize();//Trainable::Initialize(ExecutorEmbedder);//ExecutorEmbedder->Initialize();	
			Trainable::Initialize(Model);
			if (!ModelFine.is_empty())
				Trainable::Initialize(ModelFine);
		}
		
		if (params.use_lerf)
		{
			LangEmbedder->Initialize();
			Trainable::Initialize(LangModel);
			if (!LangModelFine.is_empty())
				Trainable::Initialize(LangModelFine);
		}
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
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
std::tuple <NeRFRenderResult, LeRFRenderResult> NeRFExecutor<TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: RenderView(
	const torch::Tensor render_pose,	//rays?  	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() };			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	int w,
	int h,
	const torch::Tensor k,
	const NeRFRenderParams &rparams
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

	NeRFRenderResult nerf_render_result;
	LeRFRenderResult lerf_render_result;

	if (Params.use_nerf)
	{
		nerf_render_result = NeRFRenderer->Render(h, w, k1,
			rparams,
			/*Rays*/ {torch::Tensor(), torch::Tensor()},
			render_pose.to(Device).index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
			/*rparams.C2wStaticCam*/ torch::Tensor()
		);
	}
	if (Params.use_lerf)
	{
		lerf_render_result = LeRFRenderer->Render(h, w, k1,
			rparams,
			/*Rays*/ {torch::Tensor(), torch::Tensor()},
			render_pose.to(Device).index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
			/*rparams.C2wStaticCam*/ torch::Tensor()
		);
	}

	return std::make_tuple(nerf_render_result, lerf_render_result);
}		//NeRFExecutor :: RenderView


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: RenderPath(
	const std::vector <torch::Tensor> &render_poses,
	int h,
	int w,
	float focal,
	torch::Tensor k,
	const NeRFRenderParams &rparams,
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
		if (Params.use_nerf)
		{
			auto render_result = std::move(NeRFRenderer->Render(h, w, k,
				rparams,
				rays,
				render_poses[i].index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
				c2w_staticcam
			));		
			render_result.Outputs1.DepthMap = (render_result.Outputs1.DepthMap - rparams.Near) / (rparams.Far - rparams.Near);

			//torch::Tensor normals_from_depth = NormalMapFromDepthMap(render_result.Outputs1.DepthMap.detach().cpu());
			//if (!savedir.empty())
			//	cv::imwrite((savedir / ("normals_from_depth_" + std::to_string(disps.size() - 1) + ".png")).string(), TorchTensorToCVMat(normals_from_depth));

			if (!savedir.empty())
			{
				cv::imwrite((savedir / (std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RGBMap.cpu()));
				cv::imwrite((savedir / ("disp_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DispMap));
				cv::imwrite((savedir / ("depth_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.DepthMap));
			}		//if savedir
		}		//if use_nerf

		if (Params.use_lerf)
		{
			auto render_result = std::move(LeRFRenderer->Render(h, w, k,
				rparams,
				rays,
				render_poses[i].index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
				c2w_staticcam
			));
			if (!savedir.empty())
			{
				//!!!визуализацию similarity с каким-нибудь словом относительно случайного набора (см статью и код)
				cv::Mat relevancy_img(h, w, CV_8UC1/*CV_32FC1*/);
				torch::Tensor rel = render_result.Outputs1.Relevancy.cpu();
				#pragma omp parallel for
				for (int i = 0; i < w; i++)
					for (int j = 0; j < h; j++)
					{
						float lv = rel.index({j,i,0}).item<float>();
						relevancy_img.at<uchar>(j, i) = cv::saturate_cast<uchar>((lv>0.7?lv:0) * 255);	//[-1..1] -> [0..255]
						//relevancy_img.at<float>(j, i) = /*cv::saturate_cast<uchar>(*/lv/*(lv>0.5?lv:0)*/ /** 255*/;	//[-1..1] -> [0..255]
					}
				//cv::normalize(relevancy_img, relevancy_img, 0, 255, cv::NORM_MINMAX);
				//relevancy_img.convertTo(relevancy_img, CV_8UC1);
				cv::imwrite((savedir / ("rendered_lang_embedding" + std::to_string(i) + ".png")).string(), relevancy_img);
			}	//if savedir
		}	//if use_lerf

		//if (Params.use_nerfactor)
		//{
		//	//if (rparams.CalculateNormals)
		//	{
		//		cv::imwrite((savedir / ("rendered_norm_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RenderedNormals));
		//	}
		//	//if (rparams.UsePredNormal)
		//	{
		//		cv::imwrite((savedir / ("pred_rendered_norm_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs1.RenderedPredNormals));
		//	}
		//}	//if use_nerfactor

	}	//for render_poses
	//return std::make_pair(rgbs, disps);
}			//NeRFExecutor :: RenderPath

template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: SetLeRFPrompts(const std::string &lerf_positives, const std::vector<std::string> &lerf_negatives)
{
	std::vector<torch::Tensor> canon_texts_tensors;
	for (auto &it : lerf_negatives)
		canon_texts_tensors.push_back(ClipProcessor->EncodeText(it));
	auto negatives = Clip->EncodeText(torch::stack(canon_texts_tensors).to(Device)).to(torch::kCPU); ///[3, 768]
	//LerfNegatives = LerfNegatives / LerfNegatives.norm(2/*L2*/, -1, true);
	//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));

	auto input = ClipProcessor->EncodeText(lerf_positives);
	auto positives = Clip->EncodeText(input.unsqueeze(0).to(Device)).to(torch::kCPU);		///[1, 768]
	//LerfPositives = LerfPositives / LerfPositives.norm(2/*L2*/, -1, true);
	SetLeRFPrompts(positives, negatives);
}

template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: SetLeRFPrompts(torch::Tensor lerf_positives, torch::Tensor lerf_negatives)
{
	LeRFRenderer->SetLeRFPrompts(lerf_positives, lerf_negatives);

};

template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
std::tuple<torch::Tensor, torch::Tensor> NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: GetLeRFPrompts()
{
	return LeRFRenderer->GetLeRFPrompts();
}

template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: InitializePyramidClipEmbedding(const NeRFExecutorTrainParams &params, const CompactData &data, bool test /*= false*/)
{
	if (Params.use_lerf)
	{
		PyramidEmbedderProperties pyramid_embedder_properties; 
		pyramid_embedder_properties.ImgSize = {Params.clip_input_img_size, Params.clip_input_img_size};	//Входной размер изображения сети
		pyramid_embedder_properties.Overlap = Params.pyr_embedder_overlap;										///Доля перекрытия
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

		SetLeRFPrompts(Params.lerf_positives, Params.lerf_negatives);
	}

	if (Params.use_lerf && test) 
	{
		std::cout<<"lang embedding visualization test..."<<std::endl;
		//test
		PyramidEmbedderProperties pyramid_embedder_properties; 
		pyramid_embedder_properties.ImgSize = {Params.clip_input_img_size, Params.clip_input_img_size};	//Входной размер изображения сети
		pyramid_embedder_properties.Overlap = Params.pyr_embedder_overlap;										///Доля перекрытия
		///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0 , 1, 2...
		pyramid_embedder_properties.MaxZoomOut = std::min(log2f(data.W/Params.clip_input_img_size), log2f(data.H/Params.clip_input_img_size));
		//auto scale = Clip->GetLogitScale().exp().to(torch::kCPU);

		int img_id = 50;

		//!!!В этом месте можно распараллелить через выполнение батчами на GPU
		//!Объединив несколько cat({image_features1..image_featuresТ})
		cv::Mat test_img(data.H, data.W, CV_8UC1);
		#pragma omp parallel for
		for (int i = 0; i < data.W; i++)
			for (int j = 0; j < data.H; j++)
			{
				torch::Tensor image_features = PyramidClipEmbedding.GetPixelValue(i,j,0.5f,img_id,pyramid_embedder_properties,cv::Size(data.W, data.H)).to(torch::kCPU);
				//Уже нормировано при вычислении пирамиды normalize features???
				//image_features = image_features / image_features.norm(2/*L2*/, -1, true);
				//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
				
				////cosine similarity as logits
				//auto logits = /*scale * */torch::mm(image_features, LerfPositives.t());
				//float lv = (logits[0,0].item<float>()/*/scale*/ + 1)/2;
				//test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>(/*(lv>0.5?lv:0)*/lv * 125);	//[-1..1] -> [0..255]

				//relevancy
				auto [lerf_positives, lerf_negatives] = LeRFRenderer->GetLeRFPrompts();
				torch::Tensor rel = Relevancy(image_features, lerf_positives, lerf_negatives);
				float lv = rel.index({0,0}).item<float>();
				test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>((lv>0.7?lv:0) * 255);	//[-1..1] -> [0..255]
			}
		cv::imshow("img", TorchTensorToCVMat(data.Imgs[img_id]));
		cv::imshow("test_img", test_img);
		cv::waitKey(1);
	}
}

///
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: Train(const CompactData &data, NeRFExecutorTrainParams &params)
{
	Initialize(this->Params, data.BoundingBox);
	if (Params.use_lerf)
		InitializePyramidClipEmbedding(params, data, true);

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

		std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, false, false));
		RenderPath(data.RenderPoses, data.H, data.W, data.Focal, data.K, *render_params, 
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

			if (params.NRand != 0)
			{
				///Выбор случайных точек для батча и получение для них параметров луча
				//Вычисляем границы кадрирования один раз
				int h_start, h_end, w_start, w_end;
				if (i < params.PrecorpIters)
				{
					int dh = int(data.H / 2 * params.PrecorpFrac);
					int dw = int(data.W / 2 * params.PrecorpFrac);
					h_start = data.H / 2 - dh;
					h_end = data.H / 2 + dh - 1;
					w_start = data.W / 2 - dw;
					w_end = data.W / 2 + dw - 1;

					if (i == Start + 1)
						std::cout << "[Config] Center cropping of size " << 2 * dh << "x" << 2 * dw << " enabled until iter " << params.PrecorpIters << std::endl;
				} else {
					h_start = 0;
					h_end = data.H - 1;
					w_start = 0;
					w_end = data.W - 1;
				}

				auto [rays_o, rays_d] = GetRays(data.H, data.W, data.K, pose);

				//Прямая генерация случайных координат
				auto options = torch::TensorOptions().dtype(torch::kLong).device(Device);
				torch::Tensor rand_h = torch::randint(h_start, h_end + 1, { params.NRand }, options);
				torch::Tensor rand_w = torch::randint(w_start, w_end + 1, { params.NRand }, options);
				//Эффективное извлечение данных, создаем индексы для пакетного доступа
				//auto batch_indices = torch::stack({ rand_h, rand_w }, /*dim=*/-1);
				//Извлекаем лучи и цвета за одну операцию
				batch_rays.first = rays_o.index({ rand_h, rand_w });
				batch_rays.second = rays_d.index({ rand_h, rand_w });
				target_s = target.index({ rand_h, rand_w });

				///Вычислим CLIP эмбеддинги в точках изображения которые попали в батч
				if (Params.use_lerf)
				{
					PyramidEmbedderProperties pyramid_embedder_properties; 
					pyramid_embedder_properties.ImgSize = {Params.clip_input_img_size, Params.clip_input_img_size};	//Входной размер изображения сети
					pyramid_embedder_properties.Overlap = Params.pyr_embedder_overlap;										///Доля перекрытия
					///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0, 1, 2...
					pyramid_embedder_properties.MaxZoomOut = std::min(log2f(data.W/Params.clip_input_img_size), log2f(data.H/Params.clip_input_img_size));
					//auto select_coords_cpu = select_coords.to(torch::kCPU).to(torch::kFloat);		//!!!.item<long>()почему то не находит поэтому преобразуем во float
					target_lang_embedding = torch::ones({params.NRand, Params.lang_embed_dim}, torch::kFloat32);
					#pragma omp parallel for
					for (int idx = 0; idx < rand_h.size(0)/*NRand*/; idx++)
					{
						target_lang_embedding.index_put_({idx/*, torch::indexing::Slice()*/}, PyramidClipEmbedding.GetPixelValue(
							rand_h.index({ idx }).to(torch::kCPU).to(torch::kFloat).item<float>(),
							rand_w.index({ idx }).to(torch::kCPU).to(torch::kFloat).item<float>(),
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
		
		torch::Tensor loss = torch::full({ 1/*img_loss.sizes()*/ }, 0.f).to(Device),
			psnr = torch::full({ 1/*img_loss.sizes()*/ }, 10.f).to(Device);

		if (Params.use_nerf)
		{
			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, true, Params.calculate_normals || Params.use_pred_normal));
			auto rgb_disp_acc_extras = std::move(NeRFRenderer->Render(data.H, data.W, data.K,
				*render_params, batch_rays, torch::Tensor(), torch::Tensor()				/*либо rays либо pose c2w*/
			));
			torch::Tensor mse_loss;
			torch::Tensor img_loss;
		
			mse_loss = torch::mse_loss(rgb_disp_acc_extras.Outputs1.RGBMap, target_s.detach());
			img_loss = torch::nn::functional::huber_loss(
						rgb_disp_acc_extras.Outputs1.RGBMap,
						target_s.detach()
				);
			if (params.ReturnRaw)
				auto trans = rgb_disp_acc_extras.Raw.index({ "...", -1 });		//последний элемент последнего измерения тензора

			loss = img_loss;
			psnr = -10. * torch::log(mse_loss) / torch::log(torch::full({ 1/*img_loss.sizes()*/ }, /*value=*/10.f)).to(Device);

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

			loss.backward();
		}		//if use_nerf

		//if (Params.use_nerfactor)
		if constexpr (std::is_same_v<TNeRF, NeRFactor>)
		{
			//std::unique_ptr<NeRFactorRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, true, Params.calculate_normals || Params.use_pred_normal));
			//auto nerfactor_render_result = std::move(NeRFactorRenderer->Render(data.H, data.W, data.K,
			//	*render_params, batch_rays, torch::Tensor(), torch::Tensor()				/*либо rays либо pose c2w*/
			//));
			//if (nerfactor_render_result.Outputs1.Normals.defined() && nerfactor_render_result.Outputs1.Normals.numel() != 0)
			//{
			//	const float orientation_loss_weight = 1.f;//1e-4;
			//	//Loss that encourages that all visible normals are facing towards the camera.
			//	auto orientation_loss = OrientationLoss(nerfactor_render_result.Outputs1.Weights.unsqueeze(-1).detach(), nerfactor_render_result.Outputs1.Normals, batch_rays.second/*directions*/);
			//	loss = loss + torch::mean(orientation_loss) * orientation_loss_weight;
			//}

			////
			//if (Params.use_pred_normal)
			//{
			//	const float pred_normal_loss_weight = 1.f;//1e-3;
			//	//Loss between normals calculated from density and normals from prediction network.
			//	auto pred_normal_loss = PredNormalLoss(
			//		nerfactor_render_result.Outputs1.Weights.unsqueeze(-1).detach(),
			//		nerfactor_render_result.Outputs1.Normals.detach(),	//Градиент не течет сюда потому что хотим оптимизировать предсказанные нормали за счет вычисленных
			//		nerfactor_render_result.Outputs1.PredNormals
			//	);
			//	loss = loss + torch::mean(pred_normal_loss) * pred_normal_loss_weight;
			//}
		}		//if use_nerfactor

		torch::Tensor lang_loss;
		const float lang_loss_weight = 1e-1f;//1e-2f;
		if (Params.use_lerf)
		{
			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, true, Params.calculate_normals || Params.use_pred_normal));
			auto lerf_render_result = std::move(LeRFRenderer->Render(data.H, data.W, data.K,
				*render_params, batch_rays, torch::Tensor(), torch::Tensor()				/*либо rays либо pose c2w*/
			));
			//lang_loss = torch::nn::functional::cosine_embedding_loss(
			//	rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()),
			//	target_lang_embedding.detach().to(loss.device()),//target clip embeddings provided by PyramidEmbedder for a given set of pixels
			//	torch::ones({rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.sizes()[0]})
			//);
			lang_loss = torch::nn::functional::huber_loss(
					lerf_render_result.Outputs1.RenderedLangEmbedding.to(loss.device()),
					target_lang_embedding.detach().to(loss.device()),	//target clip embeddings provided by PyramidEmbedder for a given set of pixels
					torch::nn::functional::HuberLossFuncOptions().reduction(torch::kNone).delta(1.25)
				).sum(-1).nanmean();
			//lang_loss = torch::nn::functional::huber_loss(
			//	rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()),
			//	target_lang_embedding.detach().to(loss.device())	//target clip embeddings provided by PyramidEmbedder for a given set of pixels
			//);
			//lang_loss = - torch::mm(rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()), target_lang_embedding.detach().to(loss.device()).t()).nanmean();
			//auto lang_loss = torch::mse_loss(rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()), target_lang_embedding.to(loss.device()));
			//loss = loss + lang_loss * lang_loss_weight;
			std::cout<<"lang_loss: " << lang_loss << std::endl;
			lang_loss.backward();
		}		//if use_lerf

		try {
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
			if (Params.use_nerf)
			{
				torch::save(ExecutorEmbedder, (path / "embedder_checkpoint.pt").string());
				torch::save(Model, (path / "model_checkpoint.pt").string());
				if (!ModelFine.is_empty())
					torch::save(ModelFine, (path / "model_fine_checkpoint.pt").string());
			}
			if (Params.use_lerf)
			{
				torch::save(LangEmbedder, (path / "lang_embedder_checkpoint.pt").string());
				torch::save(LangModel, (path / "lang_model_checkpoint.pt").string());
				if (!LangModelFine.is_empty())
					torch::save(LangModelFine, (path / "lang_model_fine_checkpoint.pt").string());
			}
			torch::save(torch::full({ 1 }, /*value=*/global_step), (path / "start_checkpoint.pt").string());
			torch::save(*Optimizer.get(), (path / "optimizer_checkpoint.pt").string());

			std::cout << "Saved checkpoints at " << path.string() << std::endl;
		}

		if ((i % params.ITestset == 0) && (i > 0) && !params.TestSkip)		//если TestSkip то просто не подгружены тестовые данные 
		try {
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

			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, false, false));
			RenderPath(test_poses/*poses[i_test]).to(device)*/, data.H, data.W, data.Focal, data.K, *render_params,
				{ torch::Tensor(), torch::Tensor() }, /*c2w_staticcam*/torch::Tensor(), /*test_imgs,*/ test_save_dir
			);
			std::cout << "Saved test set" << std::endl;
		} catch (std::exception &e) {
			std::cerr << e.what() << std::endl;
		}

		if ((i % params.IVideo == 0) && i > 0)
		{
			torch::NoGradGuard no_grad;
			auto dir = params.BaseDir / "render";
			if (!std::filesystem::exists(dir))	std::filesystem::create_directories(dir);
			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, false, false));
			RenderPath(data.RenderPoses, data.H, data.W, data.Focal, data.K, *render_params, { torch::Tensor(), torch::Tensor() },
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