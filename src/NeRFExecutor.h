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
		num_layers_color{ 4 },				//for color part of the HashNeRF
		hidden_dim_color{ 64 },				//for color part of the HashNeRF
		num_layers_normals{ 3 },
		hidden_dim_normals{ 64 },
		geo_feat_dim{ 15 };
	//torch::Tensor bounding_box = torch::tensor({ -4.f, -4.f, -4.f, 4.f, 4.f, 4.f });
	bool use_nerf{ true },
		use_viewdirs{ true },	//use full 5D input instead of 3D. Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		calculate_normals{ false },
		use_pred_normal{ false },	//whether to use predicted normals
		use_lerf{ true },
		thin_ray{ false };
	int n_levels{ 16 },											//for color density embedder
		n_features_per_level{ 2 },						//for color density embedder
		log2_hashmap_size{ 19 },							//for color density embedder
		base_resolution{ 16 },								//for color density embedder
		finest_resolution{ 512 },							//for color density embedder
		n_levels_le = n_levels - 2,																		//for language embedder
		n_features_per_level_le = n_features_per_level,								//for language embedder
		log2_hashmap_size_le = log2_hashmap_size - 3,									//for language embedder
		base_resolution_le = base_resolution,													//for language embedder
		finest_resolution_le = finest_resolution / 4,										//for language embedder
		clip_input_img_size{ 336 },	//Input RuClip model size
		num_layers_le{ 3 },					//Language embedder head params
		hidden_dim_le{ 64 },				//Language embedder head params
		lang_embed_dim{ 768 },			//Language embedder head params
		geo_feat_dim_le{ 32 },			//Language embedder head params
		pyr_embed_min_zoom_out{ 0 };
	torch::Device device{ torch::kCUDA };
	float learning_rate{ 5e-4f },
		pyr_embedder_overlap{0.75f};
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
		j["num_layers_color"] = num_layers_color;
		j["hidden_dim_color"] = hidden_dim_color;
		j["num_layers_normals"] = num_layers_normals;
		j["hidden_dim_normals"] = hidden_dim_normals;
		j["geo_feat_dim"] = geo_feat_dim;
		j["use_nerf"] = use_nerf;
		j["thin_ray"] = thin_ray;
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
		j["lang_embed_min_zoom_out"] = pyr_embed_min_zoom_out;
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
		j.at("num_layers_color").get_to(num_layers_color);
		j.at("hidden_dim_color").get_to(hidden_dim_color);
		j.at("num_layers_normals").get_to(num_layers_normals);
		j.at("hidden_dim_normals").get_to(hidden_dim_normals);
		j.at("geo_feat_dim").get_to(geo_feat_dim);
		j.at("use_nerf").get_to(use_nerf);
		j.at("thin_ray").get_to(thin_ray);
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
		j.at("lang_embed_min_zoom_out").get_to(pyr_embed_min_zoom_out);
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
	bool TestSkip{ false },					///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
		RenderOnly{ false },					///do not optimize, reload weights and render out render_poses path
		Ndc{ false },									///use normalized device coordinates (set for non-forward facing scenes)
		LinDisp{ false };							///sampling linearly in disparity rather than depth
	int Chunk{ 1024 * 32 },					///number of rays processed in parallel, decrease if running out of memory не влияет на качество
		NSamples{ 64 },								///number of coarse samples per ray
		NRand{ 32 * 32 * 4 },					///batch size (number of random rays per gradient step) must be < H * W
		PrecorpIters{ 0 },						///number of steps to train on central crops
		NIters{ 50000 },
		LRateDecay{ 250 },						///exponential learning rate decay (in 1000 steps)
		//logging / saving options
		IPrint{ 100 },			///frequency of console printout and metric loggin
		IImg{ 500 },				///frequency of tensorboard image logging
		IWeights{ 10000 },	///frequency of weight ckpt saving
		ITestset{ 50000 };	///frequency of testset saving
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


inline NeRFDatasetParams LoadDatasetParams (
	const std::filesystem::path &basedir,
	const torch::Device &device,
	DatasetType dataset_type = DatasetType::BLENDER,
	bool half_res = false,				///load blender synthetic data at 400x400 instead of 800x800
	bool test_skip = false,				///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
	bool white_bkgr = false				///set to render synthetic data on a white bkgd (always use for dvoxels)
)	{
	std::string dataset_type_str;
	NeRFDatasetParams data;
	//Загрузить данные
	if (dataset_type == DatasetType::BLENDER)
	{
		dataset_type_str = "BLENDER";
		data = load_blender_data(basedir, 0.f, 0.f, half_res, test_skip);
	}

	if (dataset_type == DatasetType::COLMAP)
	{
		dataset_type_str = "COLMAP";
		data = LoadFromColmapReconstruction(basedir.string());
	}

	std::cout << "Loaded " << dataset_type_str << data.Views.size() << " " << data.Views[0].H << " " << data.Views[0].W << " " << data.Views[0].Focal << " " << basedir << std::endl;
	//i_train, i_val, i_test = i_split;

	data.WhiteBgr = white_bkgr;

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
	TNeRF Model = nullptr;
	TLeRF LangModel = nullptr;
	std::unique_ptr<TNeRFRenderer> NeRFRenderer = nullptr;
	std::unique_ptr<TLeRFRenderer> LeRFRenderer = nullptr;
	std::vector<torch::Tensor> GradVars;
	std::unique_ptr<torch::optim::Adam> Optimizer;
	int Start{0},
		NImportance{0};
	float LearningRate,
		StochasticPreconditioningAlpha0;
	torch::Device Device;
	bool UseViewDirs;									//use full 5D input instead of 3D
	CLIP Clip = nullptr;
	std::shared_ptr<RuCLIPProcessor> ClipProcessor = nullptr;

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
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		torch::Tensor c2w_staticcam = torch::Tensor(),			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
		//torch::Tensor gt_imgs = torch::Tensor(),
		const std::filesystem::path savedir = ""
	);

	///
	NeRFRenderParams * FillRenderParams(
		NeRFExecutorParams executor_params,
		NeRFExecutorTrainParams executor_train_params,
		const int iter = std::numeric_limits<int>::max(),
		torch::Tensor BoundingBox = torch::Tensor(),
		const bool returm_weights = false,
		const bool calculate_normals = false
	);

	///
	void SetLeRFPrompts(const std::string &lerf_positives, const std::vector<std::string> &lerf_negatives);
	void SetLeRFPrompts(torch::Tensor lerf_positives, torch::Tensor lerf_negatives);
	std::tuple<torch::Tensor, torch::Tensor> GetLeRFPrompts();

	void InitializeTestLeRF(const NeRFExecutorTrainParams &params, const NeRFDatasetParams &data, NeRFDataset &dataset, bool test = false);

	///
	void Train(const NeRFDatasetParams &data, NeRFExecutorTrainParams &params);
	void SaveCheckpoint(const std::filesystem::path &path, const int global_step /*= 0*/);
	NeRFExecutorParams GetParams(){return Params;};
	torch::Tensor GetEmbedderBoundingBox(){return ExecutorEmbedder->GetBoundingBox();};
	torch::Tensor GetLangEmbedderBoundingBox(){return LangEmbedder->GetBoundingBox();};

};				//NeRFExecutor


template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
NeRFRenderParams * NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: FillRenderParams(
	NeRFExecutorParams executor_params,
	NeRFExecutorTrainParams executor_train_params,
	const int iter /*= std::numeric_limits<int>::max()*/,
	torch::Tensor BoundingBox /*= torch::Tensor()*/,
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
	
	render_params->ThinRay = executor_params.thin_ray;
	render_params->NSamples = executor_train_params.NSamples;
	render_params->NImportance = NImportance;
	render_params->Chunk = executor_train_params.Chunk;
	render_params->ReturnRaw = executor_train_params.ReturnRaw;
	render_params->LinDisp = executor_train_params.LinDisp;
	render_params->Perturb = 0.f;
	//render_params.WhiteBkgr = params.WhiteBkgr;
	render_params->RawNoiseStd = 0.;
	render_params->Ndc = executor_train_params.Ndc;
	render_params->UseViewdirs = UseViewDirs;
	render_params->ReturnWeights = returm_weights;
	render_params->RenderFactor = executor_train_params.RenderFactor;

	render_params->BoundingBox = BoundingBox;
	render_params->RawNoiseStd = std::max(0.0f, 1.0f - static_cast<float>(iter) / (static_cast<float>(executor_train_params.NIters) / 8));  //Локальная регуляризация плотности помогает избежать артефактов типа "облаков" затухает за n_iters / 3 итераций
	render_params->StochasticPreconditioningAlpha = StochasticPreconditioningAlpha0 * std::max(0.0f, 1.0f - static_cast<float>(iter) / (static_cast<float>(executor_train_params.NIters) / 6));  //добавляет шум к входу сети (координатам точек). Уменьшает чувствительность к инициализации. Помогает избежать "плавающих" артефактов
	
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
			NeRFRenderer = std::make_unique<TNeRFRenderer> (ExecutorEmbedder, ExecutorEmbeddirs, Model);
	}

	if (params.use_lerf)
	{
		//if constexpr (std::is_same_v<TLeRFRenderer, LeRFRenderer>)
			LeRFRenderer = std::make_unique<TLeRFRenderer> (LangEmbedder, LangModel);
	}

	//if constexpr (std::is_same_v<TNeRF, NeRF>)
	//	Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-7)/*.weight_decay(0.001)*/.betas(std::make_tuple(0.9, 0.999)));
	Optimizer = std::make_unique<torch::optim::Adam>(GradVars, torch::optim::AdamOptions(params.learning_rate).eps(1e-15).betas(std::make_tuple(0.9, 0.99)));
	
	if (/*Проверить наличие файлов*/
		std::filesystem::exists(params.ft_path / "start_checkpoint.pt") &&
		std::filesystem::exists(params.ft_path / "optimizer_checkpoint.pt") &&
		((params.use_nerf && std::filesystem::exists(params.ft_path / "model_checkpoint.pt")) || !params.use_nerf) &&
		((params.use_lerf && std::filesystem::exists(params.ft_path / "lang_embedder_checkpoint.pt")) || !params.use_lerf)
	) {
		std::cout << "restoring parameters from checkpoint..." << std::endl;

		if (params.use_nerf)
		{
			if (std::filesystem::exists(params.ft_path / "embedder_checkpoint.pt"))
				torch::load(ExecutorEmbedder, (params.ft_path / "embedder_checkpoint.pt").string());
			torch::load(Model, (params.ft_path / "model_checkpoint.pt").string());
		}

		if (params.use_lerf)
		{
			if (std::filesystem::exists(params.ft_path / "lang_embedder_checkpoint.pt"))
				torch::load(LangEmbedder, (params.ft_path / "lang_embedder_checkpoint.pt").string());
			torch::load(LangModel, (params.ft_path / "lang_model_checkpoint.pt").string());
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
		}
		
		if (params.use_lerf)
		{
			LangEmbedder->Initialize();
			Trainable::Initialize(LangModel);
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

	//for stochastic preconditioning
	std::vector<torch::Tensor> bounds = torch::split(bounding_box, { 3, 3 }, -1);
	auto min_bound = bounds[0];
	auto max_bound = bounds[1];
	auto bbox_diag = torch::norm(max_bound - min_bound).item<float>();
	StochasticPreconditioningAlpha0 = 0.02f * bbox_diag;
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
			/*Rays*/ {torch::Tensor(), torch::Tensor(), torch::Tensor()},
			render_pose.to(Device).index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }),
			/*rparams.C2wStaticCam*/ torch::Tensor()
		);
	}
	if (Params.use_lerf)
	{
		lerf_render_result = LeRFRenderer->Render(h, w, k1,
			rparams,
			/*Rays*/ {torch::Tensor(), torch::Tensor(), torch::Tensor()},
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
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rays /*= { torch::Tensor(), torch::Tensor() }*/,			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
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
			render_result.Outputs.DepthMap = (render_result.Outputs.DepthMap - render_result.Near) / (render_result.Far - render_result.Near);

			//torch::Tensor normals_from_depth = NormalMapFromDepthMap(render_result.Outputs1.DepthMap.detach().cpu());
			//if (!savedir.empty())
			//	cv::imwrite((savedir / ("normals_from_depth_" + std::to_string(disps.size() - 1) + ".png")).string(), TorchTensorToCVMat(normals_from_depth));

			if (!savedir.empty())
			{
				cv::imwrite((savedir / (std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs.RGBMap.cpu()));
				cv::imwrite((savedir / ("disp_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs.DispMap));
				cv::imwrite((savedir / ("depth_" + std::to_string(i) + ".png")).string(), TorchTensorToCVMat(render_result.Outputs.DepthMap));
			}		//if savedir
		}		//if use_nerf

		if (Params.use_lerf)
		{
			auto render_result = std::move(LeRFRenderer->Render(h, w, k,
				rparams,
				rays,
				render_poses[i].index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 4) }), c2w_staticcam));

			if (!savedir.empty())
			{
				torch::Tensor rel = render_result.Outputs.Relevancy.reshape({ h, w, 2 });
				torch::Tensor pos_probs = rel.index({ "...", 0 });  // [H, W]
				pos_probs = pos_probs.mul(255).to(torch::kU8).cpu();
				cv::Mat relevancy_img(h, w, CV_8UC1, pos_probs.data_ptr());
				cv::Mat colored_map;
				cv::applyColorMap(relevancy_img, colored_map, cv::COLORMAP_JET);
				cv::imwrite((savedir / ("relevancy_" + std::to_string(i) + ".png")).string(), colored_map);
			}
		}

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
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: InitializeTestLeRF(const NeRFExecutorTrainParams &params, const NeRFDatasetParams &data, NeRFDataset &dataset, bool test /*= false*/)
{
	try {
		if (Params.use_lerf)
		{

			SetLeRFPrompts(Params.lerf_positives, Params.lerf_negatives);
		}

		if (Params.use_lerf && test)
		{
			std::cout << "lang embedding visualization test..." << std::endl;
			//test
			PyramidEmbedderProperties pyramid_embedder_properties;
			pyramid_embedder_properties.ImgSize = { Params.clip_input_img_size, Params.clip_input_img_size };	//Входной размер изображения сети
			pyramid_embedder_properties.Overlap = Params.pyr_embedder_overlap;										///Доля перекрытия
			pyramid_embedder_properties.MinZoomOut = Params.pyr_embed_min_zoom_out;		//0 or -1
			///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0 , 1, 2...
			int wmax = 0,
				hmax = 0;
			for (auto &view : data.Views)
			{
				if (view.H > hmax)
					hmax = view.H;
				if (view.W > wmax)
					wmax = view.W;
			}
			pyramid_embedder_properties.MaxZoomOut = std::min(log2f(wmax / Params.clip_input_img_size), log2f(hmax / Params.clip_input_img_size));
			//auto scale = Clip->GetLogitScale().exp().to(torch::kCPU);

			int img_id = 0;

			//!!!В этом месте можно распараллелить через выполнение батчами на GPU
			//!Объединив несколько cat({image_features1..image_featuresТ})
			cv::Mat test_img(data.Views[img_id].H, data.Views[img_id].W, CV_8UC1);
			auto [lerf_positives, lerf_negatives] = LeRFRenderer->GetLeRFPrompts();
			#pragma omp parallel for
			for (int i = 0; i < data.Views[img_id].W; i++)
				for (int j = 0; j < data.Views[img_id].H; j++)
				{
					torch::Tensor image_features = dataset.GetPyramidClipEmbedding()->GetPixelValue(i, j, 0.5f, img_id, pyramid_embedder_properties, cv::Size(data.Views[img_id].W, data.Views[img_id].H)).to(torch::kCPU);
					//Уже нормировано при вычислении пирамиды normalize features???
					//image_features = image_features / image_features.norm(2/*L2*/, -1, true);
					//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));

					////cosine similarity as logits
					//auto logits = /*scale * */torch::mm(image_features, LerfPositives.t());
					//float lv = (logits[0,0].item<float>()/*/scale*/ + 1)/2;
					//test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>(/*(lv>0.5?lv:0)*/lv * 125);	//[-1..1] -> [0..255]

					//relevancy
					torch::Tensor rel = Relevancy(image_features, lerf_positives, lerf_negatives);
					float lv = rel.index({ 0,0 }).item<float>();
					test_img.at<uchar>(j, i) = cv::saturate_cast<uchar>(lv * 255);	//[-1..1] -> [0..255]
				}
			cv::imshow("img", cv::imread(data.Views[img_id].ImagePath.string(), cv::IMREAD_UNCHANGED));
			cv::Mat colored_map;
			cv::applyColorMap(test_img, colored_map, cv::COLORMAP_JET);
			cv::imshow("test_img", colored_map);
			cv::waitKey(1);
		}
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
}

///
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: Train(const NeRFDatasetParams &data, NeRFExecutorTrainParams &params)
{
	Initialize(this->Params, data.BoundingBox);

	LeRFDatasetParams lerf_data_params;
	lerf_data_params.UseLerf = Params.use_lerf;
	lerf_data_params.clip_input_img_size = Params.clip_input_img_size;
	lerf_data_params.lang_embed_dim = Params.lang_embed_dim;
	lerf_data_params.MinZoomOut = Params.pyr_embed_min_zoom_out;
	lerf_data_params.pyr_embedder_overlap = Params.pyr_embedder_overlap;
	lerf_data_params.PyramidClipEmbeddingSaveDir = params.PyramidClipEmbeddingSaveDir;	
	NeRFDataset dataset(data, lerf_data_params, params.NRand, params.PrecorpIters, params.PrecorpFrac, Device, Clip, ClipProcessor);

	if (Params.use_lerf)
		InitializeTestLeRF(params, data, dataset, true);

	int global_step = Start;
	for (int i = this->Start + 1; i < params.NIters; i++)
	{
		auto time0 = std::chrono::steady_clock::now();

		dataset.SetCurrentIter(i);		//for precorp

		auto batch = dataset.get_batch({});

		//Core optimization loop
		Optimizer->zero_grad();
		
		torch::Tensor loss = torch::full({ 1/*img_loss.sizes()*/ }, 0.f).to(Device),
			psnr = torch::full({ 1/*img_loss.sizes()*/ }, 10.f).to(Device);

		if (Params.use_nerf)
		{
			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, i, data.BoundingBox, true, Params.calculate_normals || Params.use_pred_normal));
			auto rgb_disp_acc_extras = std::move(NeRFRenderer->Render(0, 0, torch::Tensor(),/*либо rays либо data.H, data.W, data.K,*/
				*render_params, {batch.data.rays_o, batch.data.rays_d, batch.data.cone_angle}, torch::Tensor(), torch::Tensor()				/*либо rays либо pose c2w*/
			));
			torch::Tensor mse_loss;
			torch::Tensor img_loss;
		
			mse_loss = torch::mse_loss(rgb_disp_acc_extras.Outputs.RGBMap, batch.target.target_s.detach());
			img_loss = torch::nn::functional::huber_loss(
						rgb_disp_acc_extras.Outputs.RGBMap,
						batch.target.target_s.detach()
				);
			if (params.ReturnRaw)
				auto trans = rgb_disp_acc_extras.Raw.index({ "...", -1 });		//последний элемент последнего измерения тензора

			loss = img_loss;
			{
				torch::NoGradGuard no_grad;
				psnr = -10. * torch::log(mse_loss) / torch::log(torch::full({ 1/*img_loss.sizes()*/ }, /*value=*/10.f)).to(Device);
			}

			//add Total Variation loss
			if constexpr (std::is_same_v<TEmbedder, HashEmbedder>)
				if (i < params.NIters/2)
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
			//std::unique_ptr<NeRFactorRenderParams> render_params(FillRenderParams(Params, params, data.Near, data.Far, i, data.BoundingBox, true, Params.calculate_normals || Params.use_pred_normal));
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
			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, i, data.BoundingBox, true, Params.calculate_normals || Params.use_pred_normal));
			auto lerf_render_result = std::move(LeRFRenderer->Render(0, 0, torch::Tensor(),/*либо rays либо data.H, data.W, data.K,*/
				*render_params, { batch.data.rays_o, batch.data.rays_d, batch.data.cone_angle }, torch::Tensor(), torch::Tensor()				/*либо rays либо pose c2w*/
			));

			//lang_loss = torch::nn::functional::cosine_embedding_loss(
			//	rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.to(loss.device()),
			//	target_lang_embedding.detach().to(loss.device()),//target clip embeddings provided by PyramidEmbedder for a given set of pixels
			//	torch::ones({rgb_disp_acc_extras.Outputs1.RenderedLangEmbedding.sizes()[0]})
			//);
			lang_loss = torch::nn::functional::huber_loss(
					lerf_render_result.Outputs.RenderedLangEmbedding.to(loss.device()),
					batch.target.target_lang_embedding.detach().to(loss.device()),	//target clip embeddings provided by PyramidEmbedder for a given set of pixels
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
			SaveCheckpoint(path, global_step);
			std::cout << "Saved checkpoints at " << path.string() << std::endl;
		}

		if ((i % params.ITestset == 0) && (i > 0) && !params.TestSkip)		//если TestSkip то просто не подгружены тестовые данные 
		try {
			auto test_save_dir = params.BaseDir; /* / ("testset_" + std::to_string(i)) */;
			if (!std::filesystem::exists(test_save_dir))	std::filesystem::create_directories(test_save_dir);
			//Есть torch::Tensor poses и есть std::vector<torch::Tensor> data.Poses
			std::cout << "poses shape " << data.Views[0].Pose.sizes() << "; test poses count " << data.SplitsIdx[2] << std::endl;   //	std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			torch::NoGradGuard no_grad;		///Закоментировал для вычисления нормалей

			std::vector<torch::Tensor> test_poses,
				test_imgs;
			float test_h = data.Views[0].H,
				test_w = data.Views[0].W,
				test_focal = data.Views[0].Focal;
			torch::Tensor	test_k = data.Views[0].K.clone();

			//std::vector<std::string> Splits = { "train", "val", "test" }; std::vector<int> SplitsIdx = { 0,0,0 };
			//reset_near_plane: whether to reset the near plane to 0.0 during inference.The near plane can be
			//helpful for reducing floaters during training, but it can cause clipping artifacts during
			//inference when an evaluation or viewer camera moves closer to the object.
			if (data.SplitsIdx[2] == data.SplitsIdx[1] || data.SplitsIdx[2] == 0)
			{
				for (int k = 0; k < data.SplitsIdx[0]; k++)
					test_poses.push_back(data.Views[k].Pose.to(Device));
			} else {
				for (int k = data.SplitsIdx[0] + data.SplitsIdx[1]; k < data.SplitsIdx[0] + data.SplitsIdx[1] + data.SplitsIdx[2]; k++)
					test_poses.push_back(data.Views[k].Pose.to(Device));
			}

			std::unique_ptr<NeRFRenderParams> render_params(FillRenderParams(Params, params, std::numeric_limits<int>::max(), data.BoundingBox, false, false));
			RenderPath(test_poses/*poses[i_test]).to(device)*/, test_h, test_w, test_focal, test_k, *render_params,
				{ torch::Tensor(), torch::Tensor(), torch::Tensor() }, /*c2w_staticcam*/torch::Tensor(), /*test_imgs,*/ test_save_dir
			);
			std::cout << "Saved test set" << std::endl;
		} catch (std::exception &e) {
			std::cerr << e.what() << std::endl;
		}

		if (i % params.IPrint == 0)
			std::cout << "[TRAIN] Iter: " << i << " of " << params.NIters << " lr = " << new_lrate << " Loss: " << loss.item() << " PSNR: " << psnr.item() << std::endl;

		global_step++;
	}	//for (int i = this->Start + 1; i < params.NIters; i++)
}			//NeRFExecutor :: Train


///
template <typename TEmbedder, typename TEmbedDirs, typename TNeRF, typename TNeRFRenderer,
	typename TLeRFEmbedder, typename TLeRF, typename TLeRFRenderer>
void NeRFExecutor <TEmbedder, TEmbedDirs, TNeRF, TNeRFRenderer, TLeRFEmbedder, TLeRF, TLeRFRenderer> :: SaveCheckpoint(const std::filesystem::path &path, const int global_step /*= 0*/)
{
	if (!std::filesystem::exists(path))	std::filesystem::create_directories(path);
	if (Params.use_nerf)
	{
		torch::save(ExecutorEmbedder, (path / "embedder_checkpoint.pt").string());
		torch::save(Model, (path / "model_checkpoint.pt").string());
	}
	if (Params.use_lerf)
	{
		torch::save(LangEmbedder, (path / "lang_embedder_checkpoint.pt").string());
		torch::save(LangModel, (path / "lang_model_checkpoint.pt").string());
	}
	torch::save(torch::full({ 1 }, /*value=*/global_step), (path / "start_checkpoint.pt").string());
	torch::save(*Optimizer.get(), (path / "optimizer_checkpoint.pt").string());
}