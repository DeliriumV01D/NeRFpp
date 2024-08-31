#pragma once

#include "NeRF.h"

///Language Embedded Radiance Field Embedders
template <typename TEmbedder>
class LeRFEmbedderImpl : public BaseEmbedderImpl {
	TEmbedder ColorDensEmbedder;
	TEmbedder LangEmbedder;
protected:
public:

	LeRFEmbedderImpl(
		const std::string &module_name,
		torch::Tensor bounding_box,
		const int n_levels = 16,
		const int n_features_per_level = 2,
		const int log2_hashmap_size = 19,
		const int base_resolution = 16,
		const int finest_resolution = 512,
		const int n_levels_le = n_levels - 2,
		const int n_features_per_level_le = n_features_per_level,
		const int log2_hashmap_size_le = log2_hashmap_size - 3,
		const int base_resolution_le = base_resolution,
		const int finest_resolution_le = finest_resolution
	) : BaseEmbedderImpl(module_name),
		ColorDensEmbedder("_color_dens", bounding_box, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, finest_resolution),
		LangEmbedder("_lang", bounding_box, n_levels_le, n_features_per_level_le, log2_hashmap_size_le, base_resolution_le, finest_resolution_le)
	{
		register_module(module_name + "_color_dens", ColorDensEmbedder);
		register_module(module_name + "_lang", LangEmbedder);
	};
	virtual ~LeRFEmbedderImpl() {}

	void Initialize()
	{
		ColorDensEmbedder->Initialize();
		LangEmbedder->Initialize();
	};
	int GetNLevels() const { return ColorDensEmbedder->NLevels; }
	int	GetNFeaturesPerLevel() const { return ColorDensEmbedder->NFeaturesPerLevel; }
	int GetLog2HashmapSize() const { return ColorDensEmbedder->Log2HashmapSize; }
	int GetBaseResolution() const { return ColorDensEmbedder->BaseResolution; }
	int GetFinestResolution() const { return ColorDensEmbedder->FinestResolution; }
	torch::Tensor GetBoundingBox() const { return ColorDensEmbedder->BoundingBox; }
	int GetOutputDims() override { return ColorDensEmbedder->OutputDims; }
	int GetNLevelsLE() const { return LangEmbedder->NLevels; }
	int	GetNFeaturesPerLevelLE() const { return LangEmbedder->NFeaturesPerLevel; }
	int GetLog2HashmapSizeLE() const { return LangEmbedder->Log2HashmapSize; }
	int GetBaseResolutionLE() const { return LangEmbedder->BaseResolution; }
	int GetFinestResolutionLE() const { return LangEmbedder->FinestResolution; }
	torch::Tensor GetBoundingBoxLE() const { return LangEmbedder->BoundingBox; }
	int GetOutputDimsLE() { return LangEmbedder->OutputDims; }

	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) override
	{
		auto col_dens_embedder =  ColorDensEmbedder->forward(x);
		auto lang_embedder = LangEmbedder->forward(x);
		return std::make_pair(torch::cat({col_dens_embedder.first, lang_embedder.first}, -1), torch::cat({col_dens_embedder.second, lang_embedder.second}, -1));
	};
};

///TORCH_MODULE(LeRFEmbedder <TEmbedder>);
template <typename TEmbedder>
class LeRFEmbedder : public torch::nn::ModuleHolder<LeRFEmbedderImpl<TEmbedder>> { /* NOLINT */
public:
	using torch::nn::ModuleHolder<LeRFEmbedderImpl<TEmbedder>>::ModuleHolder;
	using Impl TORCH_UNUSED_EXCEPT_CUDA = LeRFEmbedderImpl<TEmbedder>;
};


///Language Embedded Radiance Field MLP
class LeRFImpl : public NeRFSmallImpl {
protected:
	int NumLayersLE,
		HiddenDimLE,
		LangEmbedDim,
		InputChLE;

	torch::nn::ModuleList SigmaLENet,
		LENet;
public:
	LeRFImpl(
		const int num_layers = 3,
		const int hidden_dim = 64,
		const int geo_feat_dim = 15,
		const int num_layers_color = 4,
		const int hidden_dim_color = 64,
		const bool use_pred_normal = true,
		const int num_layers_normals = 3,
		const int hidden_dim_normals = 64,
		const int input_ch = 3,
		const int input_ch_views = 3,
		const int num_layers_le = 3,
		const int hidden_dim_le = 64,
		const int lang_embed_dim = 768,
		const int input_ch_le = 0,
		const std::string module_name = "lerf"
	);

	virtual ~LeRFImpl() override {}

	virtual torch::Tensor forward(torch::Tensor x) override;
};
TORCH_MODULE(LeRF);