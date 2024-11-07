#pragma once

#include "NeRF.h"

///Language Embedded Radiance Field MLP
class LeRFImpl : public BaseNeRFImpl{
protected:
	int GeoFeatDimLE,
		NumLayersLE,
		HiddenDimLE,
		LangEmbedDim,
		InputChLE;

	torch::nn::ModuleList SigmaLENet,
		LENet;
public:
	LeRFImpl(
		const int geo_feat_dim_le = 32,
		const int num_layers_le = 3,
		const int hidden_dim_le = 64,
		const int lang_embed_dim = 768,
		const int input_ch_le = 0,
		const std::string module_name = "lerf"
	);

	virtual ~LeRFImpl(){}

	virtual torch::Tensor forward(torch::Tensor x) override;

	virtual int GetLangEmbedDim() const {return LangEmbedDim;};
};
TORCH_MODULE(LeRF);