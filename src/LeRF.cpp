#include "LeRF.h"

LeRFImpl :: LeRFImpl(
	const int geo_feat_dim_le /*= 32*/,
	const int num_layers_le /*= 3*/,
	const int hidden_dim_le /*= 64*/,
	const int lang_embed_dim,
	const int input_ch_le /*= 0*/,
	const std::string module_name /*= "lerf"*/
) : BaseNeRFImpl(module_name), GeoFeatDimLE(geo_feat_dim_le), NumLayersLE(num_layers_le), HiddenDimLE(hidden_dim_le), LangEmbedDim(lang_embed_dim), InputChLE(input_ch_le)
{
	for (int l = 0; l < NumLayersLE; l++)
		SigmaLENet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputChLE : HiddenDimLE, (l == NumLayersLE - 1) ? (1 + GeoFeatDimLE) : HiddenDimLE/*, false*/).bias(false)));

	for (int l = 0; l < NumLayersLE; l++)
		LENet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? GeoFeatDimLE + InputChLE : HiddenDimLE, (l == NumLayersLE - 1) ? (LangEmbedDim) : HiddenDimLE/*, false*/).bias(false)));	//CLIP embedding size

	for (int i = 0; i < SigmaLENet->size(); i++)
		register_module(module_name + "_sigma_le_net_" + std::to_string(i), SigmaLENet[i]);

	//for (int l = 0; l < NumLayersLE; l++)
	//	LENet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputChLE : HiddenDimLE, (l == NumLayersLE - 1) ? (LangEmbedDim) : HiddenDimLE/*, false*/).bias(false)));	//CLIP embedding size

	for (int i = 0; i < LENet->size(); i++)
		register_module(module_name + "_le_net_" + std::to_string(i), LENet[i]);
}

torch::Tensor LeRFImpl :: forward(torch::Tensor x)
{
	////Прямо из LE hash embedding в LeRF MLP
	//torch::Tensor le;
	//if (NumLayersLE > 0 && inputs_le.numel() != 0)
	//{
	//	//lerf
	//	auto h = inputs_le;
	//	for (int i = 0; i < LENet->size(); i++)
	//	{
	//		h = LENet[i]->as<torch::nn::Linear>()->forward(h);
	//		if (i != LENet->size() - 1)
	//			h = torch::relu(h);		//!!!inplace = true
	//	}
	//	//le = h;
	//	//le = torch::tanh(h);
	//	//h = torch::sigmoid(h);
	//	le = torch::nn::functional::normalize(h, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
	//}

	////Из LE hash embedding в Sigma MLP затем через псевдо skip connection уже в LeRF MLP
	//torch::Tensor le;
	//if (NumLayersLE > 0 && inputs_le.numel() != 0)
	//{
	//	//sigma le
	//	auto h = inputs_le;
	//	for (int i = 0; i < SigmaLENet->size(); i++)
	//	{
	//		h = SigmaLENet[i]->as<torch::nn::Linear>()->forward(h);
	//		if (i != SigmaLENet->size() - 1)
	//			h = torch::relu(h);		//!!!inplace = true
	//	}
	//	auto sigma_le = h.index({ "...", 0 });
	//	auto geo_feat_le = h.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });

	//	//lerf
	//	h = torch::cat({ geo_feat_le, inputs_le }, -1);
	//	for (int i = 0; i < LENet->size(); i++)
	//	{
	//		h = LENet[i]->as<torch::nn::Linear>()->forward(h);
	//		if (i != LENet->size() - 1)
	//			h = torch::relu(h);		//!!!inplace = true
	//	}
	//	le = h;
	//	//h = torch::tanh(h);
	//	//le = torch::nn::functional::normalize(h, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
	//}

	torch::Tensor inputs_le = x;

	//Полностью независимая сетка для LE со своей плотностью
	torch::Tensor le,
		sigma_le;
	if (NumLayersLE > 0 && inputs_le.numel() != 0)
	{
		//sigma le
		auto h = inputs_le;
		for (int i = 0; i < SigmaLENet->size(); i++)
		{
			h = SigmaLENet[i]->as<torch::nn::Linear>()->forward(h);
			if (i != SigmaLENet->size() - 1)
				h = torch::relu(h);		//!!!inplace = true
		}
		sigma_le = h.index({ "...", 0 });
		auto geo_feat_le = h.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });

		//lerf
		h = torch::cat({ geo_feat_le, inputs_le }, -1);
		for (int i = 0; i < LENet->size(); i++)
		{
			h = LENet[i]->as<torch::nn::Linear>()->forward(h);
			if (i != LENet->size() - 1)
				h = torch::relu(h);		//!!!inplace = true
		}
		le = h;
		//h = torch::tanh(h);
		le = torch::nn::functional::normalize(h, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
	
		sigma_le = sigma_le.unsqueeze(-1);
	}

	auto outputs = torch::cat({ le, sigma_le }, -1);
	return outputs;
}