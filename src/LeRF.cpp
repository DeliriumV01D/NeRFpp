#include "LeRF.h"

LeRFImpl :: LeRFImpl(
	const int num_layers /*= 3*/,
	const int hidden_dim /*= 64*/,
	const int geo_feat_dim /*= 15*/,
	const int num_layers_color /*= 4*/,
	const int hidden_dim_color /*= 64*/,
	const bool use_pred_normal /*= true*/,
	const int num_layers_normals /*= 3*/,
	const int hidden_dim_normals /*= 64*/,
	const int input_ch /*= 3*/,
	const int input_ch_views /*= 3*/,
	const int num_layers_le /*= 3*/,
	const int hidden_dim_le /*= 64*/,
	const int lang_embed_dim,
	const int input_ch_le /*= 0*/,
	const std::string module_name /*= "lerf"*/
) : NeRFSmallImpl(num_layers, hidden_dim, geo_feat_dim, num_layers_color, hidden_dim_color, use_pred_normal, num_layers_normals,
	hidden_dim_normals, input_ch, input_ch_views, module_name),
	NumLayersLE(num_layers_le), HiddenDimLE(hidden_dim_le), LangEmbedDim(lang_embed_dim), InputChLE(input_ch_le)
{
	for (int l = 0; l < NumLayersLE; l++)
		SigmaLENet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputChLE : HiddenDimLE, (l == NumLayersLE - 1) ? (1 + GeoFeatDim) : HiddenDimLE/*, false*/).bias(false)));

	for (int l = 0; l < NumLayersLE; l++)
		LENet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? GeoFeatDim + InputChLE : HiddenDimLE, (l == NumLayersLE - 1) ? (LangEmbedDim) : HiddenDimLE/*, false*/).bias(false)));	//CLIP embedding size

	for (int i = 0; i < SigmaLENet->size(); i++)
		register_module(module_name + "_sigma_le_net_" + std::to_string(i), SigmaLENet[i]);

	//for (int l = 0; l < NumLayersLE; l++)
	//	LENet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputChLE : HiddenDimLE, (l == NumLayersLE - 1) ? (LangEmbedDim) : HiddenDimLE/*, false*/).bias(false)));	//CLIP embedding size

	for (int i = 0; i < LENet->size(); i++)
		register_module(module_name + "_le_net_" + std::to_string(i), LENet[i]);
}

torch::Tensor LeRFImpl :: forward(torch::Tensor x)
{
	torch::Tensor input_pts_views,
		inputs_le;

	//if (InputChLE > 0)
	//{
		std::vector<torch::Tensor> splits = torch::split(x, { InputCh, InputChLE, InputChViews}, -1);
		auto input_pts = splits[0];
		auto input_views = splits[2];
		input_pts_views = torch::cat({ splits[0], splits[2] }, -1);
		inputs_le = splits[1];
	//} else {
	//	input_pts_views = x;
	//}
		
	//Можно переиспользовать если извлечь оттуда geo_feat
	auto output = NeRFSmallImpl::forward(input_pts_views);

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

	auto outputs = torch::cat({ output, sigma_le, le }, -1);
	return outputs;
}