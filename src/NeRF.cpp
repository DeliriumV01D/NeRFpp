#include "nerf.h"

EmbedderImpl :: EmbedderImpl(const std::string& module_name, int num_freqs, float max_freq_log2, bool include_input /*= true*/, int input_dims /*= 3*/, bool log_sampling /*= true*/)
	: torch::nn::Module(module_name), NumFreqs(num_freqs), MaxFreq(max_freq_log2), IncludeInput(include_input), InputDims(input_dims), LogSampling(log_sampling)
{
	if (IncludeInput)
		OutputDims += InputDims;
	OutputDims += NumFreqs * 2/*sin(x), cos(x)*/ * InputDims;

	for (int i = 0; i < NumFreqs; i++)
	{
		if (LogSampling)
		{
			FreqBands.push_back(powf(2.f, MaxFreq / (NumFreqs - 1) * i));
		}	else {
			FreqBands.push_back(powf(2.f, 0.f) + (pow(2.f, MaxFreq) - powf(2.f, 0.f)) / (NumFreqs - 1) * i);
		}
	}
}

torch::Tensor EmbedderImpl :: forward(torch::Tensor x)
{
	torch::Tensor outputs;
	if (IncludeInput)
	{
		outputs = x;
	}	else {
		///!!!Надо как-то проинициалиировать outputs
	}

	//!!!Подготовить все массивы и разом объединить  как в BatchifyRays
	for (auto& freq : FreqBands)
	{
		outputs = torch::cat({ outputs, torch::sin(x * freq) }, -1);
		outputs = torch::cat({ outputs, torch::cos(x * freq) }, -1);
	}

	return outputs;
}



NeRFImpl :: NeRFImpl(
	const int d /*= 8*/,
	const int w /*= 256*/,
	const int input_ch /*= 3*/,
	const int input_ch_views /*= 3*/,
	const int output_ch /*= 4*/,
	const std::set<int>& skips /*= std::set<int>{ 4 }*/,
	const bool use_viewdirs /*= false*/,
	const std::string module_name /*= "nerf"*/
) : Trainable(module_name), D(d), W(w), InputCh(input_ch), InputChViews(input_ch_views), OutputCh(output_ch), Skips(skips), UseViewDirs(use_viewdirs)
{
	PtsLinears->push_back(torch::nn::Linear(input_ch, w));
	for (int i = 0; i < (d - 1); i++)
		if (skips.find(i) == skips.end())
			PtsLinears->push_back(torch::nn::Linear(w, w));
		else
			PtsLinears->push_back(torch::nn::Linear(w + input_ch, w));

	//Implementation according to the official code release(https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
	ViewsLinears->push_back(torch::nn::Linear(input_ch_views + w, w / 2));

	////Implementation according to the paper
	//	ViewLinears->push_back(torch::nn::Linear(input_ch_views + w, w/2));
	//	for (int i = 0; i < d/2; i++)
	//		ViewLinears->push_back(torch::nn::Linear(w/2, w/2));

	if (use_viewdirs)
	{
		FeatureLinear = torch::nn::Linear(w, w);
		AlphaLinear = torch::nn::Linear(w, 1);
		RGBLinear = torch::nn::Linear(w / 2, 3);
	}	else {
		OutputLinear = torch::nn::Linear(W, output_ch);
	}

	for (int i = 0; i < PtsLinears->size(); i++)
		register_module(module_name + "_pts_linears_" + std::to_string(i), PtsLinears[i]);

	for (int i = 0; i < ViewsLinears->size(); i++)
		register_module(module_name + "_views_linears_" + std::to_string(i), ViewsLinears[i]);

	if (use_viewdirs)
	{
		register_module(module_name + "_feature_linear", FeatureLinear);
		register_module(module_name + "_alpha_linear", AlphaLinear);
		register_module(module_name + "_rgb_linear", RGBLinear);
	}	else {
		register_module(module_name + "_output_linear", OutputLinear);
	}
}

torch::Tensor NeRFImpl :: forward(torch::Tensor x) //override
{
	std::vector<torch::Tensor> splits = torch::split(x, { InputCh, InputChViews }, -1);
	auto input_pts = splits[0];
	auto input_views = splits[1];
	auto h = input_pts;
	for (int i = 0; i < PtsLinears->size(); i++)
	{
		h = PtsLinears[i]->as<torch::nn::Linear>()->forward(h);
		h = torch::relu(h);

		if (Skips.find(i) != Skips.end())
			h = torch::cat({ input_pts, h }, -1);
	}

	torch::Tensor outputs;
	if (UseViewDirs)
	{
		auto alpha = AlphaLinear(h);
		auto feature = FeatureLinear(h);
		h = torch::cat({ feature, input_views }, -1);

		for (int i = 0; i < ViewsLinears->size(); i++)
		{
			h = ViewsLinears[i]->as<torch::nn::Linear>()->forward(h);
			h = torch::relu(h);
		}
		auto rgb = RGBLinear(h);
		outputs = torch::cat({ rgb, alpha }, -1);
	}	else {
		outputs = OutputLinear(h);
	}
	return outputs;
}

///x.sizes()[0] должно быть кратно размеру chunk
torch::Tensor NeRFImpl :: Batchify(torch::Tensor x, const int chunk)
{
	torch::Tensor result;

	if (chunk <= 0)
		return forward(x);

	for (int i = 0; i < x.sizes()[0]; i += chunk)
	{
		if (i == 0)
			result = forward(x.index({ torch::indexing::Slice(i, i + chunk) }));
		else
			result = torch::cat({ result, forward(x.index({ torch::indexing::Slice(i, i + chunk) })) }, 0);
	}

	return result;
}