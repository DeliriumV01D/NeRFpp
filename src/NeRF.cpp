#include "NeRF.h"

///
EmbedderImpl :: EmbedderImpl(const std::string& module_name, int num_freqs, float max_freq_log2, bool include_input /*= true*/, int input_dims /*= 3*/, bool log_sampling /*= true*/)
	: BaseEmbedderImpl(module_name), NumFreqs(num_freqs), MaxFreq(max_freq_log2), IncludeInput(include_input), InputDims(input_dims), LogSampling(log_sampling)
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

std::pair<torch::Tensor, torch::Tensor> EmbedderImpl :: forward(torch::Tensor x)
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
	return std::make_pair(outputs, torch::Tensor());
}

NeRFImpl :: NeRFImpl(
	const int d /*= 8*/,
	const int w /*= 256*/,
	const int input_ch /*= 3*/,
	const int input_ch_views /*= 3*/,
	const int output_ch /*= 4*/,
	const std::set<int> &skips /*= std::set<int>{ 4 }*/,
	const bool use_viewdirs /*= false*/,
	const std::string module_name /*= "nerf"*/
) : BaseNeRFImpl(module_name), D(d), W(w), InputCh(input_ch), InputChViews(input_ch_views), OutputCh(output_ch), Skips(skips), UseViewDirs(use_viewdirs)
{
	PtsLinears->push_back(torch::nn::Linear(input_ch, w));
	for (int i = 0; i < (d - 1); i++)
		if (skips.find(i) == skips.end())
			PtsLinears->push_back(torch::nn::Linear(w, w));
		else
			PtsLinears->push_back(torch::nn::Linear(w + input_ch, w));

	if (use_viewdirs)
	{
		//Implementation according to the official code release(https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
		ViewsLinears->push_back(torch::nn::Linear(input_ch_views + w, w / 2));

		////Implementation according to the paper
		//	ViewLinears->push_back(torch::nn::Linear(input_ch_views + w, w/2));
		//	for (int i = 0; i < d/2; i++)
		//		ViewLinears->push_back(torch::nn::Linear(w/2, w/2));

		FeatureLinear = torch::nn::Linear(w, w);
		AlphaLinear = torch::nn::Linear(w, 1);
		RGBLinear = torch::nn::Linear(w / 2, 3);
	}	else {
		OutputLinear = torch::nn::Linear(w + input_ch, output_ch);		//"Skip connection" added for better convergence
	}

	for (int i = 0; i < PtsLinears->size(); i++)
		register_module(module_name + "_pts_linears_" + std::to_string(i), PtsLinears[i]);

	if (use_viewdirs)
	{
		for (int i = 0; i < ViewsLinears->size(); i++)
			register_module(module_name + "_views_linears_" + std::to_string(i), ViewsLinears[i]);

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
		h = torch::cat({ h, input_pts }, -1);		//"Skip connection" added for better convergence
		outputs = OutputLinear(h);
	}
	return outputs;
}



///
std::pair<torch::Tensor, torch::Tensor> SHEncoderImpl :: forward(torch::Tensor input)
{
	auto device = input.device();
	if (C0.device().str() != device.str())
		C0 = C0.to(device);
	if (C1.device().str() != device.str())
		C1 = C1.to(device);
	if (C2.device().str() != device.str())
		C2 = C2.to(device);
	if (C3.device().str() != device.str())
		C3 = C3.to(device);
	if (C4.device().str() != device.str())
		C4 = C4.to(device);

	std::vector<int64_t> sz = input.sizes().vec();
	sz.pop_back();
	sz.push_back(OutputDims);

	auto result = torch::empty(sz, input.dtype()).to(device);
	std::vector<torch::Tensor> v = input.unbind(-1);
	auto x = v[0];
	auto y = v[1];
	auto z = v[2];

	result.index_put_({ "...", 0 }, C0);
	if (Degree > 1)
	{
		result.index_put_({ "...", 1 }, -C1 * y);
		result.index_put_({ "...", 2 }, C1 * z);
		result.index_put_({ "...", 3 }, -C1 * x);
		torch::Tensor xx, yy, zz, xy, yz, xz;
		if (Degree > 2)
		{
			xx = x * x;
			yy = y * y;
			zz = z * z;
			xy = x * y;
			yz = y * z;
			xz = x * z;
			result.index_put_({ "...", 4 }, C2[0] * xy);
			result.index_put_({ "...", 5 }, C2[1] * yz);
			result.index_put_({ "...", 6 }, C2[2] * (2.0 * zz - xx - yy));
			result.index_put_({ "...", 7 }, C2[3] * xz);
			result.index_put_({ "...", 8 }, C2[4] * (xx - yy));
			if (Degree > 3)
			{
				result.index_put_({ "...", 9 }, C3[0] * y * (3 * xx - yy));
				result.index_put_({ "...", 10 }, C3[1] * xy * z);
				result.index_put_({ "...", 11 }, C3[2] * y * (4 * zz - xx - yy));
				result.index_put_({ "...", 12 }, C3[3] * z * (2 * zz - 3 * xx - 3 * yy));
				result.index_put_({ "...", 13 }, C3[4] * x * (4 * zz - xx - yy));
				result.index_put_({ "...", 14 }, C3[5] * z * (xx - yy));
				result.index_put_({ "...", 15 }, C3[6] * x * (xx - 3 * yy));
				if (Degree > 4)
				{
					result.index_put_({ "...", 16 }, C4[0] * xy * (xx - yy));
					result.index_put_({ "...", 17 }, C4[1] * yz * (3 * xx - yy));
					result.index_put_({ "...", 18 }, C4[2] * xy * (7 * zz - 1));
					result.index_put_({ "...", 19 }, C4[3] * yz * (7 * zz - 3));
					result.index_put_({ "...", 20 }, C4[4] * (zz * (35 * zz - 30) + 3));
					result.index_put_({ "...", 21 }, C4[5] * xz * (7 * zz - 3));
					result.index_put_({ "...", 22 }, C4[6] * (xx - yy) * (7 * zz - 1));
					result.index_put_({ "...", 23 }, C4[7] * xz * (xx - 3 * yy));
					result.index_put_({ "...", 24 }, C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)));
				}		//(degree > 4)
			}		//(degree > 3)
		}		//(degree > 2)
	}		//(degree > 1)

	return std::make_pair(result, torch::Tensor());
}



///xyz : 3D coordinates of samples. B x 3
///bounding_box : min and max x, y, z coordinates of object bbox
///resolution : number of voxels per axis
HashEmbedderImpl::VoxelVertices HashEmbedderImpl :: GetVoxelVertices(torch::Tensor xyz, torch::Tensor bounding_box, torch::Tensor/*int?*/ resolution, const int log2_hashmap_size)
{
	auto device = xyz.device();
	VoxelVertices result;
	std::vector<torch::Tensor> splits = torch::split(bounding_box, { 3, 3 }, -1);
	auto box_min = splits[0];
	auto box_max = splits[1];
	result.KeepMask = xyz == torch::max(torch::min(xyz, box_max), box_min);
	//if (!torch::all(xyz <= box_max) || !torch::all(xyz >= box_min))
	xyz = torch::clamp(xyz, box_min, box_max);
	auto grid_size = (box_max - box_min) / resolution.to(device);

	auto bottom_left_idx = torch::floor((xyz - box_min) / grid_size).to(torch::kLong/*torch::kInt32*/);
	result.VoxelMinVertex = bottom_left_idx * grid_size + box_min;
	result.VoxelMaxVertex = result.VoxelMinVertex + torch::tensor({ 1.0f, 1.0f, 1.0f }, torch::kFloat32).to(device) * grid_size;
	auto voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS.to(device);
	result.HashedVoxelIndices = Hash(voxel_indices, log2_hashmap_size);
	return result;
}

///coords: this function can process upto 7 dim coordinates
///log2_hashmap_size : log2T logarithm of T w.r.t 2
torch::Tensor HashEmbedderImpl :: Hash(torch::Tensor coords, const long log2_hashmap_size)
{
	const std::array<long long, 7> primes = { 1ll, 2654435761ll, 805459861ll, 3674653429ll, 2097192037ll, 1434869437ll, 2165219737ll };
	torch::Tensor xor_result = torch::zeros_like(coords).index({ "...", 0 }).to(coords.device());
	for (int64_t i = 0; i < coords.sizes().back(); i++)
		xor_result ^= coords.index({ "...", i }) * primes[i];
	return torch::tensor({(1ll << static_cast<long long>(log2_hashmap_size)) - 1ll}, torch::kLong).to(xor_result.device()) & xor_result;
}

HashEmbedderImpl :: HashEmbedderImpl(
	const std::string &module_name,
	//std::array<std::array<float, 3>, 2> bounding_box,
	torch::Tensor bounding_box,
	const int n_levels/* = 16*/,
	const int n_features_per_level /*= 2*/,
	const int log2_hashmap_size /*= 19*/,
	const int base_resolution /*= 16*/,
	const int finest_resolution /*= 512*/
) : BaseEmbedderImpl(module_name), BoundingBox(bounding_box), NLevels(n_levels), NFeaturesPerLevel(n_features_per_level), Log2HashmapSize(log2_hashmap_size),
BaseResolution(base_resolution), FinestResolution(finest_resolution), OutputDims(n_levels* n_features_per_level)
{
	b = exp((log(finest_resolution) - log(base_resolution)) / (n_levels - 1));

	//Embedding is a simple lookup table that stores embeddings of a fixed dictionaryand size.
	//This module is often used to store word embeddingsand retrieve them using indices.The input to the module is a list of indices, and the output is the corresponding word embeddings.
	for (int i = 0; i < NLevels; i++)
		Embeddings->push_back(torch::nn::Embedding(pow(2ll, Log2HashmapSize), NFeaturesPerLevel));

	for (int i = 0; i < Embeddings->size(); i++)
		register_module(module_name + "_embeddings_" + std::to_string(i), Embeddings[i]);

	Initialize();
}

///custom uniform initialization
void HashEmbedderImpl :: Initialize()
{
	for (int i = 0; i < Embeddings->size(); i++)
		for (auto p : Embeddings[i]->parameters())
		{
			p = torch::nn::init::uniform_(p, -0.0001, 0.0001);
		}
}

///voxel_min_vertex: B x 3
///voxel_max_vertex : B x 3
///voxel_embedds : B x 8 x 2
torch::Tensor HashEmbedderImpl :: TrilinearInterp(torch::Tensor x, torch::Tensor voxel_min_vertex, torch::Tensor voxel_max_vertex, torch::Tensor voxel_embedds)
{
	auto weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex); // B x 3

	auto c00 = voxel_embedds.index({ torch::indexing::Slice(), 0 }) * (1.f - weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice() , torch::indexing::None }))
		+ voxel_embedds.index({ torch::indexing::Slice(), 4 }) * weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None });
	auto c01 = voxel_embedds.index({ torch::indexing::Slice(), 1 }) * (1.f - weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None }))
		+ voxel_embedds.index({ torch::indexing::Slice(), 5 }) * weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None });
	auto c10 = voxel_embedds.index({ torch::indexing::Slice(), 2 }) * (1.f - weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None }))
		+ voxel_embedds.index({ torch::indexing::Slice(), 6 }) * weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None });
	auto c11 = voxel_embedds.index({ torch::indexing::Slice(), 3 }) * (1.f - weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None }))
		+ voxel_embedds.index({ torch::indexing::Slice(), 7 }) * weights.index({ torch::indexing::Slice(), 0 }).index({ torch::indexing::Slice(), torch::indexing::None });

	auto c0 = c00 * (1.f - weights.index({ torch::indexing::Slice(), 1 }).index({ torch::indexing::Slice(), torch::indexing::None }))
		+ c10 * weights.index({ torch::indexing::Slice(), 1 }).index({ torch::indexing::Slice(), torch::indexing::None });
	auto c1 = c01 * (1.f - weights.index({ torch::indexing::Slice(), 1 }).index({ torch::indexing::Slice(), torch::indexing::None }))
		+ c11 * weights.index({ torch::indexing::Slice(), 1 }).index({ torch::indexing::Slice(), torch::indexing::None });

	auto c = c0 * (1.f - weights.index({ torch::indexing::Slice(), 2 }).index({ torch::indexing::Slice(), torch::indexing::None }))
		+ c1 * weights.index({ torch::indexing::Slice(), 2 }).index({ torch::indexing::Slice(), torch::indexing::None });

	return c;
}

///
std::pair<torch::Tensor, torch::Tensor> HashEmbedderImpl :: forward(torch::Tensor x)
{
	//x is 3D point position : B x 3
	std::vector <torch::Tensor> x_embedded_all;
	VoxelVertices voxel_vertices;
	for (int i = 0; i < NLevels; i++)
	{
		torch::Tensor resolution = torch::floor(torch::tensor(BaseResolution * pow(b, i))).to(x.device());
		voxel_vertices = GetVoxelVertices(x, BoundingBox, resolution, Log2HashmapSize);
		torch::Tensor voxel_embedds = Embeddings[i]->as<torch::nn::Embedding>()->forward(voxel_vertices.HashedVoxelIndices);
		torch::Tensor x_embedded = TrilinearInterp(x, voxel_vertices.VoxelMinVertex, voxel_vertices.VoxelMaxVertex, voxel_embedds);
		x_embedded_all.push_back(x_embedded);
	}

	auto keep_mask = voxel_vertices.KeepMask.sum(-1) == voxel_vertices.KeepMask.sizes().back();
	return std::make_pair(torch::cat(x_embedded_all, -1), keep_mask);
}



NeRFSmallImpl :: NeRFSmallImpl(
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
	const std::string module_name /*= "hashnerf"*/
) : BaseNeRFImpl(module_name), InputCh(input_ch), InputChViews(input_ch_views), NumLayers(num_layers), 
	HiddenDim(hidden_dim), GeoFeatDim(geo_feat_dim), NumLayersColor(num_layers_color), HiddenDimColor(hidden_dim_color),
	UsePredNormal(use_pred_normal), NumLayersNormals(num_layers_normals), HiddenDimNormals(hidden_dim_normals)
{
	for (int l = 0; l < NumLayers; l++)
		SigmaNet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputCh : HiddenDim, (l == NumLayers - 1) ? (1 + GeoFeatDim) : HiddenDim/*, false*/).bias(false)));	// 1 sigma + 15 SH features for color

	for (int l = 0; l < NumLayersColor; l++)
		ColorNet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputChViews + GeoFeatDim : HiddenDimColor, (l == NumLayersColor - 1) ? 3 : HiddenDimColor/*, false*/).bias(false)));

	if (UsePredNormal)
	{
		for (int l = 0; l < NumLayersNormals; l++)
			NormalsNet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? 1 + GeoFeatDim + InputCh : HiddenDimNormals, (l == NumLayersNormals - 1) ? 3 : HiddenDimNormals/*, false*/).bias(false)));
	}

	for (int i = 0; i < SigmaNet->size(); i++)
		register_module(module_name + "_sigma_net_" + std::to_string(i), SigmaNet[i]);

	for (int i = 0; i < ColorNet->size(); i++)
		register_module(module_name + "_color_net_" + std::to_string(i), ColorNet[i]);

	if (UsePredNormal)
	{
		for (int i = 0; i < NormalsNet->size(); i++)
			register_module(module_name + "_normals_net_" + std::to_string(i), NormalsNet[i]);
	}
}

torch::Tensor NeRFSmallImpl :: forward(torch::Tensor x)
{
	std::vector<torch::Tensor> splits = torch::split(x, { InputCh, InputChViews}, -1);
	auto input_pts = splits[0];
	auto input_views = splits[1];
	torch::Tensor sigma,
		geo_feat;

	//sigma
	auto h = input_pts;
	for (int i = 0; i < SigmaNet->size(); i++)
	{
		h = SigmaNet[i]->as<torch::nn::Linear>()->forward(h);
		if (i != SigmaNet->size() - 1)		//Финальные активации применяются в RawToOutputs
			h = torch::relu(h);		//!!!inplace = true
	}
	sigma = h.index({ "...", 0 });
	geo_feat = h.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });

	//color
	h = torch::cat({ input_views, geo_feat }, -1);
	for (int i = 0; i < ColorNet->size(); i++)
	{
		h = ColorNet[i]->as<torch::nn::Linear>()->forward(h);
		if (i != ColorNet->size() - 1)	//Финальные активации применяются в RawToOutputs
			h = torch::relu(h);		//!!!inplace = true
	}
	auto color = h;

	//predicted normals
	torch::Tensor predicted_normals;
	if (UsePredNormal)
	{
		h = torch::cat({ sigma.unsqueeze(-1), geo_feat, input_pts }, -1);
		for (int i = 0; i < NormalsNet->size(); i++)
		{
			h = NormalsNet[i]->as<torch::nn::Linear>()->forward(h);
			if (i != NormalsNet->size() - 1)	//Финальные активации применяются в RawToOutputs
				h = torch::relu(h);		//!!!inplace = true
			//else
			//	h = torch::tanh(h);
		}
		predicted_normals = /*torch::nn::functional::normalize(*/h/*, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8))*/;
	}

	auto outputs = torch::cat({ color, sigma.unsqueeze(-1), predicted_normals/*, calculated_normals*/ }, -1);
	//calculated_normals будет добавлено позже в RunNetwork потому что там есть доступ к исходным точкам а не их эмбедингам

	return outputs;
}