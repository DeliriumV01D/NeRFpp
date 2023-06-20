#pragma once

#include "TorchHeader.h"
#include "Trainable.h"
#include "RayUtils.h"

#include <set>

///Hierarchical sampling
inline torch::Tensor SamplePDF(torch::Tensor bins, torch::Tensor weights, const int nsamples, const bool det /*= false*/)
{
	torch::Device device = weights.device();
	//Get probability density function (PDF)
	weights = weights + 1e-5;
	auto pdf = weights / torch::sum(weights, -1, true);
	auto cdf = torch::cumsum(pdf, -1);
	cdf = torch::cat({ torch::zeros_like(cdf.index({ "...", torch::indexing::Slice(torch::indexing::None, 1)})), cdf }, -1);		//[batch, len(bins)]
	torch::Tensor u;
	//Take uniform samples
	std::vector<int64_t> sz(cdf.sizes().begin(), cdf.sizes().end());
	sz.back() = nsamples;
	if (det)
	{
		u = torch::linspace(0.f, 1.f, nsamples, torch::kFloat);
		u = u.expand(sz);
	}
	else {
		u = torch::rand(sz);
	}

	//Invert cumulative distribution function (CDF)
	u = u.contiguous().to(device);
	auto inds = torch::searchsorted(cdf, u, false, true);
	auto below = torch::max(torch::zeros_like(inds - 1), inds - 1);
	auto above = torch::min((cdf.sizes().back() - 1) * torch::ones_like(inds), inds);
	auto inds_g = torch::stack({ below, above }, -1);  //[batch, N_samples, 2];

	std::vector< int64_t> matched_shape{ inds_g.sizes()[0], inds_g.sizes()[1], cdf.sizes().back() };
	auto cdf_g = torch::gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g);
	auto bins_g = torch::gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g);

	auto denom = (cdf_g.index({ "...", 1 }) - cdf_g.index({ "...", 0 }));
	denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);
	auto t = (u - cdf_g.index({ "...", 0 })) / denom;
	auto samples = bins_g.index({ "...", 0 }) + t * (bins_g.index({ "...", 1 }) - bins_g.index({ "...", 0 }));

	return samples;
}

///
class BaseEmbedderImpl : public torch::nn::Module {
protected:
public:
	BaseEmbedderImpl(const std::string &module_name) : torch::nn::Module(module_name) {}
	virtual ~BaseEmbedderImpl() {}

	virtual int GetOutputDims() { return 0; }//abstract;

	///embedding + mask(can be empty)
	virtual std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) { return std::make_pair(torch::Tensor(), torch::Tensor()); }//abstract;
};
TORCH_MODULE(BaseEmbedder);

///Positional encoding
class EmbedderImpl : public BaseEmbedderImpl {
protected:
	int NumFreqs;
	float MaxFreq;
	bool IncludeInput;
	int InputDims,
		OutputDims = 0;;
	bool LogSampling;
	std::vector<float> FreqBands;
public:
	EmbedderImpl(const std::string &module_name, int multires)
		: EmbedderImpl(module_name, multires, multires - 1){}
	EmbedderImpl(const std::string &module_name, int num_freqs, float max_freq_log2, bool include_input = true, int input_dims = 3, bool log_sampling = true);
	virtual ~EmbedderImpl() {}
	
	virtual int GetOutputDims() override { return OutputDims; }
	///embedding + mask(can be empty)
	virtual std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) override;
};
TORCH_MODULE(Embedder);

///
class BaseNeRFImpl : public Trainable {
protected:
public:
	BaseNeRFImpl(const std::string &module_name) : Trainable(module_name) {}
	virtual ~BaseNeRFImpl() {}

	virtual torch::Tensor forward(torch::Tensor x) { return torch::Tensor(); }//abstract;

	///x.sizes()[0] должно быть кратно размеру chunk
	virtual torch::Tensor Batchify(torch::Tensor x, const int chunk);
};
TORCH_MODULE(BaseNeRF);

struct NeRFImpl : public BaseNeRFImpl
{
	int D,
		W,
		InputCh,
		InputChViews,
		OutputCh;
	std::set<int> Skips;
	bool UseViewDirs;
	
	torch::nn::ModuleList PtsLinears,
		ViewsLinears;

	torch::nn::Linear FeatureLinear = nullptr,
		AlphaLinear = nullptr,
		RGBLinear = nullptr,
		OutputLinear = nullptr;

	NeRFImpl(
		const int d = 8,
		const int w = 256,
		const int input_ch = 3,
		const int input_ch_views = 3,
		const int output_ch = 4,
		const std::set<int> &skips = std::set<int>{ 4 },
		const bool use_viewdirs = false,
		const std::string module_name = "nerf"
	);
	
	virtual ~NeRFImpl() {}

	virtual torch::Tensor forward(torch::Tensor x) override;
};
TORCH_MODULE(NeRF);


///
class SHEncoderImpl : public BaseEmbedderImpl {
protected:
	torch::Tensor C0 = torch::tensor({ 0.28209479177387814f });
	torch::Tensor C1 = torch::tensor({ 0.4886025119029199f });
	torch::Tensor C2 = torch::tensor({
		1.0925484305920792f,
		-1.0925484305920792f,
		0.31539156525252005f,
		-1.0925484305920792f,
		0.5462742152960396f
		});
	torch::Tensor C3 = torch::tensor({
		-0.5900435899266435f,
		2.890611442640554f,
		-0.4570457994644658f,
		0.3731763325901154f,
		-0.4570457994644658f,
		1.445305721320277f,
		-0.5900435899266435f
		});
	torch::Tensor C4 = torch::tensor({
		2.5033429417967046f,
		-1.7701307697799304f,
		0.9461746957575601f,
		-0.6690465435572892f,
		0.10578554691520431f,
		-0.6690465435572892f,
		0.47308734787878004f,
		-1.7701307697799304f,
		0.6258357354491761f
		});

	int InputDim,
		Degree,
		OutputDims;
public:

	SHEncoderImpl(
		const std::string &module_name,
		const int input_dim = 3,
		const int degree = 4
	) : BaseEmbedderImpl(module_name), InputDim(input_dim), Degree(degree), OutputDims(pow(degree, 2))
	{
		//assert input_dim == 3
		//assert degree >= 1 && self.degree <= 5
	}
	virtual ~SHEncoderImpl() {}

	int GetOutputDims() override { return OutputDims; }

	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) override
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
};
TORCH_MODULE(SHEncoder);

///Hash encoding
class HashEmbedderImpl : public BaseEmbedderImpl {
protected:
	torch::Tensor BOX_OFFSETS = torch::tensor({ 
		{{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1}}}, torch::kLong /*, cuda*/);
	//std::array<std::array<float, 3>, 2> BoundingBox;
	torch::Tensor BoundingBox;
	int NLevels,
		NFeaturesPerLevel,
		Log2HashmapSize,
		BaseResolution,
		FinestResolution,
		OutputDims;
	float b;
	torch::nn::ModuleList Embeddings;

	struct VoxelVertices {
		torch::Tensor VoxelMinVertex,
			VoxelMaxVertex,
			HashedVoxelIndices,
			KeepMask;
	};

	///xyz : 3D coordinates of samples. B x 3
	///bounding_box : min and max x, y, z coordinates of object bbox
	///resolution : number of voxels per axis
	VoxelVertices GetVoxelVertices(torch::Tensor xyz, torch::Tensor bounding_box, torch::Tensor/*int?*/ resolution, const int log2_hashmap_size)
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
public:
	///coords: this function can process upto 7 dim coordinates
	///log2_hashmap_size : log2T logarithm of T w.r.t 2
	static torch::Tensor Hash(torch::Tensor coords, const long log2_hashmap_size)
	{
		const std::array<long long, 7> primes = { 1ll, 2654435761ll, 805459861ll, 3674653429ll, 2097192037ll, 1434869437ll, 2165219737ll };
		torch::Tensor xor_result = torch::zeros_like(coords).index({ "...", 0 }).to(coords.device());
		for (int64_t i = 0; i < coords.sizes().back(); i++)
			xor_result ^= coords.index({ "...", i }) * primes[i];
		return torch::tensor((1ll << static_cast<long long>(log2_hashmap_size)) - 1ll, torch::kLong).to(xor_result.device()) & xor_result;
	}

	torch::nn::ModuleList GetEmbeddings() const { return Embeddings; }
	int GetNLevels() const { return NLevels; }
	int	GetNFeaturesPerLevel() const { return NFeaturesPerLevel; }
	int GetLog2HashmapSize() const { return Log2HashmapSize; }
	int GetBaseResolution() const { return BaseResolution; }
	int GetFinestResolution() const { return FinestResolution; }


	HashEmbedderImpl(
		const std::string &module_name,
		//std::array<std::array<float, 3>, 2> bounding_box,
		torch::Tensor bounding_box,
		const int n_levels = 16,
		const int n_features_per_level = 2,
		const int log2_hashmap_size = 19,
		const int base_resolution = 16,
		const int finest_resolution = 512
	) : BaseEmbedderImpl(module_name), BoundingBox(bounding_box), NLevels(n_levels), NFeaturesPerLevel(n_features_per_level), Log2HashmapSize(log2_hashmap_size),
		BaseResolution(base_resolution), FinestResolution(finest_resolution), OutputDims(n_levels * n_features_per_level)
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
	virtual ~HashEmbedderImpl() {}

	///custom uniform initialization
	void Initialize()
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
	torch::Tensor TrilinearInterp(torch::Tensor x, torch::Tensor voxel_min_vertex, torch::Tensor voxel_max_vertex, torch::Tensor voxel_embedds)
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

	int GetOutputDims() override { return OutputDims; }
	
	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) override
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
};
TORCH_MODULE(HashEmbedder);


///Small NeRF for Hash embeddings
struct NeRFSmallImpl : public BaseNeRFImpl
{
	int InputCh,
		InputChViews,
		NumLayers,
		HiddenDim,
		GeoFeatDim,
		NumLayersColor,
		HiddenDimColor;

	torch::nn::ModuleList SigmaNet,
		ColorNet;

	NeRFSmallImpl(
		const int num_layers = 3,
		const int hidden_dim = 64,
		const int geo_feat_dim = 15,
		const int num_layers_color = 4,
		const int hidden_dim_color = 64,
		const int input_ch = 3,
		const int input_ch_views = 3,
		const std::string module_name = "hashnerf"
	) : BaseNeRFImpl(module_name), InputCh(input_ch), InputChViews(input_ch_views), NumLayers(num_layers), HiddenDim(hidden_dim), GeoFeatDim(geo_feat_dim), NumLayersColor(num_layers_color), HiddenDimColor(hidden_dim_color)
	{
		for (int l = 0; l < NumLayers; l++)
			SigmaNet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputCh : HiddenDim, (l == NumLayers - 1) ? (1 + GeoFeatDim) : HiddenDim/*, false*/).bias(false)));	// 1 sigma + 15 SH features for color

		for (int l = 0; l < NumLayersColor; l++)
			ColorNet->push_back(torch::nn::Linear(torch::nn::LinearOptions((l == 0) ? InputChViews + GeoFeatDim : HiddenDimColor, (l == NumLayersColor - 1) ? 3 : HiddenDimColor/*, false*/).bias(false)));

		for (int i = 0; i < SigmaNet->size(); i++)
			register_module(module_name + "_sigma_net_" + std::to_string(i), SigmaNet[i]);

		for (int i = 0; i < ColorNet->size(); i++)
			register_module(module_name + "_color_net_" + std::to_string(i), ColorNet[i]);

	}

	virtual ~NeRFSmallImpl() override {}

	virtual torch::Tensor forward(torch::Tensor x) override
	{
		std::vector<torch::Tensor> splits = torch::split(x, { InputCh, InputChViews }, -1);
		auto input_pts = splits[0];
		auto input_views = splits[1];
		torch::Tensor sigma,
			geo_feat;

		//sigma
		auto h = input_pts;
		for (int i = 0; i < SigmaNet->size(); i++)
		{
			h = SigmaNet[i]->as<torch::nn::Linear>()->forward(h);
			if (i != SigmaNet->size() - 1)
				h = torch::relu(h);		//!!!inplace = true
		}
		sigma = h.index({ "...", 0 });
		geo_feat = h.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });

		//color
		h = torch::cat({ input_views, geo_feat }, -1);
		for (int i = 0; i < ColorNet->size(); i++)
		{
			h = ColorNet[i]->as<torch::nn::Linear>()->forward(h);
			if (i != ColorNet->size() - 1)
				h = torch::relu(h);		//!!!inplace = true
		}
		auto color = h;
		auto outputs = torch::cat({ color, sigma.unsqueeze(-1) }, -1);
		return outputs;
	}

	/////x.sizes()[0] должно быть кратно размеру chunk
	//virtual torch::Tensor Batchify(torch::Tensor x, const int chunk);
};
TORCH_MODULE(NeRFSmall);


inline torch::Tensor TotalVariationLoss(
	HashEmbedder embeddings,
	const torch::Device &device,
	const int min_resolution,
	const int max_resolution,
	const int level,
	const int log2_hashmap_size,
	const int n_levels = 16
){
	//Get resolution
	double b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1));
	torch::Tensor resolution = torch::tensor(floor(pow(b, level) * min_resolution)).to(torch::kLong);//.to(device);

	//Cube size to apply TV loss
	int	min_cube_size = min_resolution - 1;
	int	max_cube_size = 50; //can be tuned
	if (min_cube_size > max_cube_size)
		throw std::runtime_error("TotalVariationLoss: Error: min cuboid size greater than max!");
	torch::Tensor cube_size = torch::floor(torch::clip(resolution / 10.f, min_cube_size, max_cube_size)).to(torch::kLong);// .to(device);

	//Sample cuboid
	torch::Tensor min_vertex = torch::randint(0, (resolution - cube_size).item<int64_t>(), {3}, torch::kLong);
	torch::Tensor idx = min_vertex + torch::stack({ torch::arange(cube_size.item<int64_t>() + 1), torch::arange(cube_size.item<int64_t>() + 1), torch::arange(cube_size.item<int64_t>() + 1) }, -1);			//[16, 3]
	torch::Tensor cube_indices = torch::stack(torch::meshgrid({ idx.index({ torch::indexing::Slice(), 0 }), idx.index({ torch::indexing::Slice(), 1}), idx.index({torch::indexing::Slice(), 2}) }), -1).view({ int64_t(pow(cube_size.item<int64_t>() + 1, 3)), 3 }).to(device);		//[16, 16, 16, 3] -> [4096 , 3]
	torch::Tensor hashed_indices = HashEmbedderImpl::Hash(cube_indices, log2_hashmap_size);
	torch::Tensor cube_embeddings = embeddings->GetEmbeddings()[level]->as<torch::nn::Embedding>()->forward(hashed_indices);
	cube_embeddings = cube_embeddings.view({ cube_size.item<int64_t>() + 1, cube_size.item<int64_t>() + 1, cube_size.item<int64_t>() + 1, cube_embeddings.sizes().back() });		//[4096 , 3] -> [16, 16, 16, 3]
	auto tv_x = torch::pow(
		cube_embeddings.index({ torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice() , torch::indexing::Slice() , torch::indexing::Slice() })
		- cube_embeddings.index({ torch::indexing::Slice(torch::indexing::None, -1), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice() })
	,2 ).sum();
	auto tv_y = torch::pow(
		cube_embeddings.index({ torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice(), torch::indexing::Slice() })
		- cube_embeddings.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1), torch::indexing::Slice(), torch::indexing::Slice() })
	,2 ).sum();
	auto tv_z = torch::pow(
		cube_embeddings.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice() })
		- cube_embeddings.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1), torch::indexing::Slice() })
	, 2).sum();

	return (tv_x + tv_y + tv_z) / cube_size;

	//Посчитать кубы для всех уровней сразу?
	//Получить эмбеддинги всех уровней сразу штатными средствами как в RunNetwork ?
	//torch::Tensor inputs_flat = torch::reshape(inputs, { -1, inputs.sizes().back()/*[-1]*/ });  //[1024, 256, 3] -> [262144, 3]
	//auto [embedded, keep_mask] = embed_fn->forward(inputs_flat);
}

inline torch::Tensor SigmaSparsityLoss(torch::Tensor sigmas)
{
	//Using Cauchy Sparsity loss on sigma values
	return torch::log(1.0 + 2 * torch::pow(sigmas, 2)).sum(-1);
}
