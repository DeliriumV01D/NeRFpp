#pragma once

#include "TorchHeader.h"
#include "Trainable.h"
#include "RayUtils.h"
#include "BaseEmbedder.h"

#include <set>


///Positional encoding
class EmbedderImpl : public BaseEmbedderImpl {
protected:
	int NumFreqs;
	float MaxFreq;
	bool IncludeInput;
	int InputDims,
		OutputDims = 0;
	bool LogSampling;
	std::vector<float> FreqBands;
public:
	EmbedderImpl(const std::string &module_name, int multires)
		: EmbedderImpl(module_name, multires, float(multires - 1)){}
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
	//virtual torch::Tensor Batchify(torch::Tensor x, const int chunk);
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
	) : BaseEmbedderImpl(module_name), InputDim(input_dim), Degree(degree), OutputDims((int)pow(degree, 2))
	{
		//assert input_dim == 3
		//assert degree >= 1 && self.degree <= 5
	}
	virtual ~SHEncoderImpl() {}

	int GetOutputDims() override { return OutputDims; }

	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) override;
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
	VoxelVertices GetVoxelVertices(torch::Tensor xyz, torch::Tensor bounding_box, torch::Tensor/*int?*/ resolution, const int log2_hashmap_size);
public:
	///coords: this function can process upto 7 dim coordinates
	///log2_hashmap_size : log2T logarithm of T w.r.t 2
	static torch::Tensor Hash(torch::Tensor coords, const long log2_hashmap_size);

	torch::nn::ModuleList GetEmbeddings() const { return Embeddings; }
	int GetNLevels() const { return NLevels; }
	int	GetNFeaturesPerLevel() const { return NFeaturesPerLevel; }
	int GetLog2HashmapSize() const { return Log2HashmapSize; }
	int GetBaseResolution() const { return BaseResolution; }
	int GetFinestResolution() const { return FinestResolution; }
	torch::Tensor GetBoundingBox() const { return BoundingBox; }


	HashEmbedderImpl(
		const std::string &module_name,
		//std::array<std::array<float, 3>, 2> bounding_box,
		torch::Tensor bounding_box,
		const int n_levels = 16,
		const int n_features_per_level = 2,
		const int log2_hashmap_size = 19,
		const int base_resolution = 16,
		const int finest_resolution = 512
	);
	virtual ~HashEmbedderImpl() {}

	///custom uniform initialization
	void Initialize();

	///voxel_min_vertex: B x 3
	///voxel_max_vertex : B x 3
	///voxel_embedds : B x 8 x 2
	torch::Tensor TrilinearInterp(torch::Tensor x, torch::Tensor voxel_min_vertex, torch::Tensor voxel_max_vertex, torch::Tensor voxel_embedds);

	int GetOutputDims() override { return OutputDims; }
	
	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) override;
};
TORCH_MODULE(HashEmbedder);


///Small NeRF for Hash embeddings
class NeRFSmallImpl : public BaseNeRFImpl {
protected:
	int InputCh,
		InputChViews,
		NumLayers,
		HiddenDim,
		GeoFeatDim,
		NumLayersColor,
		HiddenDimColor,
		NumLayersNormals,
		HiddenDimNormals;

	bool UsePredNormal;	//whether to use predicted normals

	torch::nn::ModuleList SigmaNet,
		ColorNet,
		NormalsNet;
public:
	NeRFSmallImpl(
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
		const std::string module_name = "hashnerf"
	);

	virtual ~NeRFSmallImpl() override {}

	virtual torch::Tensor forward(torch::Tensor x) override;

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
	const int n_levels
){
	//Get resolution
	double b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1));
	torch::Tensor resolution = torch::tensor(floor(pow(b, level) * min_resolution)).to(torch::kLong);//.to(device);

	//Cube size to apply TV loss
	int	min_cube_size = min_resolution - 1;
	int	max_cube_size = max_resolution - 1; //can be tuned
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
	//torch::Tensor inputs_flat = torch::reshape(inputs, { -1, inputs.sizes().back()/*[-1]*/ });  //[1024, 256, 3] -> [262144, 3]
	//auto [embedded, keep_mask] = embed_fn->forward(inputs_flat);
}

///Using Cauchy Sparsity loss on sigma values
inline torch::Tensor SigmaSparsityLoss(torch::Tensor sigmas)
{
	return torch::log(1.0 + 2 * torch::pow(sigmas, 2)).sum(-1);
}

///Loss that encourages that all visible normals are facing towards the camera.
inline torch::Tensor OrientationLoss(
	torch::Tensor weights,		//[bs, num_samples, 1]
	torch::Tensor normals,		//[bs, num_samples, 3]
	torch::Tensor viewdirs		//[bs, 3]
) {
	auto n_dot_minus_v = (normals * (viewdirs * -1).index({ "...", torch::indexing::None, torch::indexing::Slice() })).sum(-1);
	return (weights.index({ "...", 0 }) * torch::pow(torch::fmin(torch::zeros_like(n_dot_minus_v), n_dot_minus_v), 2)).sum(-1);
}

///Loss between normals calculated from density and normals from prediction network.
inline torch::Tensor PredNormalLoss(
	torch::Tensor weights,		//[bs, num_samples, 1]
	torch::Tensor normals,		//[bs, num_samples, 3]
	torch::Tensor pred_normals		//[bs, num_samples, 3]
) {
	return torch::mse_loss(weights * pred_normals, weights * normals);
	//return (weights.index({ "...", 0 }) * (1.0 - (normals * pred_normals).sum(-1))).sum(-1);
}