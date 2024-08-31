#pragma once

#include "TorchHeader.h"
#include "BaseEmbedder.h"


///Hash encoding
class CuHashEmbedderImpl : public BaseEmbedderImpl {
protected:
	float b;
public:
	torch::Tensor BoundingBox;
	bool RandBias{false};
	int NLevels,
		NFeaturesPerLevel,
		Log2HashmapSize,
		BaseResolution,
		FinestResolution,
		OutputDims,
		NVolumes{1};
	torch::Tensor Embeddings,
		Primes,
		Biases,
		FeatLocalSize,
		FeatLocalIdx,
		QueryPoints,
		QueryVolumeIdx;


	//torch::nn::ModuleList GetEmbeddings() const { return Embeddings; }
	int GetNLevels() const { return NLevels; }
	int	GetNFeaturesPerLevel() const { return NFeaturesPerLevel; }
	int GetLog2HashmapSize() const { return Log2HashmapSize; }
	int GetBaseResolution() const { return BaseResolution; }
	int GetFinestResolution() const { return FinestResolution; }
	torch::Tensor GetBoundingBox() const { return BoundingBox; }


	CuHashEmbedderImpl(
		const std::string &module_name,
		//std::array<std::array<float, 3>, 2> bounding_box,
		torch::Tensor bounding_box,
		const int n_levels = 16,
		const int n_features_per_level = 2,
		const int log2_hashmap_size = 19,
		const int base_resolution = 16,
		const int finest_resolution = 512
	);
	virtual ~CuHashEmbedderImpl() {}

	void Initialize();
	int GetOutputDims() override { return OutputDims; }
	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) override;
};
TORCH_MODULE(CuHashEmbedder);



class CuHashEmbedderInfo : public torch::CustomClassHolder {
public:
	CuHashEmbedderImpl * HashEmbedder = nullptr;
};

namespace torch::autograd {

class CuHashEmbedderFunction : public Function<CuHashEmbedderFunction> {
public:
	static variable_list forward(AutogradContext *ctx, torch::Tensor embeddings, IValue embedder_info);
	static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}

inline torch::Tensor TotalVariationLoss(CuHashEmbedder embedder)
{
	std::vector<torch::Tensor> splits = torch::split(embedder->BoundingBox, { 3, 3 }, -1);
	auto box_min = splits[0];
	auto box_max = splits[1];
	int n_samples = static_cast<int> (pow(embedder->GetFinestResolution()/100, 3));

	torch::Tensor samples = torch::rand({ n_samples, {3} }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * (box_max - box_min) + box_min;
	
	std::pair<torch::Tensor, torch::Tensor> v = embedder->forward(samples),
		vx = embedder->forward(samples + torch::tensor({((box_max - box_min)/embedder->GetFinestResolution())[0].item<float>(), 0.f, 0.f}).to(torch::kCUDA)),
		vy = embedder->forward(samples + torch::tensor({0.f, ((box_max - box_min)/embedder->GetFinestResolution())[1].item<float>(), 0.f}).to(torch::kCUDA)),
		vz = embedder->forward(samples + torch::tensor({0.f, 0.f, ((box_max - box_min)/embedder->GetFinestResolution())[2].item<float>()}).to(torch::kCUDA));

	auto tv_x = torch::pow(v.first - vx.first, 2).sum();
	auto tv_y = torch::pow(v.first - vy.first, 2).sum();
	auto tv_z = torch::pow(v.first - vz.first, 2).sum();

	std::cout<<"tv_x + tv_y + tv_z: "<<tv_x + tv_y + tv_z<<std::endl;
	return (tv_x + tv_y + tv_z);
}