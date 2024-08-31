#include "CuHashEmbedder.h"

TORCH_LIBRARY(cu_hash_embedder, m)
{
	std::cout << "register CuHashEmbedderInfo" << std::endl;
	m.class_<CuHashEmbedderInfo>("CuHashEmbedderInfo").def(torch::init());
}


CuHashEmbedderImpl :: CuHashEmbedderImpl(
	const std::string &module_name,
	//std::array<std::array<float, 3>, 2> bounding_box,
	torch::Tensor bounding_box,
	const int n_levels/* = 16*/,
	const int n_features_per_level /*= 2*/,
	const int log2_hashmap_size /*= 19*/,
	const int base_resolution /*= 16*/,
	const int finest_resolution /*= 512*/
) : BaseEmbedderImpl(module_name), BoundingBox(bounding_box), NLevels(n_levels), NFeaturesPerLevel(n_features_per_level), Log2HashmapSize(log2_hashmap_size),
BaseResolution(base_resolution), FinestResolution(finest_resolution), OutputDims(n_levels* n_features_per_level), RandBias(false)
{
	//b = exp((log(finest_resolution) - log(base_resolution)) / (n_levels - 1));

	Embeddings = register_parameter(module_name+"_embeddings", (torch::rand({(1ll << static_cast<long long>(log2_hashmap_size)) * NLevels, NFeaturesPerLevel}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) /** .5f - 1.f*/) * 1e-4f, /*requires_grad = */true);
	CHECK(Embeddings.is_contiguous());

	// Get prime numbers
	auto is_prim = [](int x) 
	{
		for (int i = 2; i * i <= x; i++) 
		{
			if (x % i == 0) return false;
		}
		return true;
	};

	std::vector<int> prim_selected;
	int min_local_prim = 1 << 28;
	int max_local_prim = 1 << 30;

	for (int i = 0; i < 3 * NLevels * NVolumes; i++) 
	{
		int val;
		do {
			val = torch::randint(min_local_prim, max_local_prim, {1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).item<int>();
		} while (!is_prim(val));
		prim_selected.push_back(val);
	}
	CHECK(prim_selected.size() == 3 * NLevels * NVolumes);

	Primes = torch::from_blob(prim_selected.data(), 3 * NLevels * NVolumes, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).to(torch::kCUDA);
	Primes = Primes.reshape({NLevels, NVolumes, 3}).contiguous();

	if (RandBias) 
	{
		Biases = (torch::rand({ NLevels * NVolumes, 3 }, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)) * 1000.f + 100.f).contiguous();
	}	else {
		Biases = torch::zeros({ NLevels * NVolumes, 3 }, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)).contiguous();
	}

	// Size of each level & each volume.
	{
		int local_size = 1ll << static_cast<long long>(Log2HashmapSize);	//pow(2ll, Log2HashmapSize);
		local_size = (local_size >> 4) << 4;
		FeatLocalSize = torch::full({ NLevels }, local_size, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)).contiguous();
		FeatLocalIdx = torch::cumsum(FeatLocalSize, 0) - local_size;
		FeatLocalIdx = FeatLocalIdx.to(torch::kInt32).contiguous();
	}


	////register_buffer(module_name+"_embeddings", Embeddings);
	////Embeddings = register_parameter(module_name+"_embeddings", Embeddings, /*requires_grad = */true);
	Primes = register_buffer(module_name+"_primes", Primes);
	Biases = register_buffer(module_name+"_biases", Biases);
	FeatLocalSize = register_buffer(module_name+"_feat_local_size", FeatLocalSize);
	FeatLocalIdx = register_buffer(module_name+"_feat_local_idx", FeatLocalIdx);
	//QueryPoints = register_buffer(module_name+"_query_points", QueryPoints);
	//QueryVolumeIdx = register_buffer(module_name+"_query_volume_idx", QueryVolumeIdx);
}

void CuHashEmbedderImpl :: Initialize()
{}

///
std::pair<torch::Tensor, torch::Tensor> CuHashEmbedderImpl :: forward(torch::Tensor x)
{
	auto info = torch::make_intrusive<CuHashEmbedderInfo>();

	std::vector<torch::Tensor> splits = torch::split(BoundingBox, { 3, 3 }, -1);
	auto box_min = splits[0];
	auto box_max = splits[1];
	torch::Tensor keep_mask = x == torch::max(torch::min(x, box_max), box_min);
	//if (!torch::all(xyz <= box_max) || !torch::all(xyz >= box_min))
	x = torch::clamp(x, box_min, box_max);

	QueryPoints = x.contiguous();//((x + 1.f) * .5f).contiguous();   // [-1, 1] -> [0, 1]		[ n_points, 3 ]
	QueryVolumeIdx = torch::zeros({QueryPoints.sizes()[0], 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)).contiguous();// = anchors.contiguous();		//[ n_points, 1 ]
	info->HashEmbedder = this;
	auto x_embedded_all = torch::autograd::CuHashEmbedderFunction::apply(Embeddings, torch::IValue(info))[0];  // [n_points, n_levels * n_channels];

	torch::Tensor mask = keep_mask.sum(-1) == keep_mask.sizes().back();
	return std::make_pair(x_embedded_all/*torch::cat(x_embedded_all, -1)*/, mask);
}