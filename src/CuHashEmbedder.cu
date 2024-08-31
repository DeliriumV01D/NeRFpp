#include "CuHashEmbedder.h"
#include "vector_types.h"

using emb_fp = __half;			//float;			//__half;
using emb_fp2 = __half2;		//float2;		//__half2;
torch::Dtype t_emb_fp = torch::kFloat16;		//torch::kFloat32;		//torch::kFloat16;

template<typename T>
__global__ void CuHashEmbedderForwardKernel(
	int base_resolution,
	int finest_resolution,
	int n_levels,
	int n_channels,
	int n_points, 
	int n_volumes,
	T* feat_pool, 
	int* prim_pool, 
	int* feat_local_idx, 
	int* feat_local_size,
	float3* bias_pool,
	float3* points_ptr,
	float3* pbox_min,
	float3* pbox_max,
	int* volume_idx,
	T* out_feat
) {
	int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int level_idx = blockIdx.y;
	if (pts_idx >= n_points) 
		return;

	points_ptr  = points_ptr + pts_idx;
	volume_idx = volume_idx + pts_idx;
	out_feat = out_feat + pts_idx * (n_levels * n_channels);

	float3 pt = points_ptr[0],
		box_min = pbox_min[0],
		box_max = pbox_max[0];

	float mul = exp2f((log2f(finest_resolution) - log2f(base_resolution)) * float(level_idx) / float(n_levels - 1) + log2f(base_resolution));
	//!!!pt *= mul;
	//for (auto &it : pt)
	//	it *= mul;
	pt.x = (pt.x - box_min.x) / (box_max.x - box_min.x) * mul;
	pt.y = (pt.y - box_min.y) / (box_max.y - box_min.y) * mul;
	pt.z = (pt.z - box_min.z) / (box_max.z - box_min.z) * mul;
	unsigned prim_a, prim_b, prim_c, local_size;
	{
		const int offset = (level_idx * n_volumes + volume_idx[0]) * 3;
		prim_a = prim_pool[offset + 0];
		prim_b = prim_pool[offset + 1];
		prim_c = prim_pool[offset + 2];
	}
	feat_pool = feat_pool + feat_local_idx[level_idx];
	local_size = feat_local_size[level_idx];

	int transf_idx = level_idx * n_volumes + volume_idx[0];
	//!!!pt = pt + bias_pool[transf_idx];
	//for (int it = 0; it < pt.size(); it++)
	//	pt[it] += bias_pool[transf_idx][it];
	pt.x += bias_pool[transf_idx].x;
	pt.y += bias_pool[transf_idx].y;
	pt.z += bias_pool[transf_idx].z;


	auto pos_x = static_cast<unsigned>(floorf(pt.x));
	auto pos_y = static_cast<unsigned>(floorf(pt.y));
	auto pos_z = static_cast<unsigned>(floorf(pt.z));

	unsigned pos_000 = ((pos_x * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_001 = ((pos_x * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
	unsigned pos_010 = ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_011 = ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
	unsigned pos_100 = (((pos_x + 1u) * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_101 = (((pos_x + 1u) * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
	unsigned pos_110 = (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_111 = (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;

	float a = pt.x - floorf(pt.x);
	float b = pt.y - floorf(pt.y);
	float c = pt.z - floorf(pt.z);

	float w000 = (1.f - a) * (1.f - b) * (1.f - c);
	float w001 = (1.f - a) * (1.f - b) * c;
	float w010 = (1.f - a) * b * (1.f - c);
	float w011 = (1.f - a) * b * c;
	float w100 = a * (1.f - b) * (1.f - c);
	float w101 = a * (1.f - b) * c;
	float w110 = a * b * (1.f - c);
	float w111 = a * b * c;

	#pragma unroll
	for (int k = 0; k < n_channels; k++) 
	{
		out_feat[level_idx * n_channels + k] = (T) (
			w000 * float(feat_pool[pos_000 * n_channels + k]) + w001 * float(feat_pool[pos_001 * n_channels + k]) +
			w010 * float(feat_pool[pos_010 * n_channels + k]) + w011 * float(feat_pool[pos_011 * n_channels + k]) +
			w100 * float(feat_pool[pos_100 * n_channels + k]) + w101 * float(feat_pool[pos_101 * n_channels + k]) +
			w110 * float(feat_pool[pos_110 * n_channels + k]) + w111 * float(feat_pool[pos_111 * n_channels + k])
		);
	}
}


template<typename T>
__global__ void CuHashEmbedderBackwardKernel(
	int base_resolution,
	int finest_resolution,
	int n_levels,
	int n_channels,
	int n_points, int n_volumes,
	int* prim_pool, int* feat_local_idx, int* feat_local_size,
	float3* bias_pool,
	float3* points_ptr,
	float3* pbox_min,
	float3* pbox_max,
	int* volume_idx,
	T* grad_in, // [ n_points, n_levels, n_channels ]
	T* grad_out // [ pool_size, n_channels ]
) {
	int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int level_idx = blockIdx.y;
	if (pts_idx >= n_points)
		return;

	points_ptr  = points_ptr + pts_idx;
	volume_idx = volume_idx + pts_idx;
	grad_in = grad_in + (n_levels * n_channels) * pts_idx + level_idx * n_channels;
	float3 pt = points_ptr[0],
		box_min = pbox_min[0],
		box_max = pbox_max[0];
	
	float mul = exp2f((log2f(finest_resolution) - log2f(base_resolution)) * float(level_idx) / float(n_levels - 1) + log2f(base_resolution));
	//!!!pt *= mul;
	//for (auto &it : pt)
	//	it *= mul;
	pt.x = (pt.x - box_min.x) / (box_max.x - box_min.x) * mul;
	pt.y = (pt.y - box_min.y) / (box_max.y - box_min.y) * mul;
	pt.z = (pt.z - box_min.z) / (box_max.z - box_min.z) * mul;
	unsigned prim_a, prim_b, prim_c, local_size;
	{
		const int offset = (level_idx * n_volumes + volume_idx[0]) * 3;
		prim_a = prim_pool[offset + 0];
		prim_b = prim_pool[offset + 1];
		prim_c = prim_pool[offset + 2];
	}

	grad_out = grad_out + feat_local_idx[level_idx];
	local_size = feat_local_size[level_idx];

	int transf_idx = level_idx * n_volumes + volume_idx[0];
	//!!!pt = pt + bias_pool[transf_idx];
	//for (int it = 0; it < pt.size(); it++)
	//	pt[it] += bias_pool[transf_idx][it];
	pt.x += bias_pool[transf_idx].x;
	pt.y += bias_pool[transf_idx].y;
	pt.z += bias_pool[transf_idx].z;

	unsigned pos_x = static_cast<unsigned>(floorf(pt.x));
	unsigned pos_y = static_cast<unsigned>(floorf(pt.y));
	unsigned pos_z = static_cast<unsigned>(floorf(pt.z));

	unsigned pos_000 = ((pos_x * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_001 = ((pos_x * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
	unsigned pos_010 = ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_011 = ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
	unsigned pos_100 = (((pos_x + 1u) * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_101 = (((pos_x + 1u) * prim_a) ^ ((pos_y) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
	unsigned pos_110 = (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z) * prim_c)) % local_size;
	unsigned pos_111 = (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;

	float a = pt.x - floorf(pt.x);
	float b = pt.y - floorf(pt.y);
	float c = pt.z - floorf(pt.z);

	float w000 = (1.f - a) * (1.f - b) * (1.f - c);
	float w001 = (1.f - a) * (1.f - b) * c;
	float w010 = (1.f - a) * b * (1.f - c);
	float w011 = (1.f - a) * b * c;
	float w100 = a * (1.f - b) * (1.f - c);
	float w101 = a * (1.f - b) * c;
	float w110 = a * b * (1.f - c);
	float w111 = a * b * c;

	float ws[8] = { w000, w001, w010, w011, w100, w101, w110, w111 };
	unsigned pos[8] = { pos_000, pos_001, pos_010, pos_011, pos_100, pos_101, pos_110, pos_111 };

	#pragma unroll
	for (int d = 0; d < 8; d++) 
	{
		for (int k = 0; k < n_channels; k += 2) 
		{
			float w0 = (float) grad_in[k];
			float w1 = (float) grad_in[k + 1];
			if (w0 != 0.f || w1 != 0.f) 
			{
				emb_fp2 cur_w = {(T) (float(w0) * ws[d]), (T) (float(w1) * ws[d])};
				atomicAdd((emb_fp2 *) (grad_out + pos[d] * n_channels + k), cur_w);
			}
		}
	}
	//#pragma unroll
	//for (int d = 0; d < 8; d++) 
	//{
	//	for (int k = 0; k < n_channels; k += 1) 
	//	{
	//		float w0 = (float) grad_in[k];
	//		if (w0 != 0.f) 
	//		{
	//			T cur_w = {(T) (float(w0) * ws[d])};
	//			atomicAdd((T *) (grad_out + pos[d] * n_channels + k), cur_w);
	//		}
	//	}
	//}

}


namespace torch::autograd {

variable_list CuHashEmbedderFunction::forward(
	AutogradContext* ctx,
	torch::Tensor embeddings,
	IValue hash3d_info
) {
	auto info_ptr = hash3d_info.toCustomClass<CuHashEmbedderInfo>();
	ctx->saved_data["cu_hash_embedder_info"] = hash3d_info;
	torch::Tensor &points  = info_ptr->HashEmbedder->QueryPoints;                       // [ n_points, 3 ]
	torch::Tensor &volume_idx = info_ptr->HashEmbedder->QueryVolumeIdx;                // [ n_points, 1 ]
	torch::Tensor &prim_pool = info_ptr->HashEmbedder->Primes;
	torch::Tensor &bias_pool = info_ptr->HashEmbedder->Biases;
	torch::Tensor &feat_local_idx = info_ptr->HashEmbedder->FeatLocalIdx;
	torch::Tensor &feat_local_size = info_ptr->HashEmbedder->FeatLocalSize;
	if (!points.device().is_cuda())
		points = points.to(torch::kCUDA);
	if (!volume_idx.device().is_cuda())
		volume_idx = volume_idx.to(torch::kCUDA);
	CHECK(points.device().is_cuda());
	CHECK(volume_idx.device().is_cuda());

	std::vector<torch::Tensor> splits = torch::split(info_ptr->HashEmbedder->BoundingBox, { 3, 3 }, -1);
	auto box_min = splits[0].to(torch::kCUDA);
	auto box_max = splits[1].to(torch::kCUDA);

	int n_points = static_cast<int>(points.sizes()[0]);

	int n_volumes = info_ptr->HashEmbedder->NVolumes;

	const unsigned thread_cap = 512;
	dim3 block_dim = { unsigned(thread_cap), 1, 1 };
	dim3 grid_dim  = { (n_points + thread_cap - 1)/thread_cap, unsigned(info_ptr->HashEmbedder->NLevels), 1 };

	torch::Tensor out_feat = torch::zeros({ n_points, info_ptr->HashEmbedder->NLevels * info_ptr->HashEmbedder->NFeaturesPerLevel }, torch::TensorOptions().dtype(t_emb_fp).device(torch::kCUDA));
	CHECK(out_feat.is_contiguous());

	//Тут можно собрать в один тензор ведь все равно делается преобразование в float16
	torch::Tensor feat_pool_true = (embeddings.type().scalarType() != t_emb_fp) ? embeddings.to(t_emb_fp).contiguous() : embeddings;
	CHECK(feat_pool_true.is_contiguous());

	CuHashEmbedderForwardKernel<emb_fp><<<grid_dim, block_dim>>>(
		info_ptr->HashEmbedder->BaseResolution, info_ptr->HashEmbedder->FinestResolution,
		info_ptr->HashEmbedder->NLevels, info_ptr->HashEmbedder->NFeaturesPerLevel,
		n_points, n_volumes,
		reinterpret_cast<emb_fp*>(feat_pool_true.data_ptr()),
		prim_pool.data_ptr<int>(), feat_local_idx.data_ptr<int>(), feat_local_size.data_ptr<int>(),
		reinterpret_cast<float3*>(bias_pool.data_ptr()),
		reinterpret_cast<float3*>(points.data_ptr()),
		reinterpret_cast<float3*>(box_min.data_ptr()),
		reinterpret_cast<float3*>(box_max.data_ptr()),
		volume_idx.data_ptr<int>(),
		reinterpret_cast<emb_fp*>(out_feat.data_ptr())
	);

	return { out_feat.to(torch::kFloat32) };
}

variable_list CuHashEmbedderFunction::backward(AutogradContext* ctx, variable_list grad_output) 
{
	auto info_ptr = ctx->saved_data["cu_hash_embedder_info"].toCustomClass<CuHashEmbedderInfo>();
	Tensor& points  = info_ptr->HashEmbedder->QueryPoints;							// [ n_points, 3 ]
	Tensor& volume_idx = info_ptr->HashEmbedder->QueryVolumeIdx;
	Tensor& prim_pool = info_ptr->HashEmbedder->Primes;
	Tensor& bias_pool = info_ptr->HashEmbedder->Biases;
	Tensor& feat_local_idx = info_ptr->HashEmbedder->FeatLocalIdx;
	Tensor& feat_local_size = info_ptr->HashEmbedder->FeatLocalSize;
	CHECK(points.device().is_cuda());
	CHECK(volume_idx.device().is_cuda());

	std::vector<torch::Tensor> splits = torch::split(info_ptr->HashEmbedder->BoundingBox, { 3, 3 }, -1);
	auto box_min = splits[0].to(torch::kCUDA);
	auto box_max = splits[1].to(torch::kCUDA);

	const float grad_scale = 128.f;
	int n_points = static_cast<int>(grad_output[0].sizes()[0]);//points.sizes()[0];

	int pool_size = (1ll << static_cast<long long>(info_ptr->HashEmbedder->Log2HashmapSize)) * info_ptr->HashEmbedder->NLevels;	//pow(2ll, info_ptr->HashEmbedder->Log2HashmapSize) * NLevels;
	int n_volumes = info_ptr->HashEmbedder->NVolumes;

	const unsigned thread_cap = 512;
	dim3 block_dim = { unsigned(thread_cap), 1, 1 };
	dim3 grid_dim  = { (n_points + thread_cap - 1)/thread_cap, unsigned(info_ptr->HashEmbedder->NLevels), 1 };

	torch::Tensor grad_in = (grad_output[0] * grad_scale).to(t_emb_fp).contiguous();
	//torch::Tensor grad_in = grad_output[0].type().scalarType() != t_emb_fp ? (grad_output[0] * grad_scale).to(t_emb_fp).contiguous() : grad_output[0].contiguous();

	torch::Tensor true_grad_out = torch::zeros({ pool_size, info_ptr->HashEmbedder->NFeaturesPerLevel }, torch::TensorOptions().dtype(t_emb_fp).device(torch::kCUDA)).contiguous();

	CuHashEmbedderBackwardKernel<emb_fp><<<grid_dim, block_dim>>>(
		info_ptr->HashEmbedder->BaseResolution, info_ptr->HashEmbedder->FinestResolution,
		info_ptr->HashEmbedder->NLevels, info_ptr->HashEmbedder->NFeaturesPerLevel,
		n_points, n_volumes,
		prim_pool.data_ptr<int>(),
		feat_local_idx.data_ptr<int>(),
		feat_local_size.data_ptr<int>(),
		reinterpret_cast<float3*>(bias_pool.data_ptr()),
		reinterpret_cast<float3*>(points.data_ptr()),
		reinterpret_cast<float3*>(box_min.data_ptr()),
		reinterpret_cast<float3*>(box_max.data_ptr()),
		volume_idx.data_ptr<int>(),
		reinterpret_cast<emb_fp*>(grad_in.data_ptr()),
		reinterpret_cast<emb_fp*>(true_grad_out.data_ptr())
	);
	return {true_grad_out.to(torch::kFloat32) / grad_scale, torch::Tensor() };
	//return {true_grad_out.type().scalarType() != torch::kFloat32 ? true_grad_out.to(torch::kFloat32) / grad_scale : true_grad_out, torch::Tensor() };
}

}