#pragma once

#include "NeRF.h"
#include "NeRFRenderer.h"
#include "LeRF.h"
#include "Sampler.h"
#include "CuHashEmbedder.h"

///
struct LeRFRendererOutputs {
	//NeRFRendererOutputs NeRFOutputs;		//или унаследоваться от NeRFRendererOutputs
	torch::Tensor LangEmbedding,				///[num_rays, num_samples, lang_embed_dim]
		RenderedLangEmbedding,						///[num_rays, lang_embed_dim]
		DispMapLE,		///[num_rays] .Disparity map.Inverse of depth map.
		AccMapLE,			///[num_rays] .Sum of weights along each ray.
		WeightsLE,		///[num_rays, num_samples] .Weights assigned to each sampled embedding.
		DepthMapLE;		///[num_rays] .Estimated distance to object.
};

struct LeRFRenderResult
{
	LeRFRendererOutputs Outputs1,		///Comes from fine model.
		Outputs0;					///Output for coarse model.
	torch::Tensor Raw;	///[num_rays, num_samples, lang_embed_dim] .Raw predictions from model.
};

////test
//int embed_dim = 4,
//	bs = 2,
//	num_samples = 3;
//torch::Tensor embeds = torch::rand({ bs, num_samples, embed_dim }),
//	weights = torch::rand({ bs, num_samples, 1 });
//std::cout << "embeds: " << embeds << std::endl;
//std::cout << "weights: " << weights << std::endl;
//auto output = weights * embeds;
//std::cout << "mm: " << output << std::endl;
//output = torch::sum(output, -2);
//std::cout << "sum: " << output << std::endl;
//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
//std::cout << "normalize: " << output << std::endl;
//return 0 ;
///Calculate CLIP embeddings along ray.
inline torch::Tensor RenderCLIPEmbedding(		///[bs, embed_dim]
	const torch::Tensor embeds,		///[bs, num_samples, embed_dim]
	const torch::Tensor weights,	///[bs, num_samples, 1]
	const bool normalize = true
) {
	auto output = torch::sum(weights * embeds, -2);
	//output = output / torch::linalg::norm(output, -1, keepdim=true);
	output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
	return output;
}

///
class LeRFRenderer {
protected:
	CuHashEmbedder LangEmbedFn = nullptr;
	LeRF Lerf = nullptr;
	LeRF LerfFine = nullptr;										///"fine" network with same spec as Lerf.

	///Prepares inputs and applies network lerf or lerf_fine.
	virtual torch::Tensor RunLENetwork(torch::Tensor inputs, LeRF lerf, CuHashEmbedder lang_embed_fn);

	///Transforms model's predictions to semantically meaningful values.
	virtual LeRFRendererOutputs RawToLEOutputs(
		torch::Tensor raw,			///raw : [num_rays, num_samples along ray, 4+3+(3)] .Prediction from model.
		torch::Tensor z_vals,		///z_vals : [num_rays, num_samples along ray] .Integration time.
		torch::Tensor rays_d,		///rays_d : [num_rays, 3] .Direction of each ray.
		const int lang_embed_dim = 768
	);

public:
	LeRFRenderer(
		CuHashEmbedder lang_embed_fn,
		LeRF lerf,
		LeRF lerf_fine
	) : LangEmbedFn (lang_embed_fn), Lerf(lerf), LerfFine(lerf_fine) {};
	virtual ~LeRFRenderer(){};

	///Volumetric rendering.
	virtual LeRFRenderResult RenderRays(
		torch::Tensor ray_batch,		///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
		const int n_samples,
		const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
		const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
		const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		const int n_importance = 0,							///Number of additional times to sample along each ray.
		const bool white_bkgr = false,					///If True, assume a white background.
		const float raw_noise_std = 0.,
		//const int lang_embed_dim = 768,
		const bool return_weights = true
	);

	///Render rays in smaller minibatches to save memory
	///rays_flat.sizes()[0] должно быть кратно размеру chunk
	virtual LeRFRenderResult BatchifyRays(
		torch::Tensor rays_flat,								///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
		const int n_samples,
		const int chunk = 1024 * 32,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
		const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
		const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
		const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		const int n_importance = 0,							///Number of additional times to sample along each ray.
		const bool white_bkgr = false,					///If True, assume a white background.
		const float raw_noise_std = 0.,
		const bool return_weights = true
	);

	///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
	virtual LeRFRenderResult Render(
		const int h,					///Height of image in pixels.
		const int w,					///Width of image in pixels.
		torch::Tensor k,			///Сamera calibration
		const NeRFRenderParams &render_params,
		std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		torch::Tensor c2w_staticcam = torch::Tensor()			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
	);
};			//LeRFRenderer