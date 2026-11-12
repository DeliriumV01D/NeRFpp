#pragma once

#include "NeRF.h"
#include "NeRFRenderer.h"
#include "LeRF.h"
#include "Sampler.h"
#include "CuHashEmbedder.h"
#include <functional>
#include <vector>

///
struct LeRFRendererOutputs {
	//NeRFRendererOutputs NeRFOutputs;		//или унаследоваться от NeRFRendererOutputs
	torch::Tensor LangEmbedding,				///[num_rays, num_samples, lang_embed_dim]
		RenderedLangEmbedding,						///[num_rays, lang_embed_dim]
		DispMapLE,		///[num_rays] .Disparity map.Inverse of depth map.
		AccMapLE,			///[num_rays] .Sum of weights along each ray.
		WeightsLE,		///[num_rays, num_samples] .Weights assigned to each sampled embedding.
		DepthMapLE,		///[num_rays] .Estimated distance to object.
		RelevancyMap;		///[num_rays, 1] .Positive probability vs the hardest negative prompt. With per-ray pyramid scales: at the pass scale. Inference without scales: max over the pyramid scale sweep.
};

struct LeRFRenderResult
{
	LeRFRendererOutputs Outputs;		///Comes from model
	torch::Tensor Raw;	///[num_rays, num_samples, lang_embed_dim] .Raw predictions from model.
	float Near,
		Far;
	bool Aborted{false};		///True if rendering was cancelled between batches; maps hold a partial result
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


///Composites the LeRF relevancy heatmap over the RGB render with the original lerf threshold semantics (p = clamp(rel - 0.5, 0, 1)) as a translucent alpha blend with a capped opacity
inline cv::Mat ComposeRelevancyOverRGB(const torch::Tensor &relevancy, const torch::Tensor &rgb)
{
	auto rel = relevancy.detach().squeeze().cpu().to(torch::kFloat32).contiguous();	//[H, W]
	const int64_t H = rel.size(0),
		W = rel.size(1);

	const bool has_rgb = rgb.defined() && rgb.numel() != 0 && rgb.dim() == 3 && rgb.size(0) == H && rgb.size(1) == W && rgb.size(2) == 3;

	//No usable RGB underlay: keep the full-range relevancy colormap behavior
	if (!has_rgb)
	{
		auto q = rel.clamp(0, 1).mul(255).clamp(0, 255).to(torch::kU8).contiguous();
		cv::Mat gray(static_cast<int>(H), static_cast<int>(W), CV_8UC1, q.data_ptr()), out;
		cv::applyColorMap(gray, out, cv::COLORMAP_JET);
		return out;
	}

	auto p = torch::clamp(rel - 0.5, 0.0, 1.0);			//Original lerf threshold
	const float p_max = p.max().item<float>();

	//Base layer: the RGB render via the standard conversion
	cv::Mat base = TorchTensorToCVMat(rgb);

	//Nothing above the threshold: the pure RGB render
	if (p_max <= 1e-6f)
		return base;

	//Above the threshold: colormap of p / max(p) alpha-blended over the RGB base
	auto alpha2d = p.div(p_max).clamp(0.f, 1.f);	//[H, W] in [0, 1]
	auto gray8 = alpha2d.mul(255).to(torch::kU8).contiguous();
	cv::Mat gray(static_cast<int>(H), static_cast<int>(W), CV_8UC1, gray8.data_ptr()), jet;
	cv::applyColorMap(gray, jet, cv::COLORMAP_JET);

	//Opacity cap: even at peak relevancy the colormap stays at most alpha_max opaque, so the RGB underlay always shows through noticeably
	const float alpha_max = 0.65f;			//(0.65 = 65% JET / 35% RGB)

	//Translucent blend in float space: out = base + alpha * (jet - base), alpha capped at alpha_max
	auto alpha = alpha2d.mul(alpha_max).unsqueeze(-1).expand({ H, W, 3 }).contiguous();
	auto jet_t = torch::from_blob(jet.data, { H, W, 3 }, torch::kU8).to(torch::kFloat32).div(255.);
	auto base_t = torch::from_blob(base.data, { H, W, 3 }, torch::kU8).to(torch::kFloat32).div(255.);
	auto blended = base_t.add(alpha.mul(jet_t.sub(base_t))).mul(255).clamp(0, 255).to(torch::kU8).contiguous();
	return cv::Mat(static_cast<int>(H), static_cast<int>(W), CV_8UC3, blended.data_ptr()).clone();
}


///
class LeRFRenderer {
protected:
	CuHashEmbedder LangEmbedFn = nullptr;
	LeRF Lerf = nullptr;
	torch::Tensor LerfPositives, 
		LerfNegatives;

	///Prepares inputs and applies network lerf or lerf_fine.
	virtual torch::Tensor RunLENetwork(torch::Tensor inputs, LeRF lerf, CuHashEmbedder lang_embed_fn, torch::Tensor lerf_point_scales = torch::Tensor());

	///Transforms model's predictions to semantically meaningful values.
	virtual LeRFRendererOutputs RawToLEOutputs(
		torch::Tensor raw_le,			///raw : [num_rays, num_samples along ray, 4+3+(3)] .Prediction from model.
		torch::Tensor z_vals_le,		///z_vals : [num_rays, num_samples along ray] .Integration time.
		torch::Tensor rays_d,		///rays_d : [num_rays, 3] .Direction of each ray.
		const int lang_embed_dim = 768,
		const float raw_noise_std = 0.f
	);

public:
	LeRFRenderer(
		CuHashEmbedder lang_embed_fn,
		LeRF lerf,
		torch::Tensor lerf_positives = torch::Tensor(), 
		torch::Tensor lerf_negatives = torch::Tensor()
	) : LangEmbedFn (lang_embed_fn), Lerf(lerf), LerfPositives(lerf_positives), LerfNegatives(lerf_negatives) {};
	virtual ~LeRFRenderer(){};

	std::tuple<torch::Tensor, torch::Tensor> GetLeRFPrompts(){ return std::make_tuple(LerfPositives, LerfNegatives); };
	void SetLeRFPrompts (const torch::Tensor lerf_positives, const torch::Tensor lerf_negatives){LerfPositives = lerf_positives; LerfNegatives = lerf_negatives;};

	///Volumetric rendering.
	virtual LeRFRenderResult RenderRays(
		torch::Tensor ray_batch,		///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
		torch::Tensor cone_angle,
		const int n_samples,
		const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
		const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
		const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		const int n_importance = 0,							///Number of additional times to sample along each ray.
		const bool white_bkgr = false,					///If True, assume a white background.
		const float raw_noise_std = 0.f,				///Локальная регуляризация плотности (выход) помогает избежать артефактов типа "облаков" затухает за n_iters / 3 итераций
		const float stochastic_preconditioning_alpha = 0.f,///добавляет шум к входу сети (координатам точек). Уменьшает чувствительность к инициализации. Помогает избежать "плавающих" артефактов
		torch::Tensor bounding_box = torch::Tensor(),
		//const int lang_embed_dim = 768,
		const bool return_weights = true,
		torch::Tensor lerf_scales = torch::Tensor(),			///Per-ray normalized pyramid scale (B,1); empty = eval default
		const float lerf_image_height = 0.f,			///View height in pixels for world-scale conversion
		const float lerf_focal_y = 0.f,			///View focal length fy in pixels
		const std::vector<float> &lerf_sweep_scales_norm = {}			///Actual pyramid levels the field was trained on; empty = legacy hardcoded list
	);

	///Render rays in smaller minibatches to save memory
	///rays_flat.sizes()[0] должно быть кратно размеру chunk
	virtual LeRFRenderResult BatchifyRays(
		torch::Tensor rays_flat,								///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
		torch::Tensor cone_angle,
		const int n_samples,
		const int chunk = 1024 * 32,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
		const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
		const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
		const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		const int n_importance = 0,							///Number of additional times to sample along each ray.
		const bool white_bkgr = false,					///If True, assume a white background.
		const float raw_noise_std = 0.,
		const float stochastic_preconditioning_alpha = 0.f,
		torch::Tensor bounding_box = torch::Tensor(),
		const bool return_weights = true,
		torch::Tensor lerf_scales = torch::Tensor(),			///Per-ray normalized pyramid scale (B,1); empty = eval default
		const float lerf_image_height = 0.f,			///View height in pixels for world-scale conversion
		const float lerf_focal_y = 0.f,			///View focal length fy in pixels
		const std::function<bool()> &is_aborted = {},			///Cancellation hook
		const std::vector<float> &lerf_sweep_scales_norm = {}			///Actual pyramid levels the field was trained on; empty = legacy hardcoded list
	);

	///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
	virtual LeRFRenderResult Render(
		const int h,					///Height of image in pixels.
		const int w,					///Width of image in pixels.
		torch::Tensor k,			///Сamera calibration
		const NeRFRenderParams &render_params,
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		torch::Tensor c2w_staticcam = torch::Tensor(),			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
		const std::function<bool()> &is_aborted = {}			///Cancellation hook polled between batches; return true to abort
	);
};			//LeRFRenderer