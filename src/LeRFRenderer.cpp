#include "LeRFRenderer.h"
#include "RuCLIPProcessor.h"	//Relevancy
#include <ATen/cuda/CUDAContext.h>

///Prepares inputs and applies network lerf or lerf_fine.
torch::Tensor LeRFRenderer :: RunLENetwork(torch::Tensor inputs, LeRF lerf, CuHashEmbedder lang_embed_fn, torch::Tensor lerf_point_scales)
{
	//return NeRFRenderer :: RunNetwork(inputs,	torch::Tensor(), lerf, lang_embed_fn, nullptr);

	//можно попробовать научить работать эмбедер с батчами чтобы не плющить тензоры?
	torch::Tensor inputs_flat = inputs.view({ -1, inputs.sizes().back()/*[-1]*/ });  //[1024, 256, 3] -> [262144, 3]

	auto [embedded, keep_mask] = lang_embed_fn->forward(inputs_flat);
	//Per-point scale conditioning: flatten (B, N, 1) -> (B*N, 1) aligned with the embedded rows; empty = default 1.0 in the field
	torch::Tensor scales_flat = torch::Tensor();
	if (lerf_point_scales.defined() && lerf_point_scales.numel() != 0)
		scales_flat = lerf_point_scales.reshape({ -1, 1 }).to(embedded.device(), embedded.scalar_type());

	torch::Tensor outputs_flat = lerf->forward(embedded, scales_flat);

	//set sigma to 0 for invalid points
	if (keep_mask.defined() && keep_mask.numel() != 0)
		outputs_flat.index_put_({ ~keep_mask, -1 }, 0);

	std::vector<int64_t> sz = inputs.sizes().vec();
	sz.pop_back();
	sz.push_back(outputs_flat.sizes().back());
	return outputs_flat.view(sz);  //list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]  //[262144, 5] -> [1024, 256, 5]
}

///Transforms model's predictions to semantically meaningful values.
LeRFRendererOutputs LeRFRenderer :: RawToLEOutputs(
	torch::Tensor raw_le,			///raw : [num_rays, num_samples along ray, 4+3+(3)] .Prediction from model.
	torch::Tensor z_vals_le,		///z_vals : [num_rays, num_samples along ray] .Integration time.
	torch::Tensor rays_d,			///rays_d : [num_rays, 3] .Direction of each ray.
	const int lang_embed_dim /*= 768*/,
	const float raw_noise_std /*= 0.*/
) {
	torch::Device device = raw_le.device();
	LeRFRendererOutputs result;

	auto raw2alpha = [](torch::Tensor raw, torch::Tensor dists) {return 1.f - torch::exp(- torch::clamp(torch::relu(raw) * dists.contiguous(), 0.0f, 38.0f));};

	auto dists_le = z_vals_le.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) - z_vals_le.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });
	//dists_le = torch::cat({ dists_le, dists_le.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }) }, -1);
	dists_le = torch::cat({ dists_le, (torch::ones(1, torch::kFloat32) * 1e8).expand(dists_le.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }).sizes()).to(device)}, -1);  // [N_rays, N_samples]
	dists_le = dists_le * torch::norm(rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }), 2/*L2*/, /*dim*/-1);
	if (!dists_le.requires_grad())
		dists_le.set_requires_grad(true);

	int cur_pos = 0;
	result.LangEmbedding = /*torch::tanh(*/raw_le.index({ "...",  torch::indexing::Slice(cur_pos, cur_pos + lang_embed_dim) })/*)*/;
	cur_pos += lang_embed_dim;
	
	auto le_density_before_activation = raw_le.index({ "...", cur_pos});		//    извлекает значения (sigma_le) очередного столбца raw
	if (raw_noise_std > 0.f)
		le_density_before_activation = le_density_before_activation + torch::randn_like(le_density_before_activation) * raw_noise_std;

	torch::Tensor le_alpha = raw2alpha(le_density_before_activation, dists_le);  //[N_rays, N_samples]

	//auto compute_weights_in_log_space = [](torch::Tensor alpha, torch::Device device)
	//{
	//	//Вычисляем log(1 - alpha) с защитой от log(0)
	//	auto log_transmittance = torch::cumsum(torch::log(torch::clamp(1.0f - alpha, 1e-8f, 1.f)), -1);
	//	auto weights = alpha * torch::exp(log_transmittance);
	//	return weights;
	//};
	//result.WeightsLE = compute_weights_in_log_space(le_alpha, device);
	result.WeightsLE = le_alpha * torch::cumprod(
			torch::cat({ torch::ones({le_alpha.sizes()[0], 1}).to(device), torch::clamp(1.f - le_alpha , 1e-8f, 1.f) }, -1),
			-1
		).index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1) });
	result.DepthMapLE = torch::sum(result.WeightsLE * z_vals_le, -1) / torch::clamp_min(torch::sum(result.WeightsLE, -1), 1e-10f);
	result.DispMapLE = 1. / torch::max(1e-10f * torch::ones_like(result.DepthMapLE), result.DepthMapLE);
	result.AccMapLE = torch::sum(result.WeightsLE, -1);
	cur_pos += 1;

	result.RenderedLangEmbedding = RenderCLIPEmbedding(result.LangEmbedding, result.WeightsLE.unsqueeze(-1));
	//result.RenderedLangEmbedding = RenderCLIPEmbedding(result.LangEmbedding, result.Weights.unsqueeze(-1).detach());
	//result.RenderedLanguageEmbedding = CLIPEmbeddingShader(result.RenderedLanguageEmbedding);

	//Relevancy is no longer stored here: RenderRays derives the single RelevancyMap at the pass scale or via the inference sweep

	return result;
}

///Volumetric rendering.
LeRFRenderResult LeRFRenderer :: RenderRays(
	torch::Tensor ray_batch,		///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
	torch::Tensor cone_angle,
	const int n_samples,
	const bool return_raw /*= false*/,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp /*= false*/,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb /*= 0.f*/,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance /*= 0*/,							///Number of additional times to sample along each ray.
	const bool white_bkgr /*= false*/,					///If True, assume a white background.
	const float raw_noise_std /*= 0.f*/,				///Локальная регуляризация плотности (выход) помогает избежать артефактов типа "облаков" затухает за n_iters / 3 итераций
	const float stochastic_preconditioning_alpha /*= 0.f*/,///добавляет шум к входу сети (координатам точек). Уменьшает чувствительность к инициализации. Помогает избежать "плавающих" артефактов
	torch::Tensor bounding_box /*= torch::Tensor()*/,
	//const int lang_embed_dim /*= 768*/,
	const bool return_weights /*= true*/,
	torch::Tensor lerf_scales /*= torch::Tensor()*/,					///Per-ray normalized pyramid scale (B,1); empty = eval default
	const float lerf_image_height /*= 0.f*/,					///View height in pixels for world-scale conversion
	const float lerf_focal_y /*= 0.f*/,						///View focal length fy in pixels
	const std::vector<float> &lerf_sweep_scales_norm /*= {}*/					///Actual pyramid levels the field was trained on; empty = legacy hardcoded list
	) {
	LeRFRenderResult lerf_result;
	torch::Device device = ray_batch.device();		//Передать параметром??

	///!!!Можно просто передавать структурку не парсить тензор
	int nrays = (int)ray_batch.sizes()[0];
	auto rays_o = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3) });			//[N_rays, 3]		Origins
	auto rays_d = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(3, 6) });			//[N_rays, 3]		Directions
	torch::Tensor viewdirs;
	if (ray_batch.sizes().back() > 8)
		viewdirs = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(-3, torch::indexing::None) });
	auto ray_bounds = torch::reshape(ray_batch.index({ "...", torch::indexing::Slice(6, 8) }), { -1, 1, 2 });
	auto near = ray_bounds.index({ "...", 0 });
	auto far = ray_bounds.index({ "...", 1 }); //[-1, 1

	//Per-point world-scale conversion: s_world = s_norm * H * dist / fy.  dist is the Euclidean
	auto compute_point_scales = [&](const torch::Tensor &sample_pts) -> torch::Tensor {
		if (!lerf_scales.defined() || lerf_scales.numel() == 0 || lerf_image_height <= 0.f || lerf_focal_y <= 0.f)
			return torch::Tensor();
		//s_norm must broadcast against the 3-D dist [B, N, 1]: a 2-D [B, 1] 
		auto s_norm = lerf_scales.to(ray_batch.device()).to(torch::kFloat32).reshape({ -1, 1, 1 });
		auto dist = torch::norm(sample_pts - rays_o.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice() }), 2/*L2*/, /*dim=*/-1, /*keepdim=*/true);	//[B, N, 1]
		return s_norm * lerf_image_height * dist / lerf_focal_y;
	};

	torch::Tensor t_vals = torch::linspace(0.f, 1.f, n_samples, torch::kFloat).to(device);
	torch::Tensor z_vals;

	if (!lin_disp)
	{
		z_vals = near * (1.f - t_vals) + far * (t_vals);
	} else {
		//z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals));
		auto safe_inv = [](torch::Tensor x, float eps = 1e-8f) {return torch::where(torch::abs(x) < eps, torch::ones_like(x) / eps, 1.0f / x); };
		z_vals = safe_inv(safe_inv(near) * (1. - t_vals) + safe_inv(far) * t_vals);
	}

	if (perturb > 0.)
	{
		//get intervals between samples
		auto mids = 0.5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto upper = torch::cat({ mids, z_vals.index({ "...", torch::indexing::Slice(-1, torch::indexing::None)}) }, -1);
		auto lower = torch::cat({ z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, 1)}), mids }, -1);
		//stratified samples in those intervals
		//auto t_rand = torch::rand(z_vals.sizes());
		//z_vals = lower + (upper - lower) * t_rand;
		auto intervals = upper - lower;
		auto mask = intervals > 1e-8f;
		auto t_rand = torch::rand(z_vals.sizes(), device = device);
		z_vals = lower + torch::where(mask, intervals * t_rand, torch::zeros_like(intervals));
	}

	auto pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) + rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None}); //[N_rays, N_samples, 3]
	pts = TangentScatter(pts, z_vals, near, cone_angle, rays_d, device, bounding_box);

	torch::Tensor raw =  RunLENetwork(pts, Lerf, LangEmbedFn, compute_point_scales(pts));
	auto outputs1 = RawToLEOutputs(raw, z_vals, rays_d, Lerf->GetLangEmbedDim(), raw_noise_std);

	if (n_importance > 0)
	{
		auto z_vals_mid = .5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto z_samples = SamplePDF(z_vals_mid, outputs1.WeightsLE.index({ "...", torch::indexing::Slice(1, -1) }), n_importance, perturb == 0.);
		z_samples = z_samples.detach();
		torch::Tensor z_indices;
		std::tie(z_vals, z_indices) = torch::sort(torch::cat({ z_vals, z_samples }, -1), -1);
		pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) + rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None }); // [N_rays, N_samples + N_importance, 3]
		pts = TangentScatter(pts, z_vals, near, cone_angle, rays_d, device, bounding_box);

		//Применяем стохастическое предобуславливание
		if (stochastic_preconditioning_alpha > 0.0f)
		{
			std::vector<torch::Tensor> bounds = torch::split(bounding_box, { 3, 3 }, -1);
			auto min_bound = bounds[0].to(device);
			auto max_bound = bounds[1].to(device);
			auto noise = torch::randn_like(pts) * stochastic_preconditioning_alpha;
			pts = pts + noise.to(device);
			pts = ReflectBoundary(pts, min_bound, max_bound);	//Обрабатываем границы отражением
		}

		raw = RunLENetwork(pts, Lerf, LangEmbedFn, compute_point_scales(pts));
		lerf_result.Outputs = RawToLEOutputs(raw, z_vals, rays_d, Lerf->GetLangEmbedDim(), raw_noise_std);
		//result.ZStd = torch::std(z_samples, -1, false);  // [N_rays]
	}	else
		lerf_result.Outputs = outputs1;

	//Final relevancy. With per-ray pyramid scales (training-style passes) it is taken at the pass scale;
	//the inference sweep below overwrites it with the max over scales when no per-ray scales are supplied.
	if ((lerf_scales.defined() && lerf_scales.numel() != 0) && LerfPositives.defined() && LerfPositives.numel() != 0 && LerfNegatives.defined() && LerfNegatives.numel() != 0)
	{
		torch::NoGradGuard no_grad;
		auto rel = Relevancy(lerf_result.Outputs.RenderedLangEmbedding, LerfPositives.to(device), LerfNegatives.to(device));
		lerf_result.Outputs.RelevancyMap = rel.index({ torch::indexing::Slice(), 0 }).unsqueeze(-1);	//[B, 1]
	}

	//Inference scale sweep (get_max_across): composite relevancy is evaluated at several observation scales and the per-ray max is kept. Training passes carry
	//per-ray pyramid scales and skip this block; sweep scales are pyramid-normalized and converted to world units exactly like training: s = s_norm * H * dist / fy.
	if ((!lerf_scales.defined() || lerf_scales.numel() == 0)
		&& lerf_image_height > 0.f && lerf_focal_y > 0.f
		&& LerfPositives.defined() && LerfPositives.numel() != 0
		&& LerfNegatives.defined() && LerfNegatives.numel() != 0)
	{
		//Max over scales is non-differentiable and inference-only: never build a graph here
		//(also protects RenderPath test renders issued from inside the training loop)
		torch::NoGradGuard no_grad;
		auto positives_d = LerfPositives.to(device),
			 negatives_d = LerfNegatives.to(device);
		//Sequential sweep: one network pass per scale with a running max over the positive probability (Relevancy column 0).
		//dist is the Euclidean distance to the actual sample positions (after TangentScatter offsets),
		auto dist = torch::norm(pts - rays_o.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice() }), 2/*L2*/, /*dim=*/-1, /*keepdim=*/true);	//[B, N, 1]
		const float world_scale = lerf_image_height / lerf_focal_y;
		const std::vector<float> legacy_sweep_scales_norm = { 1.0f, 1.0f / 3.0f, 1.0f / 9.0f, 0.0f };
		const std::vector<float> &sweep_scales = lerf_sweep_scales_norm.empty() ? legacy_sweep_scales_norm : lerf_sweep_scales_norm;
		torch::Tensor pos_prob_max;	//[B] running max over the scales
		for (const float s_norm : sweep_scales)
		{
			//Per-point scale conditioning in world units: s = s_norm * H * dist / fy (same as training)
			auto point_scales = (s_norm * world_scale) * dist;	//[B, N, 1] (dist is already [B, N, 1])
			auto raw_s = RunLENetwork(pts, Lerf, LangEmbedFn, point_scales);
			auto outs_s = RawToLEOutputs(raw_s, z_vals, rays_d, Lerf->GetLangEmbedDim(), 0.f);
			//Positive probability vs the hardest negative (column 0)
			auto rel_s = Relevancy(outs_s.RenderedLangEmbedding, positives_d, negatives_d);
			auto pos_prob = rel_s.index({ torch::indexing::Slice(), 0 });	//[B]
			pos_prob_max = pos_prob_max.defined() ? torch::maximum(pos_prob_max, pos_prob) : pos_prob;
		}
		lerf_result.Outputs.RelevancyMap = pos_prob_max.unsqueeze(-1);	//[B, 1]
	}

	if (return_raw)
		lerf_result.Raw = raw;

	if (!return_weights)
	{
		lerf_result.Outputs.WeightsLE = torch::Tensor();
		lerf_result.Outputs.LangEmbedding = torch::Tensor();
		lerf_result.Outputs.RenderedLangEmbedding = torch::Tensor();
	}
	return lerf_result;
}


///Render rays in smaller minibatches to save memory
///rays_flat.sizes()[0] должно быть кратно размеру chunk
LeRFRenderResult LeRFRenderer :: BatchifyRays(
	torch::Tensor rays_flat,								///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
	torch::Tensor cone_angle,
	const int n_samples,
	const int chunk /*= 1024 * 32*/,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
	const bool return_raw /*= false*/,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp /*= false*/,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb /*= 0.f*/,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance /*= 0*/,							///Number of additional times to sample along each ray.
	const bool white_bkgr /*= false*/,					///If True, assume a white background.
	const float raw_noise_std /*= 0.*/,
	const float stochastic_preconditioning_alpha /*= 0.f*/,
	torch::Tensor bounding_box /* = torch::Tensor()*/,
	const bool return_weights /*= true*/,
	torch::Tensor lerf_scales /*= torch::Tensor()*/,					///Per-ray normalized pyramid scale (B,1); empty = eval default
	const float lerf_image_height /*= 0.f*/,					///View height in pixels for world-scale conversion
	const float lerf_focal_y /*= 0.f*/,						///View focal length fy in pixels
	const std::function<bool()> &is_aborted /*= {}*/,			///Cancellation hook polled between batches; return true to abort
	const std::vector<float> &lerf_sweep_scales_norm /*= {}*/					///Actual pyramid levels the field was trained on; empty = legacy hardcoded list
) {
	LeRFRenderResult result;
	std::vector<LeRFRenderResult> all_results;
	all_results.reserve(rays_flat.sizes()[0] / chunk);
	for (int i = 0; i < rays_flat.sizes()[0]; i += chunk)
	{
		//Cooperative cancellation point: an abort is noticed within at most one chunk of GPU work
		if (is_aborted && is_aborted())
		{
			result.Aborted = true;
			break;
		}
		all_results.emplace_back(RenderRays(
			rays_flat.index({ torch::indexing::Slice(i, ((i + chunk) <= rays_flat.sizes()[0])?(i+chunk):(rays_flat.sizes()[0])) }),
			cone_angle,/*пока скаляр, потом .index({ torch::indexing::Slice(i, ((i + chunk) <= cone_angle.sizes()[0]) ? (i + chunk) : (cone_angle.sizes()[0])) }),*/
			n_samples,
			return_raw,
			lin_disp,
			perturb,
			n_importance,
			white_bkgr,
			raw_noise_std,
			stochastic_preconditioning_alpha,
			bounding_box,
			return_weights,
			(lerf_scales.defined() && lerf_scales.numel() != 0)
				? lerf_scales.index({ torch::indexing::Slice(i, ((i + chunk) <= lerf_scales.sizes()[0]) ? (i + chunk) : lerf_scales.sizes()[0]) })
				: torch::Tensor(),
			lerf_image_height,
			lerf_focal_y,
			lerf_sweep_scales_norm
		));
		//Pacing (interactive only): wait until the GPU has actually executed this chunk.
		//Without it the host enqueues the whole level in microseconds and blocks in the
		//end-of-level item<float>() sync, so the cancellation check above would only fire
		//between fully executed levels, making the whole level the indivisible unit of work.
		if (is_aborted && rays_flat.is_cuda())
			at::cuda::getCurrentCUDAStream().synchronize();
	}
	//Слить all_results в один RenderResult используя torch::cat(torch::TensorList - at::ArrayRef ...
	//!!!make this part cleaner and shorter
	std::vector <torch::Tensor> out_disp_map_le,
		out_acc_map_le,
		out_weights_le,
		out_depth_map_le,
		out_lang_embedding,
		out_rendered_lang_embedding,
		out_relevancy_map,
		out0_disp_map_le,
		out0_acc_map_le,
		out0_weights_le,
		out0_depth_map_le,
		out0_lang_embedding,
		out0_rendered_lang_embedding,
		out0_relevancy_map,
		raw;
	for (auto it : all_results)
	{
		if (it.Outputs.DispMapLE.defined()) out_disp_map_le.push_back(it.Outputs.DispMapLE);
		if (it.Outputs.AccMapLE.defined()) out_acc_map_le.push_back(it.Outputs.AccMapLE);
		if (it.Outputs.WeightsLE.defined()) out_weights_le.push_back(it.Outputs.WeightsLE);
		if (it.Outputs.DepthMapLE.defined()) out_depth_map_le.push_back(it.Outputs.DepthMapLE);
		if (it.Outputs.LangEmbedding.defined()) out_lang_embedding.push_back(it.Outputs.LangEmbedding);
		if (it.Outputs.RenderedLangEmbedding.defined()) out_rendered_lang_embedding.push_back(it.Outputs.RenderedLangEmbedding);
		if (it.Outputs.RelevancyMap.defined()) out_relevancy_map.push_back(it.Outputs.RelevancyMap);
		if (it.Raw.defined()) raw.push_back(it.Raw);
	}
	if (!out_disp_map_le.empty()) result.Outputs.DispMapLE = torch::cat(out_disp_map_le, 0);
	if (!out_acc_map_le.empty()) result.Outputs.AccMapLE = torch::cat(out_acc_map_le, 0);
	if (!out_weights_le.empty()) result.Outputs.WeightsLE = torch::cat(out_weights_le, 0);
	if (!out_depth_map_le.empty()) result.Outputs.DepthMapLE = torch::cat(out_depth_map_le, 0);
	if (!out_lang_embedding.empty()) result.Outputs.LangEmbedding = torch::cat(out_lang_embedding, 0);
	if (!out_rendered_lang_embedding.empty()) result.Outputs.RenderedLangEmbedding = torch::cat(out_rendered_lang_embedding, 0);
	if (!out_relevancy_map.empty()) result.Outputs.RelevancyMap = torch::cat(out_relevancy_map, 0);
	if (!raw.empty()) result.Raw = torch::cat(raw, 0);

	return result;
}

///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
LeRFRenderResult LeRFRenderer :: Render(
	const int h,					///Height of image in pixels.
	const int w,					///Width of image in pixels.
	torch::Tensor k,			///Сamera calibration
	const NeRFRenderParams &render_params,
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rays /*= { torch::Tensor(), torch::Tensor(), torch::Tensor() }*/,			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	torch::Tensor c2w /*= torch::Tensor()*/,			///array of shape[3, 4].Camera - to - world transformation matrix.
	torch::Tensor c2w_staticcam /*= torch::Tensor()*/,			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
	const std::function<bool()> &is_aborted /*= {}*/			///Cancellation hook polled between batches; return true to abort
) {
	torch::Tensor rays_o, rays_d, cone_angle;
	if (c2w.defined() && c2w.numel() != 0)
	{
		//special case to render full image
		std::tie(rays_o, rays_d, cone_angle) = GetRays(h, w, k, c2w);
	}	else {
		//use provided ray batch
		std::tie(rays_o, rays_d, cone_angle) = rays;
	}

	auto sh = rays_d.sizes();			//[..., 3]
	if (render_params.Ndc)
	{
		//for forward facing scenes
		std::tie(rays_o, rays_d, cone_angle) = NDCRays(h, w, k[0][0].item<float>()/*focal*/, 1.f, rays_o, rays_d, render_params.ThinRay ? torch::Tensor() : cone_angle);
	}

	//Create ray batch
	rays_o = torch::reshape(rays_o, { -1, 3 });//.float()
	rays_d = torch::reshape(rays_d, { -1, 3 });//.float()

	//Calculate individual near and far for each ray using AABB intersection
	torch::Tensor near_, far_;
	std::tie(near_, far_) = IntersectWithAABB(rays_o, rays_d, render_params.BoundingBox, 0.f/*near*/);
	near_ = near_.unsqueeze(-1);  // Shape: [num_rays, 1]
	far_ = far_.unsqueeze(-1);    // Shape: [num_rays, 1]

	auto rays_ = torch::cat({ rays_o, rays_d, near_, far_ }, -1);
	
	//Inference sweep geometry: the viewer path supplies no per-ray pyramid scales, so fall
	//back to the rendered image height and fy (RenderRays rebuilds per-point world scales);
	//downsampling cancels out: (H/f)*dist/(fy/f) == H*dist/fy, matching training
	const float sweep_height = (render_params.LeRFImageHeight > 0.f) ? render_params.LeRFImageHeight : static_cast<float>(h);
	const float sweep_fy = (render_params.LeRFFocalY > 0.f) ? render_params.LeRFFocalY : k[1][1].item<float>();
	//Render and reshape
	LeRFRenderResult all_ret = std::move(BatchifyRays(
		rays_, render_params.ThinRay ? torch::Tensor() : cone_angle, render_params.NSamples, render_params.Chunk, render_params.ReturnRaw, render_params.LinDisp, render_params.Perturb,
		render_params.NImportance, render_params.WhiteBkgr, render_params.RawNoiseStd, render_params.StochasticPreconditioningAlpha, render_params.BoundingBox, render_params.ReturnWeights,
		render_params.LeRFScales, sweep_height, sweep_fy, is_aborted, render_params.LeRFSweepScalesNorm
	));

	//A partial (aborted) result has fewer rows than h*w; reshaping it to the full image would throw
	if (!all_ret.Aborted)
	{
		if (sh.size() > 2)			//не [4096, 3] а [800,800,3]
		{
			if (all_ret.Outputs.DispMapLE.numel() != 0)
				all_ret.Outputs.DispMapLE = torch::reshape(all_ret.Outputs.DispMapLE, { sh[0], sh[1] });	//[640000] -> [800,800]
			if (all_ret.Outputs.DepthMapLE.numel() != 0)
				all_ret.Outputs.DepthMapLE = torch::reshape(all_ret.Outputs.DepthMapLE, { sh[0], sh[1] });

			if (all_ret.Outputs.RenderedLangEmbedding.numel() != 0)
				all_ret.Outputs.RenderedLangEmbedding = torch::reshape(all_ret.Outputs.RenderedLangEmbedding, { sh[0], sh[1], Lerf->GetLangEmbedDim() });

			if (all_ret.Outputs.RelevancyMap.numel() != 0)
				all_ret.Outputs.RelevancyMap = torch::reshape(all_ret.Outputs.RelevancyMap, { sh[0], sh[1] });
		} else {
			if (all_ret.Outputs.RenderedLangEmbedding.numel() != 0)
				all_ret.Outputs.RenderedLangEmbedding = torch::reshape(all_ret.Outputs.RenderedLangEmbedding, { sh[0], Lerf->GetLangEmbedDim() });
			if (all_ret.Outputs.RelevancyMap.numel() != 0)
				all_ret.Outputs.RelevancyMap = torch::reshape(all_ret.Outputs.RelevancyMap, { sh[0] });		}
	}
	all_ret.Near = near_.min().item<float>();
	all_ret.Far = far_.max().item<float>();
	return all_ret;
}