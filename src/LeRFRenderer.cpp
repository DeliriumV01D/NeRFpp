#include "LeRFRenderer.h"
#include "RuCLIPProcessor.h"	//Relevancy

///Prepares inputs and applies network lerf or lerf_fine.
torch::Tensor LeRFRenderer :: RunLENetwork(torch::Tensor inputs, LeRF lerf, CuHashEmbedder lang_embed_fn)
{
	//return NeRFRenderer :: RunNetwork(inputs,	torch::Tensor(), lerf, lang_embed_fn, nullptr);

	//можно попробовать научить работать эмбедер с батчами чтобы не плющить тензоры?
	torch::Tensor inputs_flat = inputs.view({ -1, inputs.sizes().back()/*[-1]*/ });  //[1024, 256, 3] -> [262144, 3]
	inputs_flat.set_requires_grad(false);
	auto [embedded, keep_mask] = lang_embed_fn->forward(inputs_flat); 

	torch::Tensor outputs_flat = lerf->forward(embedded);

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

	auto raw2alpha = [](torch::Tensor raw, torch::Tensor dists) {return -torch::exp(-torch::relu(raw) * dists) + 1.f; };

	auto dists_le = z_vals_le.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) - z_vals_le.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });
	dists_le = torch::cat({ dists_le, (torch::ones(1, torch::kFloat32) * 1e10).expand(dists_le.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }).sizes()).to(device)}, -1);  // [N_rays, N_samples]
	dists_le = dists_le * torch::norm(rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }), 2/*L2*/, /*dim*/-1);
	if (!dists_le.requires_grad())
		dists_le.set_requires_grad(true);

	int cur_pos = 0;
	result.LangEmbedding = /*torch::tanh(*/raw_le.index({ "...",  torch::indexing::Slice(cur_pos, cur_pos + lang_embed_dim) })/*)*/;
	cur_pos += lang_embed_dim;
		
	auto le_density_before_activation = raw_le.index({ "...", cur_pos});		//    извлекает значения (sigma_le) очередного столбца raw
	if (raw_noise_std > 0.f)
	{
		le_density_before_activation = le_density_before_activation + torch::randn_like(le_density_before_activation) * raw_noise_std;
	}	
	torch::Tensor le_alpha = raw2alpha(le_density_before_activation, dists_le);  //[N_rays, N_samples]
	result.WeightsLE = le_alpha * torch::cumprod(
			torch::cat({ torch::ones({le_alpha.sizes()[0], 1}).to(device), -le_alpha + 1.f + 1e-10f }, -1),
			-1
		).index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1) });
	result.DepthMapLE = torch::sum(result.WeightsLE * z_vals_le, -1) / torch::sum(result.WeightsLE, -1);
	result.DispMapLE = 1. / torch::max(1e-10 * torch::ones_like(result.DepthMapLE), result.DepthMapLE);
	result.AccMapLE = torch::sum(result.WeightsLE, -1);
	cur_pos += 1;

	result.RenderedLangEmbedding = RenderCLIPEmbedding(result.LangEmbedding, result.WeightsLE.unsqueeze(-1));
	//result.RenderedLangEmbedding = RenderCLIPEmbedding(result.LangEmbedding, result.Weights.unsqueeze(-1).detach());
	//result.RenderedLanguageEmbedding = CLIPEmbeddingShader(result.RenderedLanguageEmbedding);

	result.Relevancy = Relevancy(result.RenderedLangEmbedding, LerfPositives.to(device), LerfNegatives.to(device));

	return result;
}

///Volumetric rendering.
LeRFRenderResult LeRFRenderer :: RenderRays(
	torch::Tensor ray_batch,		///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
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
	const bool return_weights /*= true*/
){
	LeRFRenderResult lerf_result;
	torch::Device device = ray_batch.device();		//Передать параметром??

	///!!!Можно просто передавать структурку не парсить тензор
	int nrays = ray_batch.sizes()[0];
	auto rays_o = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3) });			//[N_rays, 3]		Origins
	auto rays_d = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(3, 6) });			//[N_rays, 3]		Directions
	torch::Tensor viewdirs;
	if (ray_batch.sizes().back() > 8)
		viewdirs = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(-3, torch::indexing::None) });
	auto ray_bounds = torch::reshape(ray_batch.index({ "...", torch::indexing::Slice(6, 8) }), { -1, 1, 2 });
	auto near = ray_bounds.index({ "...", 0 });
	auto far = ray_bounds.index({ "...", 1 }); //[-1, 1

	torch::Tensor t_vals = torch::linspace(0.f, 1.f, n_samples, torch::kFloat).to(device);
	torch::Tensor z_vals;
	if (!lin_disp)
	{
		z_vals = near * (1. - t_vals) + far * (t_vals);
	}
	else {
		z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals));
	}

	if (perturb > 0.)
	{
		//get intervals between samples
		auto mids = 0.5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto upper = torch::cat({ mids, z_vals.index({ "...", torch::indexing::Slice(-1, torch::indexing::None)}) }, -1);
		auto lower = torch::cat({ z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, 1)}), mids }, -1);
		//stratified samples in those intervals
		auto t_rand = torch::rand(z_vals.sizes());
		z_vals = lower + (upper - lower) * t_rand;
	}

	auto pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) + rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None}); //[N_rays, N_samples, 3]
	torch::Tensor raw =  RunLENetwork(pts, Lerf, LangEmbedFn);
	lerf_result.Outputs1 = RawToLEOutputs(raw, z_vals, rays_d, Lerf->GetLangEmbedDim(), raw_noise_std);

	if (n_importance > 0)
	{
		lerf_result.Outputs0 = lerf_result.Outputs1;
		auto z_vals_mid = .5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto z_samples = SamplePDF(z_vals_mid, lerf_result.Outputs1.WeightsLE.index({ "...", torch::indexing::Slice(1, -1) }), n_importance, perturb == 0.);
		z_samples = z_samples.detach();
		torch::Tensor z_indices;
		std::tie(z_vals, z_indices) = torch::sort(torch::cat({ z_vals, z_samples }, -1), -1);
		pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) + rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None }); // [N_rays, N_samples + N_importance, 3]
		
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

		if (!LerfFine /*!= nullptr*/)
		{
			raw = RunLENetwork(pts, Lerf, LangEmbedFn);
		}	else {
			raw = RunLENetwork(pts, LerfFine, LangEmbedFn);
		}
		lerf_result.Outputs1 = RawToLEOutputs(raw, z_vals, rays_d, Lerf->GetLangEmbedDim(), raw_noise_std);
		//result.ZStd = torch::std(z_samples, -1, false);  // [N_rays]
	}

	if (return_raw)
		lerf_result.Raw = raw;

	if (!return_weights)
	{
		lerf_result.Outputs0.WeightsLE = torch::Tensor();
		lerf_result.Outputs1.WeightsLE = torch::Tensor();
		lerf_result.Outputs0.LangEmbedding = torch::Tensor();
		lerf_result.Outputs1.LangEmbedding = torch::Tensor();
		lerf_result.Outputs0.RenderedLangEmbedding = torch::Tensor();
		lerf_result.Outputs1.RenderedLangEmbedding = torch::Tensor();
	}
	return lerf_result;
}


///Render rays in smaller minibatches to save memory
///rays_flat.sizes()[0] должно быть кратно размеру chunk
LeRFRenderResult LeRFRenderer :: BatchifyRays(
	torch::Tensor rays_flat,								///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
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
	const bool return_weights /*= true*/
) {
	LeRFRenderResult result;
	std::vector<LeRFRenderResult> all_results;
	all_results.reserve(rays_flat.sizes()[0] / chunk);
	for (int i = 0; i < rays_flat.sizes()[0]; i += chunk)
	{
		all_results.emplace_back(RenderRays(
			rays_flat.index({ torch::indexing::Slice(i, ((i + chunk) <= rays_flat.sizes()[0])?(i+chunk):(rays_flat.sizes()[0])) }),
			n_samples,
			return_raw,
			lin_disp,
			perturb,
			n_importance,
			white_bkgr,
			raw_noise_std,
			stochastic_preconditioning_alpha,
			bounding_box,
			return_weights
		));
	}
	//Слить all_results в один RenderResult используя torch::cat(torch::TensorList - at::ArrayRef ...
	//!!!make this part cleaner and shorter
	std::vector <torch::Tensor> out_disp_map_le,
		out_acc_map_le,
		out_weights_le,
		out_depth_map_le,
		out_lang_embedding,
		out_rendered_lang_embedding,
		out_relevancy,
		out0_disp_map_le,
		out0_acc_map_le,
		out0_weights_le,
		out0_depth_map_le,
		out0_lang_embedding,
		out0_rendered_lang_embedding,
		out0_relevancy,
		raw;
	for (auto it : all_results)
	{
		if (it.Outputs1.DispMapLE.defined()) out_disp_map_le.push_back(it.Outputs1.DispMapLE);
		if (it.Outputs1.AccMapLE.defined()) out_acc_map_le.push_back(it.Outputs1.AccMapLE);
		if (it.Outputs1.WeightsLE.defined()) out_weights_le.push_back(it.Outputs1.WeightsLE);
		if (it.Outputs1.DepthMapLE.defined()) out_depth_map_le.push_back(it.Outputs1.DepthMapLE);
		if (it.Outputs1.LangEmbedding.defined()) out_lang_embedding.push_back(it.Outputs1.LangEmbedding);
		if (it.Outputs1.RenderedLangEmbedding.defined()) out_rendered_lang_embedding.push_back(it.Outputs1.RenderedLangEmbedding);
		if (it.Outputs1.Relevancy.defined()) out_relevancy.push_back(it.Outputs1.Relevancy);
		if (it.Outputs0.DispMapLE.defined()) out0_disp_map_le.push_back(it.Outputs0.DispMapLE);
		if (it.Outputs0.AccMapLE.defined()) out0_acc_map_le.push_back(it.Outputs0.AccMapLE);
		if (it.Outputs0.WeightsLE.defined()) out0_weights_le.push_back(it.Outputs0.WeightsLE);
		if (it.Outputs0.DepthMapLE.defined()) out0_depth_map_le.push_back(it.Outputs0.DepthMapLE);
		if (it.Outputs0.LangEmbedding.defined()) out0_lang_embedding.push_back(it.Outputs0.LangEmbedding);
		if (it.Outputs0.RenderedLangEmbedding.defined()) out0_rendered_lang_embedding.push_back(it.Outputs0.RenderedLangEmbedding);
		if (it.Outputs0.Relevancy.defined()) out0_relevancy.push_back(it.Outputs0.Relevancy);
		if (it.Raw.defined()) raw.push_back(it.Raw);
	}
	if (!out_disp_map_le.empty()) result.Outputs1.DispMapLE = torch::cat(out_disp_map_le, 0);
	if (!out_acc_map_le.empty()) result.Outputs1.AccMapLE = torch::cat(out_acc_map_le, 0);
	if (!out_weights_le.empty()) result.Outputs1.WeightsLE = torch::cat(out_weights_le, 0);
	if (!out_depth_map_le.empty()) result.Outputs1.DepthMapLE = torch::cat(out_depth_map_le, 0);
	if (!out_lang_embedding.empty()) result.Outputs1.LangEmbedding = torch::cat(out_lang_embedding, 0);
	if (!out_rendered_lang_embedding.empty()) result.Outputs1.RenderedLangEmbedding = torch::cat(out_rendered_lang_embedding, 0);
	if (!out_relevancy.empty()) result.Outputs1.Relevancy = torch::cat(out_relevancy, 0);
	if (!out0_disp_map_le.empty()) result.Outputs0.DispMapLE = torch::cat(out0_disp_map_le, 0);
	if (!out0_acc_map_le.empty()) result.Outputs0.AccMapLE = torch::cat(out0_acc_map_le, 0);
	if (!out0_weights_le.empty()) result.Outputs0.WeightsLE = torch::cat(out0_weights_le, 0);
	if (!out0_depth_map_le.empty()) result.Outputs0.DepthMapLE = torch::cat(out0_depth_map_le, 0);
	if (!out0_lang_embedding.empty()) result.Outputs0.LangEmbedding = torch::cat(out0_lang_embedding, 0);
	if (!out0_rendered_lang_embedding.empty()) result.Outputs0.RenderedLangEmbedding = torch::cat(out0_rendered_lang_embedding, 0);
	if (!out0_relevancy.empty()) result.Outputs0.Relevancy = torch::cat(out0_relevancy, 0);
	if (!raw.empty()) result.Raw = torch::cat(raw, 0);

	return result;
}

///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
LeRFRenderResult LeRFRenderer :: Render(
	const int h,					///Height of image in pixels.
	const int w,					///Width of image in pixels.
	torch::Tensor k,			///Сamera calibration
	const NeRFRenderParams &render_params,
	std::pair<torch::Tensor, torch::Tensor> rays /*= { torch::Tensor(), torch::Tensor() }*/,			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	torch::Tensor c2w /*= torch::Tensor()*/,			///array of shape[3, 4].Camera - to - world transformation matrix.
	torch::Tensor c2w_staticcam /*= torch::Tensor()*/			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
) {
	torch::Tensor rays_o, rays_d;
	if (c2w.defined() && c2w.numel() != 0)
	{
		//special case to render full image
		std::tie(rays_o, rays_d) = GetRays(h, w, k, c2w);
	}	else {
		//use provided ray batch
		std::tie(rays_o, rays_d) = rays;
	}

	auto sh = rays_d.sizes();			//[..., 3]
	if (render_params.Ndc)
	{
		//for forward facing scenes
		std::tie(rays_o, rays_d) = NDCRays(h, w, k[0][0].item<float>()/*focal*/, 1.f, rays_o, rays_d);
	}

	//Create ray batch
	rays_o = torch::reshape(rays_o, { -1, 3 });//.float()
	rays_d = torch::reshape(rays_d, { -1, 3 });//.float()

	auto near_ = render_params.Near * torch::ones_like(rays_d.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }));
	auto far_ = render_params.Far * torch::ones_like(rays_d.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }));
	auto rays_ = torch::cat({ rays_o, rays_d, near_, far_ }, -1);
	
	//Render and reshape
	LeRFRenderResult all_ret = std::move(BatchifyRays(
		rays_, 
		render_params.NSamples, render_params.Chunk, render_params.ReturnRaw, render_params.LinDisp, render_params.Perturb,
		render_params.NImportance, render_params.WhiteBkgr, render_params.RawNoiseStd, render_params.StochasticPreconditioningAlpha, render_params.BoundingBox, render_params.ReturnWeights
	));

	if (sh.size() > 2)			//не [4096, 3] а [800,800,3]
	{
		if (all_ret.Outputs1.DispMapLE.numel() != 0)
			all_ret.Outputs1.DispMapLE = torch::reshape(all_ret.Outputs1.DispMapLE, { sh[0], sh[1] });	//[640000] -> [800,800]
		if (all_ret.Outputs0.DispMapLE.numel() != 0)
			all_ret.Outputs0.DispMapLE = torch::reshape(all_ret.Outputs0.DispMapLE, { sh[0], sh[1] });
		if (all_ret.Outputs1.DepthMapLE.numel() != 0)
			all_ret.Outputs1.DepthMapLE = torch::reshape(all_ret.Outputs1.DepthMapLE, { sh[0], sh[1] });
		if (all_ret.Outputs0.DepthMapLE.numel() != 0)
			all_ret.Outputs0.DepthMapLE = torch::reshape(all_ret.Outputs0.DepthMapLE, { sh[0], sh[1] });

		if (all_ret.Outputs1.RenderedLangEmbedding.numel() != 0)
			all_ret.Outputs1.RenderedLangEmbedding = torch::reshape(all_ret.Outputs1.RenderedLangEmbedding, { sh[0], sh[1], Lerf->GetLangEmbedDim() });
		if (all_ret.Outputs0.RenderedLangEmbedding.numel() != 0)
			all_ret.Outputs0.RenderedLangEmbedding = torch::reshape(all_ret.Outputs0.RenderedLangEmbedding, { sh[0], sh[1], Lerf->GetLangEmbedDim() });

		if (all_ret.Outputs1.Relevancy.numel() != 0)
			all_ret.Outputs1.Relevancy = torch::reshape(all_ret.Outputs1.Relevancy, { sh[0], sh[1], 2 });
		if (all_ret.Outputs0.Relevancy.numel() != 0)
			all_ret.Outputs0.Relevancy = torch::reshape(all_ret.Outputs0.Relevancy, { sh[0], sh[1], 2 });
	} else {
		if (all_ret.Outputs1.RenderedLangEmbedding.numel() != 0)
			all_ret.Outputs1.RenderedLangEmbedding = torch::reshape(all_ret.Outputs1.RenderedLangEmbedding, { sh[0], Lerf->GetLangEmbedDim() });
		if (all_ret.Outputs0.RenderedLangEmbedding.numel() != 0)
			all_ret.Outputs0.RenderedLangEmbedding = torch::reshape(all_ret.Outputs0.RenderedLangEmbedding, { sh[0], Lerf->GetLangEmbedDim() });

		if (all_ret.Outputs1.Relevancy.numel() != 0)
			all_ret.Outputs1.Relevancy = torch::reshape(all_ret.Outputs1.Relevancy, { sh[0], 2 });
		if (all_ret.Outputs0.Relevancy.numel() != 0)
			all_ret.Outputs0.Relevancy = torch::reshape(all_ret.Outputs0.Relevancy, { sh[0], 2 });
	}

	return all_ret;
}