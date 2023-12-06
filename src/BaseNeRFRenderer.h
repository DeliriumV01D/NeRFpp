#pragma once

#include "NeRF.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

///Rename?
struct Outputs {
	torch::Tensor RGBMap,			///[num_rays, 3] .Estimated RGB color of a ray.
		DispMap,		///[num_rays] .Disparity map.Inverse of depth map.
		AccMap,			///[num_rays] .Sum of weights along each ray.
		Weights,		///[num_rays, num_samples] .Weights assigned to each sampled color.
		DepthMap,		///[num_rays] .Estimated distance to object.
		Normals,		///Calculated normals  [num_rays, num_samples, 3]
		PredNormals,///Predicted normals  [num_rays, num_samples, 3]
		RenderedNormals,		///Rendered calculated normals [num_rays, 3]
		RenderedPredNormals;///Rendered predicted normals [num_rays, 3]
};

struct RenderResult
{
	Outputs Outputs1,		///Estimated RGB color, disparity map, accumulated opacity along each ray.Comes from fine model.
		Outputs0;					///Estimated RGB color, disparity map, accumulated opacity along each ray.Output for coarse model.
	torch::Tensor Raw,	///[num_rays, num_samples, 4] .Raw predictions from model.
		ZStd;							///Standard deviation of distances along ray for each sample.
};

inline torch::Tensor CVMatToTorchTensor(cv::Mat img, const bool perm = false)
{
	auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
	if (perm)
		tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image.unsqueeze_(0);
	tensor_image = tensor_image.toType(c10::kFloat).div(255);
	return tensor_image;
}

inline cv::Mat TorchTensorToCVMat(torch::Tensor tensor_image, const bool perm = false)
{
	auto t = tensor_image.detach().squeeze().cpu();
	if (perm)
		t = t.permute({ 1, 2, 0 });
	t = t.mul(255).clamp(0, 255).to(torch::kU8);
	cv::Mat result_img;
	cv::Mat(t.size(0), t.size(1), CV_MAKETYPE(CV_8U, t.sizes().size() >= 3 ? t.size(2) : 1), t.data_ptr()).copyTo(result_img);
	return result_img;
}

///Prepares inputs and applies network 'fn'.
template <class TEmbedder, class TEmbedDirs, class TNeRF>
inline torch::Tensor RunNetwork(
	torch::Tensor inputs,
	torch::Tensor view_dirs,			///defined if use_view_dirs
	TNeRF fn,
	TEmbedder embed_fn,
	TEmbedDirs embeddirs_fn,
	const bool calculate_normals
) {
	//можно попробовать научить работать эмбедер с батчами чтобы не плющить тензоры?
	torch::Tensor inputs_flat = torch::reshape(inputs, { -1, inputs.sizes().back()/*[-1]*/ });  //[1024, 256, 3] -> [262144, 3]
	inputs_flat.set_requires_grad(calculate_normals);
	auto [embedded, keep_mask] = embed_fn->forward(inputs_flat); 

	if (view_dirs.defined() && view_dirs.numel() != 0)
	{
		torch::Tensor input_dirs = view_dirs.index({ torch::indexing::Slice(), torch::indexing::None }).expand(inputs.sizes());
		torch::Tensor input_dirs_flat = torch::reshape(input_dirs, { -1, input_dirs.sizes().back()/*[-1]*/ });
		auto [embedded_dirs, _] = embeddirs_fn(input_dirs_flat);
		embedded = torch::cat({ embedded, embedded_dirs }, -1);
	}

	torch::Tensor outputs_flat = fn->forward(embedded);

	//set sigma to 0 for invalid points
	if (keep_mask.defined() && keep_mask.numel() != 0)
		outputs_flat = outputs_flat.index_put_({ ~keep_mask, -1 }, 0);

	//Calculated normals, 	//inputs_flat.set_requires_grad(true);
	if (calculate_normals)
	{
		auto density_before_activation = outputs_flat.index({ "...", 3 }).unsqueeze(-1);		//    извлекает значения (sigma) из четвертого столбца raw
		/*density_before_activation.set_requires_grad(true);*/
		auto grad_outputs = torch::ones_like(density_before_activation);
		torch::Tensor calculated_normals = torch::autograd::grad(
			{ density_before_activation },
			{ inputs_flat },
			{ grad_outputs },
			true
		)[0];
		//Берем направление обратное градиенту
		calculated_normals = - torch::nn::functional::normalize(calculated_normals, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
		outputs_flat = torch::cat({ outputs_flat, calculated_normals }, -1);
	}

	std::vector<int64_t> sz = inputs.sizes().vec();
	sz.pop_back();
	sz.push_back(outputs_flat.sizes().back());
	torch::Tensor outputs = torch::reshape(outputs_flat, sz);  //list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]  //[262144, 5] -> [1024, 256, 5]
	return outputs;
}

/////Calculate normals along the ray.
//inline torch::Tensor RenderNormals(		///[bs, 3]
//	const torch::Tensor normals, ///[bs, num_samples, 3]
//	const torch::Tensor weights, ///[bs, num_samples, 1]
//	const torch::Tensor rays_d,	///rays_d : [bs, 3] .Direction of each ray.
//	const bool normalize = true
//) {
//	auto n = (weights * normals).sum(-2, false);
//	auto cos_n_rd  = ((n / torch::norm(n, 2/*L2*/, -1/*dim*/, true)) * (rays_d / torch::norm(rays_d, 2/*L2*/, -1/*dim*/, true))).sum(-1, false);
//	//auto cos_n_rd  = torch::mm((n / torch::norm(n, 2/*L2*/, -1/*dim*/, true)), (rays_d / torch::norm(rays_d, 2/*L2*/, -1/*dim*/, true)).t());
//	//if (normalize)
//	//	n = torch::nn::functional::normalize(n, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
//	std::cout<<"rays_d: "<<rays_d.sizes()<<std::endl;
//	std::cout<<"n: "<<n.sizes()<<std::endl;
//	std::cout<<"cos_n_rd: "<<cos_n_rd.sizes()<<std::endl;
//	return torch::ones_like(n) * cos_n_rd.unsqueeze(-1);
//}

/////Calculate normals along the ray.
//inline torch::Tensor RenderNormals(		///[bs, 3]
//	const torch::Tensor normals, ///[bs, num_samples, 3]
//	const torch::Tensor weights, ///[bs, num_samples, 1]
//	const bool normalize = true
//) {
//	auto n = (weights * normals).sum(-2, false);
//	if (normalize)
//		n = torch::nn::functional::normalize(n, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));
//	return n;
//}

///Calculate normals along the ray.
inline torch::Tensor RenderNormals(		///[bs, 3]
	const torch::Tensor normals, ///[bs, num_samples, 3]
	const torch::Tensor weights, ///[bs, num_samples, 1]
	const bool normalize = true
) {
	auto bg = torch::zeros_like(normals);
	auto alpha = torch::cat({weights, weights, weights}, -1);

	auto n = (alpha * (normals + 1) / 2 + (1. - alpha) * bg).sum(-2, false);

	//if (normalize)
	//	n = torch::nn::functional::normalize(n, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8));

	//if (keep_alpha)
	//	normals = torch::cat({weights, normals}, -1);
	return n;
}

///Calculate shading for normals.
inline torch::Tensor NormalsShader(const torch::Tensor normals) ///[bs, num_samples, 3]
{
	return normals;//(normals + 1) / 2;
}

///Transforms model's predictions to semantically meaningful values.
inline Outputs RawToOutputs(
	torch::Tensor raw,			///raw : [num_rays, num_samples along ray, 4+3+(3)] .Prediction from model.
	torch::Tensor z_vals,		///z_vals : [num_rays, num_samples along ray] .Integration time.
	torch::Tensor rays_d,		///rays_d : [num_rays, 3] .Direction of each ray.
	const float raw_noise_std = 0.f,
	const bool white_bkgr = false,
	const bool calculate_normals = false,
	const bool use_pred_normal = false
) {
	torch::Device device = raw.device();
	Outputs result;
	auto raw2alpha = [](torch::Tensor raw, torch::Tensor dists) {return -torch::exp(-torch::relu(raw) * dists) + 1.f; };

	auto dists = z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) - z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });
	dists = torch::cat({ dists, (torch::ones(1, torch::kFloat32) * 1e10).expand(dists.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }).sizes()).to(device)}, -1);  // [N_rays, N_samples]
	dists = dists * torch::norm(rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }), 2/*L2*/, /*dim*/-1);
	if (!dists.requires_grad())
		dists.set_requires_grad(true);

	auto rgb = torch::sigmoid(raw.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) }));		//[N_rays, N_samples, 3] извлекает значения из первых трех столбцов тензора raw
	torch::Tensor noise = torch::zeros(raw.index({ "...", 3 }).sizes(), torch::kFloat32).to(device);
	if (raw_noise_std > 0.f)
	{
		noise = (torch::randn(raw.index({ "...", 3 }).sizes()) * raw_noise_std).to(device);
	}
	auto density_before_activation = raw.index({ "...", 3 });		//    извлекает значения (sigma) из четвертого столбца raw
	torch::Tensor alpha = raw2alpha(density_before_activation + noise, dists);  //[N_rays, N_samples]
	result.Weights = alpha * torch::cumprod(
		torch::cat({ torch::ones({alpha.sizes()[0], 1}).to(device), -alpha + 1.f + 1e-10f }, -1),
		-1
	).index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1) });
	result.RGBMap = torch::sum(result.Weights.index({ "...", torch::indexing::None }) * rgb, -2);  //[N_rays, 3]

	result.DepthMap = torch::sum(result.Weights * z_vals, -1) / torch::sum(result.Weights, -1);
	result.DispMap = 1. / torch::max(1e-10 * torch::ones_like(result.DepthMap), result.DepthMap);
	result.AccMap = torch::sum(result.Weights, -1);

	if (white_bkgr)
		result.RGBMap = result.RGBMap + (1. - result.AccMap.index({ "...", torch::indexing::None }));

	//Вычисленные нормали коннектятся в конце. Если есть предсказанные нормали то есть и вычисленные но не наоборот
	
	//Calculated normals
	if (calculate_normals)
	{
		if (use_pred_normal)
		{
			result.Normals = torch::tanh(raw.index({ "...",  torch::indexing::Slice(6, 9) }));		//извлекает значения из 7-9 столбцов raw
		} else {
			result.Normals = torch::tanh(raw.index({ "...",  torch::indexing::Slice(3, 6) }));		//извлекает значения из 4-6 столбцов raw //Уже отнормировано в месте вычисления
		}
		result.RenderedNormals = RenderNormals(result.Normals, result.Weights.unsqueeze(-1));
		result.RenderedNormals = NormalsShader(result.RenderedNormals);
	}

	//Predicted normals
	if (use_pred_normal)
	{
		result.PredNormals = torch::tanh(raw.index({ "...",  torch::indexing::Slice(3, 6) }));		//извлекает значения из 4-6 столбцов raw
		//result.PredNormals = torch::nn::functional::normalize(result.PredNormals, torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-8)); //Уже отнормировано в месте вычисления
		result.RenderedPredNormals = RenderNormals(result.PredNormals, result.Weights.unsqueeze(-1));
		result.RenderedPredNormals = NormalsShader(result.RenderedPredNormals);
	}

	return result;
}

///Volumetric rendering.
template <class TEmbedder, class TEmbedDirs, class TNeRF>
inline RenderResult RenderRays(
	torch::Tensor ray_batch,		///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
	TNeRF nerf,
	TEmbedder embed_fn,
	TEmbedDirs embeddirs_fn,
	TNeRF network_fine,											///"fine" network with same spec as network_fn.
	const int n_samples,
	const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance = 0,							///Number of additional times to sample along each ray.
	const bool white_bkgr = false,					///If True, assume a white background.
	const float raw_noise_std = 0.,
	const bool calculate_normals = false,
	const bool use_pred_normal = false
) {
	torch::Device device = ray_batch.device();		//Передать параметром??
	RenderResult result;
	///!!!Можно просто передавать структурку не парсить тензор
	int nrays = ray_batch.sizes()[0];
	auto rays_o = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3) });			//[N_rays, 3]		Origins
	auto rays_d = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(3, 6) });			//[N_rays, 3]		Directions
	torch::Tensor viewdirs;
	if (ray_batch.sizes().back() > 8)
		viewdirs = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(-3, torch::indexing::None) });
	auto bounds = torch::reshape(ray_batch.index({ "...", torch::indexing::Slice(6, 8) }), { -1, 1, 2 });
	auto near = bounds.index({ "...", 0 });
	auto far = bounds.index({ "...", 1 }); //[-1, 1

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
	torch::Tensor raw = RunNetwork(pts, viewdirs, /*torch::Tensor(),*/ nerf, embed_fn, embeddirs_fn, (n_importance <= 0) && calculate_normals);
	result.Outputs1 = RawToOutputs(raw, z_vals, rays_d, raw_noise_std, white_bkgr, (n_importance <= 0) && calculate_normals, (n_importance <= 0) && use_pred_normal);

	if (n_importance > 0)
	{
		result.Outputs0 = result.Outputs1;
		auto z_vals_mid = .5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto z_samples = SamplePDF(z_vals_mid, result.Outputs1.Weights.index({ "...", torch::indexing::Slice(1, -1) }), n_importance, perturb == 0.);
		z_samples = z_samples.detach();
		torch::Tensor z_indices;
		std::tie(z_vals, z_indices) = torch::sort(torch::cat({ z_vals, z_samples }, -1), -1);
		pts = rays_o.index({ "...", torch::indexing::None, torch::indexing::Slice() }) + rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }) * z_vals.index({ "...", torch::indexing::Slice(), torch::indexing::None }); // [N_rays, N_samples + N_importance, 3]
		
		if (!network_fine /*!= nullptr*/)
		{
			raw = RunNetwork(pts, viewdirs, nerf, embed_fn, embeddirs_fn, calculate_normals);
		}	else {
			raw = RunNetwork(pts, viewdirs, network_fine, embed_fn, embeddirs_fn, calculate_normals);
		}
		result.Outputs1 = RawToOutputs(raw, z_vals, rays_d, raw_noise_std, white_bkgr, calculate_normals, use_pred_normal);
		result.ZStd = torch::std(z_samples, -1, false);  // [N_rays]
	}

	if (return_raw)
		result.Raw = raw;

	return result;
}


///Render rays in smaller minibatches to save memory
///rays_flat.sizes()[0] должно быть кратно размеру chunk
template <class TEmbedder, class TEmbedDirs, class TNeRF>
inline RenderResult BatchifyRays(
	torch::Tensor rays_flat,			///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
	TNeRF nerf,
	TEmbedder embed_fn,
	TEmbedDirs embeddirs_fn,
	TNeRF network_fine,											///"fine" network with same spec as network_fn.
	const int n_samples,
	const int chunk = 1024 * 32,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
	const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance = 0,							///Number of additional times to sample along each ray.
	const bool white_bkgr = false,					///If True, assume a white background.
	const float raw_noise_std = 0.,
	const bool calculate_normals = false,
	const bool use_pred_normal = false
) {
	RenderResult result;
	std::vector<RenderResult> all_results;
	all_results.reserve(rays_flat.sizes()[0] / chunk);
	for (int i = 0; i < rays_flat.sizes()[0]; i += chunk)
		all_results.push_back(RenderRays(
			rays_flat.index({ torch::indexing::Slice(i, i + chunk) }),
			nerf,
			embed_fn,
			embeddirs_fn,
			network_fine,
			n_samples,
			return_raw,
			lin_disp,
			perturb,
			n_importance,
			white_bkgr,
			raw_noise_std,
			calculate_normals,
			use_pred_normal
		));
	//Слить all_results в один RenderResult используя torch::cat(torch::TensorList - at::ArrayRef ...
	//!!!make this part cleaner and shorter
	std::vector <torch::Tensor> out_rgb_map,
		out_disp_map,
		out_acc_map,
		out_weights,
		out_depth_map,
		out_normals,
		out_pred_normals,
		out_rendered_normals,
		out_rendered_pred_normals,
		out0_rgb_map,
		out0_disp_map,
		out0_acc_map,
		out0_weights,
		out0_depth_map,
		out0_normals,
		out0_pred_normals,
		out0_rendered_normals,
		out0_rendered_pred_normals,
		raw,
		z_std;
	for (auto it : all_results)
	{
		if (it.Outputs1.RGBMap.defined()) out_rgb_map.push_back(it.Outputs1.RGBMap);
		if (it.Outputs1.DispMap.defined()) out_disp_map.push_back(it.Outputs1.DispMap);
		if (it.Outputs1.AccMap.defined()) out_acc_map.push_back(it.Outputs1.AccMap);
		if (it.Outputs1.Weights.defined()) out_weights.push_back(it.Outputs1.Weights);
		if (it.Outputs1.DepthMap.defined()) out_depth_map.push_back(it.Outputs1.DepthMap);
		if (it.Outputs1.Normals.defined()) out_normals.push_back(it.Outputs1.Normals);
		if (it.Outputs1.PredNormals.defined()) out_pred_normals.push_back(it.Outputs1.PredNormals);
		if (it.Outputs1.RenderedNormals.defined()) out_rendered_normals.push_back(it.Outputs1.RenderedNormals);
		if (it.Outputs1.RenderedPredNormals.defined()) out_rendered_pred_normals.push_back(it.Outputs1.RenderedPredNormals);
		if (it.Outputs0.RGBMap.defined()) out0_rgb_map.push_back(it.Outputs0.RGBMap);
		if (it.Outputs0.DispMap.defined()) out0_disp_map.push_back(it.Outputs0.DispMap);
		if (it.Outputs0.AccMap.defined()) out0_acc_map.push_back(it.Outputs0.AccMap);
		if (it.Outputs0.Weights.defined()) out0_weights.push_back(it.Outputs0.Weights);
		if (it.Outputs0.DepthMap.defined()) out0_depth_map.push_back(it.Outputs0.DepthMap);
		if (it.Outputs0.Normals.defined()) out0_normals.push_back(it.Outputs0.Normals);
		if (it.Outputs0.PredNormals.defined()) out0_pred_normals.push_back(it.Outputs0.PredNormals);
		if (it.Outputs0.RenderedNormals.defined()) out0_rendered_normals.push_back(it.Outputs0.RenderedNormals);
		if (it.Outputs0.RenderedPredNormals.defined()) out0_rendered_pred_normals.push_back(it.Outputs0.RenderedPredNormals);
		if (it.Raw.defined()) raw.push_back(it.Raw);
		if (it.ZStd.defined()) z_std.push_back(it.ZStd);
	}
	if (!out_rgb_map.empty()) result.Outputs1.RGBMap = torch::cat(out_rgb_map, 0);
	if (!out_disp_map.empty()) result.Outputs1.DispMap = torch::cat(out_disp_map, 0);
	if (!out_acc_map.empty()) result.Outputs1.AccMap = torch::cat(out_acc_map, 0);
	if (!out_weights.empty()) result.Outputs1.Weights = torch::cat(out_weights, 0);
	if (!out_depth_map.empty()) result.Outputs1.DepthMap = torch::cat(out_depth_map, 0);
	if (!out_normals.empty()) result.Outputs1.Normals = torch::cat(out_normals, 0);
	if (!out_pred_normals.empty()) result.Outputs1.PredNormals = torch::cat(out_pred_normals, 0);
	if (!out_rendered_normals.empty()) result.Outputs1.RenderedNormals = torch::cat(out_rendered_normals, 0);
	if (!out_rendered_pred_normals.empty()) result.Outputs1.RenderedPredNormals = torch::cat(out_rendered_pred_normals, 0);
	if (!out0_rgb_map.empty()) result.Outputs0.RGBMap = torch::cat(out0_rgb_map, 0);
	if (!out0_disp_map.empty()) result.Outputs0.DispMap = torch::cat(out0_disp_map, 0);
	if (!out0_acc_map.empty()) result.Outputs0.AccMap = torch::cat(out0_acc_map, 0);
	if (!out0_weights.empty()) result.Outputs0.Weights = torch::cat(out0_weights, 0);
	if (!out0_depth_map.empty()) result.Outputs0.DepthMap = torch::cat(out0_depth_map, 0);
	if (!out0_normals.empty()) result.Outputs0.Normals = torch::cat(out0_normals, 0);
	if (!out0_pred_normals.empty()) result.Outputs0.PredNormals = torch::cat(out0_pred_normals, 0);
	if (!out0_rendered_normals.empty()) result.Outputs0.RenderedNormals = torch::cat(out0_rendered_normals, 0);
	if (!out0_rendered_pred_normals.empty()) result.Outputs0.RenderedPredNormals = torch::cat(out0_rendered_pred_normals, 0);
	if (!raw.empty()) result.Raw = torch::cat(raw, 0);
	if (!z_std.empty()) result.ZStd = torch::cat(z_std, 0);

	return result;
}

///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
template <class TEmbedder, class TEmbedDirs, class TNeRF>
inline RenderResult Render(
	const int h,					///Height of image in pixels.
	const int w,					///Width of image in pixels.
	torch::Tensor k,			///Сamera calibration
	TNeRF nerf,
	TEmbedder embed_fn,
	TEmbedDirs embeddirs_fn,
	TNeRF network_fine,											///"fine" network with same spec as network_fn.
	const int n_samples,
	const int chunk = 1024 * 32,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
	const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance = 0,							///Number of additional times to sample along each ray.
	const bool white_bkgr = false,					///If True, assume a white background.
	const float raw_noise_std = 0.,
	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
	const bool ndc = true,						///If True, represent ray origin, direction in NDC coordinates.
	const float near = 0.,						///float or array of shape[batch_size].Nearest distance for a ray.
	const float far = 1.,							///float or array of shape[batch_size].Farthest distance for a ray.
	const bool use_viewdirs = false,	///If True, use viewing direction of a point in space in model.
	const bool calculate_normals = false,
	const bool use_pred_normal = false,	///whether to use predicted normals
	torch::Tensor c2w_staticcam = torch::Tensor()			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
) {
	torch::Tensor rays_o, rays_d;
	if (c2w.defined() && c2w.numel() != 0)
	{
		//special case to render full image
		std::tie(rays_o, rays_d) = GetRays(h, w, k, c2w);
	}
	else {
		//use provided ray batch
		std::tie(rays_o, rays_d) = rays;
	}

	torch::Tensor viewdirs;
	if (use_viewdirs)
	{
		//provide ray directions as input
		viewdirs = rays_d;
		if (c2w_staticcam.defined() && c2w_staticcam.numel() != 0)
		{
			//special case to visualize effect of viewdirs
			std::tie(rays_o, rays_d) = GetRays(h, w, k, c2w_staticcam);
		}
		viewdirs = viewdirs / torch::norm(viewdirs, 2/*L2*/, -1/*dim*/, true);
		viewdirs = torch::reshape(viewdirs, { -1, 3 });//.float();
	}

	auto sh = rays_d.sizes();			//[..., 3]
	if (ndc)
	{
		//for forward facing scenes
		std::tie(rays_o, rays_d) = NDCRays(h, w, k[0][0].item<float>()/*focal*/, 1.f, rays_o, rays_d);
	}

	//Create ray batch
	rays_o = torch::reshape(rays_o, { -1, 3 });//.float()
	rays_d = torch::reshape(rays_d, { -1, 3 });//.float()

	auto near_ = near * torch::ones_like(rays_d.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }));
	auto far_ = far * torch::ones_like(rays_d.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }));
	auto rays_ = torch::cat({ rays_o, rays_d, near_, far_ }, -1);
	
	if (use_viewdirs)
		rays_ = torch::cat({ rays_, viewdirs }, -1);

	//Renderand reshape
	RenderResult all_ret = BatchifyRays(
		rays_, nerf, embed_fn, embeddirs_fn, network_fine, n_samples, chunk, return_raw, lin_disp, perturb, n_importance, white_bkgr, raw_noise_std, calculate_normals, use_pred_normal
	);

	if (all_ret.Outputs1.RGBMap.numel() != 0)
		all_ret.Outputs1.RGBMap = torch::reshape(all_ret.Outputs1.RGBMap, sh);		//[640000, 3] -> [800, 800, 3]
	if (all_ret.Outputs0.RGBMap.numel() != 0)
		all_ret.Outputs0.RGBMap = torch::reshape(all_ret.Outputs0.RGBMap, sh);
	if (sh.size() > 2)			//не [4096, 3] а [800,800,3]
	{
		if (all_ret.Outputs1.DispMap.numel() != 0)
			all_ret.Outputs1.DispMap = torch::reshape(all_ret.Outputs1.DispMap, { sh[0], sh[1] });	//[640000] -> [800,800]
		if (all_ret.Outputs0.DispMap.numel() != 0)
			all_ret.Outputs0.DispMap = torch::reshape(all_ret.Outputs0.DispMap, { sh[0], sh[1] });
		if (all_ret.Outputs1.DepthMap.numel() != 0)
			all_ret.Outputs1.DepthMap = torch::reshape(all_ret.Outputs1.DepthMap, { sh[0], sh[1] });
		if (all_ret.Outputs0.DepthMap.numel() != 0)
			all_ret.Outputs0.DepthMap = torch::reshape(all_ret.Outputs0.DepthMap, { sh[0], sh[1] });
	}


	//if (all_ret.Outputs1.Normals.numel() != 0)
	//	all_ret.Outputs1.Normals = torch::reshape(all_ret.Outputs1.Normals, sh);
	//if (all_ret.Outputs0.Normals.numel() != 0)
	//	all_ret.Outputs0.Normals = torch::reshape(all_ret.Outputs0.Normals, sh);

	//if (all_ret.Outputs1.PredNormals.numel() != 0)
	//	all_ret.Outputs1.PredNormals = torch::reshape(all_ret.Outputs1.PredNormals, sh);
	//if (all_ret.Outputs0.PredNormals.numel() != 0)
	//	all_ret.Outputs0.PredNormals = torch::reshape(all_ret.Outputs0.PredNormals, sh);

	if (calculate_normals)
	{
		if (all_ret.Outputs1.RenderedNormals.numel() != 0)
			all_ret.Outputs1.RenderedNormals = torch::reshape(all_ret.Outputs1.RenderedNormals, sh);
		if (all_ret.Outputs0.RenderedNormals.numel() != 0)
			all_ret.Outputs0.RenderedNormals = torch::reshape(all_ret.Outputs0.RenderedNormals, sh);
	}

	if (use_pred_normal)
	{
		if (all_ret.Outputs1.RenderedPredNormals.numel() != 0)
			all_ret.Outputs1.RenderedPredNormals = torch::reshape(all_ret.Outputs1.RenderedPredNormals, sh);
		if (all_ret.Outputs0.RenderedPredNormals.numel() != 0)
			all_ret.Outputs0.RenderedPredNormals = torch::reshape(all_ret.Outputs0.RenderedPredNormals, sh);
	}

	return all_ret;
}