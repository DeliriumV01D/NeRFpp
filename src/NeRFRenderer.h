#pragma once

#include "RayUtils.h"
#include "Sampler.h"
#include "CustomOps.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

///
struct NeRFRendererOutputs {
	torch::Tensor RGBMap,			///[num_rays, 3] .Estimated RGB color of a ray.
		DispMap,		///[num_rays] .Disparity map.Inverse of depth map.
		AccMap,			///[num_rays] .Sum of weights along each ray.
		Weights,		///[num_rays, num_samples] .Weights assigned to each sampled color.
		DepthMap;		///[num_rays] .Estimated distance to object.
};

struct NeRFRenderResult
{
	NeRFRendererOutputs Outputs;		///Estimated RGB color, disparity map, accumulated opacity along each ray.Comes from model.
	torch::Tensor Raw;	///[num_rays, num_samples, 4] .Raw predictions from model.
	float Near,
		Far;
};

struct NeRFRenderParams {
	int NSamples{64};								///Samples along ray
	int NImportance{192};						///Number of additional times to sample along each ray.
	int Chunk{1024 * 32};						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
	bool ReturnRaw{false};					///If True, include model's raw, unprocessed predictions.
	bool LinDisp{false};						///If True, sample linearly in inverse depth rather than in depth.
	float Perturb{0.f};							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	bool WhiteBkgr{false};					///If True, assume a white background.
	float RawNoiseStd{0.};			///Локальная регуляризация плотности (выход) помогает избежать артефактов типа "облаков" затухает за n_iters / 3 итераций
	bool Ndc{true};							///If True, represent ray origin, direction in NDC coordinates.
	bool UseViewdirs{false};		///If True, use viewing direction of a point in space in model.
	bool ReturnWeights{false};
	bool ThinRay{false};
	float RenderFactor{0};
	torch::Tensor BoundingBox{torch::Tensor()};
	float StochasticPreconditioningAlpha{0};	///добавляет шум к входу сети (координатам точек). Уменьшает чувствительность к инициализации. Помогает избежать "плавающих" артефактов
};

inline torch::Tensor CVMatToTorchTensor(cv::Mat img, const bool perm = false)
{
	if (!img.isContinuous())
		img = img.clone();
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
	t = t.contiguous();
	cv::Mat result_img;
	cv::Mat((int)t.size(0), (int)t.size(1), (int)CV_MAKETYPE(CV_8U, t.sizes().size() >= 3 ? t.size(2) : 1), t.data_ptr()).copyTo(result_img);
	return result_img;
}

///c2w <-> w2c, w2c <-> c2w
static torch::Tensor C2W2C(torch::Tensor in)
{
	torch::Tensor out = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(in.device()));
	//Извлекаем компоненты
	torch::Tensor R = in.index({ torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3) });
	torch::Tensor t = in.index({ torch::indexing::Slice(0, 3), 3 });
	//Вычисляем обратные компоненты
	torch::Tensor R_inv = torch::linalg_inv(R);		//R.transpose(0, 1); //Для ортогональной матрицы обратная = транспонированная
	torch::Tensor t_inv = -torch::matmul(R_inv, t);
	//Собираем обратную матрицу
	out.index_put_({ torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3) }, R_inv);
	out.index_put_({ torch::indexing::Slice(0, 3), 3 }, t_inv);
	//out[3][3] = 1;
	return out;
}

///
template <class TEmbedder, class TEmbedDirs, class TNeRF>
class NeRFRenderer {
protected:
	TEmbedder EmbedFn;
	TEmbedDirs EmbeddirsFn;
	TNeRF NeRF;

	///Prepares inputs and applies network 'fn'.
	virtual torch::Tensor RunNetwork(torch::Tensor inputs, torch::Tensor view_dirs, TNeRF fn, TEmbedder embed_fn, TEmbedDirs embeddirs_fn);
	
	///Transforms model's predictions to semantically meaningful values.
	virtual NeRFRendererOutputs RawToOutputs(
		torch::Tensor raw,			///raw : [num_rays, num_samples along ray, 4+3+(3)] .Prediction from model.
		torch::Tensor cone_angle,
		torch::Tensor z_vals,		///z_vals : [num_rays, num_samples along ray] .Integration time.
		torch::Tensor rays_d,		///rays_d : [num_rays, 3] .Direction of each ray.
		const float raw_noise_std = 0.f,
		const bool white_bkgr = false
	);
public:
	NeRFRenderer(
		TEmbedder embed_fn,
		TEmbedDirs embeddirs_fn,
		TNeRF nerf
	) : NeRF(nerf), EmbedFn(embed_fn), EmbeddirsFn(embeddirs_fn){};
	virtual ~NeRFRenderer(){};

	///Volumetric rendering.
	virtual NeRFRenderResult RenderRays(
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
		torch::Tensor bounding_box  = torch::Tensor(),
		const bool return_weights = true
	);

	///Render rays in smaller minibatches to save memory
	///rays_flat.sizes()[0] должно быть кратно размеру chunk
	virtual NeRFRenderResult BatchifyRays(
		torch::Tensor rays_flat,			///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
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
		torch::Tensor bounding_box  = torch::Tensor(),
		const bool return_weights = true
	);

	///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
	virtual NeRFRenderResult Render(
		const int h,					///Height of image in pixels.
		const int w,					///Width of image in pixels.
		torch::Tensor k,			///Сamera calibration
		const NeRFRenderParams &render_params,
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		torch::Tensor c2w_staticcam = torch::Tensor()			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
	);
};			//NeRFRenderer


///Prepares inputs and applies network 'fn'.
template <class TEmbedder, class TEmbedDirs, class TNeRF>
torch::Tensor NeRFRenderer<TEmbedder, TEmbedDirs, TNeRF> ::RunNetwork(
	torch::Tensor inputs,
	torch::Tensor view_dirs,			///defined if use_view_dirs
	TNeRF fn,
	TEmbedder embed_fn,
	TEmbedDirs embeddirs_fn
) {
	//можно попробовать научить работать эмбедер с батчами чтобы не плющить тензоры?
	torch::Tensor inputs_flat = inputs.view({ -1, inputs.sizes().back()/*[-1]*/ });  //[1024, 256, 3] -> [262144, 3]
	inputs_flat.set_requires_grad(false);
	
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
		outputs_flat.index_put_({ ~keep_mask, -1 }, 0);

	std::vector<int64_t> sz = inputs.sizes().vec();
	sz.pop_back();
	sz.push_back(outputs_flat.sizes().back());
	return outputs_flat.view(sz);
}


///Transforms model's predictions to semantically meaningful values.
template <class TEmbedder, class TEmbedDirs, class TNeRF>
NeRFRendererOutputs NeRFRenderer<TEmbedder, TEmbedDirs, TNeRF> :: RawToOutputs(
	torch::Tensor raw,			///raw : [num_rays, num_samples along ray, 4+3+(3)] .Prediction from model.
	torch::Tensor cone_angle,
	torch::Tensor z_vals,		///z_vals : [num_rays, num_samples along ray] .Integration time.
	torch::Tensor rays_d,		///rays_d : [num_rays, 3] .Direction of each ray.
	const float raw_noise_std /*= 0.f*/,
	const bool white_bkgr /*= false*/
) {
	torch::Device device = raw.device();
	NeRFRendererOutputs result;
	
	//Функция перевода плотности в альфа с учетом объема сегмента конуса
	auto raw2alpha_with_volume = [](torch::Tensor raw, torch::Tensor dists, torch::Tensor radii)
	{
		//if (radii.defined())
		//{
		//	//Альфа зависит от объема усеченного конуса между двумя сечениямм, а не от длины
		//	auto r_start = radii.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });  // [N_rays, N_samples-1]
		//	auto r_end = radii.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });     // [N_rays, N_samples-1]
		//	//dists для внутренних сегментов (кроме последнего)
		//	auto dists_segments = dists.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });  // [N_rays, N_samples-1]
		//	//Объем усеченного конуса: v = (π/3) * h * (r₁² + r₁*r₂ + r₂²)
		//	//auto volume_segments = (M_PI / 3.0f) * dists_segments * (r_start.square() + r_start * r_end + r_end.square());  // [N_rays, N_samples-1]
		//	auto safe_volume = [](torch::Tensor h, torch::Tensor r1, torch::Tensor r2, float eps = 1e-10f) {
		//		r1 = torch::clamp_min(r1, eps);
		//		r2 = torch::clamp_min(r2, eps);
		//		h = torch::clamp_min(h, eps);
		//		return (M_PI / 3.0f) * h * (r1.square() + r1 * r2 + r2.square());
		//	};
		//	auto volume_segments = safe_volume(dists_segments, r_start, r_end);			
		//	auto volume = torch::cat({ volume_segments, volume_segments.index({ "...", torch::indexing::Slice(-1, torch::indexing::None) }) }, -1);  // [N_rays, N_samples]
		//	//return - torch::autograd::TruncExp::apply(- torch::relu(raw) * volume)[0] + 1.f;
		//	return 1.0f - torch::exp(-torch::clamp(torch::relu(raw) * torch::clamp(volume, 1e-10f, 1e10f), 0.0f, 80.0f));
		//} else {
			//Стандартный NeRF (бесконечно тонкий луч)
			return - torch::autograd::TruncExp::apply(- torch::relu(raw) * dists)[0] + 1.f;
		//}
	};

	//Вычисление расстояний между сэмплами
	auto dists = z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) - z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) });
	dists = torch::cat({ dists, (torch::ones(1, torch::kFloat32) * 1e10).expand(dists.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }).sizes()).to(device)}, -1);  // [N_rays, N_samples]
	dists = dists * torch::norm(rays_d.index({ "...", torch::indexing::None, torch::indexing::Slice() }), 2/*L2*/, /*dim*/-1);
	if (!dists.requires_grad())
		dists.set_requires_grad(true);

	//Вычисление радиусов конуса для каждой точки. Радиус конуса линейно растет с расстоянием
	torch::Tensor cone_radii;
	if (cone_angle.defined() && cone_angle.numel()) //if (cone_angle > 0.0f)
		cone_radii = cone_angle * z_vals;  // [N_rays, N_samples]

	auto rgb = torch::sigmoid(raw.index({ "...", torch::indexing::Slice(torch::indexing::None, 3) }));		//[N_rays, N_samples, 3] извлекает значения из первых трех столбцов тензора raw
	
	auto density_before_activation = raw.index({ "...", 3 });		//    извлекает значения (sigma) из четвертого столбца raw
	if (raw_noise_std > 0.f)
		density_before_activation = density_before_activation + torch::randn_like(density_before_activation) * raw_noise_std;

	torch::Tensor alpha = raw2alpha_with_volume(density_before_activation, dists, cone_radii);  //[N_rays, N_samples]
	//result.Weights = alpha * torch::cumprod(
	//	torch::cat({ torch::ones({alpha.sizes()[0]/*nrays*/, 1}).to(device), -alpha + 1.f + 1e-10f }, -1), -1
	//).index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1) });
	auto compute_weights_in_log_space = [](torch::Tensor alpha, torch::Device device)
	{
		//Вычисляем log(1 - alpha) с защитой от log(0)
		auto log_transmittance = torch::cat({
				torch::zeros({alpha.size(0)/*nrays*/, 1}, torch::TensorOptions().device(device)),
				torch::cumsum(torch::log(torch::clamp_min(1.0f - alpha, 1e-10f)), -1)
			}, -1).index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1) });
		auto weights = alpha * torch::autograd::TruncExp::apply(log_transmittance)[0];
		return weights;
	};
	result.Weights = compute_weights_in_log_space(alpha, device);
	result.RGBMap = torch::sum(result.Weights.index({ "...", torch::indexing::None }) * rgb, -2);  //[N_rays, 3]
	result.DepthMap = torch::sum(result.Weights * z_vals, -1) / torch::clamp_min(torch::sum(result.Weights, -1), 1e-10f);
	result.DispMap = 1. / torch::max(1e-10 * torch::ones_like(result.DepthMap), result.DepthMap);
	result.AccMap = torch::sum(result.Weights, -1);

	if (white_bkgr)
		result.RGBMap = result.RGBMap + (1. - result.AccMap.index({ "...", torch::indexing::None }));

	int cur_pos = 4;

	return result;
}

//Обрабатываем границы отражением
inline torch::Tensor ReflectBoundary(torch::Tensor pts, torch::Tensor min_bound, torch::Tensor max_bound)
{
	//Приводим точки к диапазону [0,1]^3
	auto normalized_pts = (pts - min_bound) / (max_bound - min_bound);

	//Функция отражения для одного измерения
	auto reflect_dim = [](torch::Tensor x) {
		x = torch::fmod(x, 2.0f);
		auto mask = x > 1.0f;
		return torch::where(mask, 2.0f - x, x);
	};

	//Применяем отражение по всем измерениям
	auto x = reflect_dim(normalized_pts.index({ "...", 0 }));
	auto y = reflect_dim(normalized_pts.index({ "...", 1 }));
	auto z = reflect_dim(normalized_pts.index({ "...", 2 }));

	auto reflected = torch::stack({ x, y, z }, -1);
	return reflected * (max_bound - min_bound) + min_bound;
}

///Сэмплим в расходящимся конусе
inline torch::Tensor TangentScatter(torch::Tensor pts, torch::Tensor z_vals, torch::Tensor cone_angle, torch::Tensor rays_d, torch::Device device, torch::Tensor bounding_box)
{
	if (cone_angle.defined() && cone_angle.numel())//if (cone_angle > 0.0f)
	{
		auto safe_normalize = [](torch::Tensor v, float eps = 1e-8f) {auto norm = torch::norm(v, 2, -1, true).clamp_min(eps); return v / norm; };

		//Вычисление радиусов конуса для каждой точки. Радиус конуса линейно растет с расстоянием
		torch::Tensor cone_radii = cone_angle * z_vals;  // [N_rays, N_samples]
		//Добавление случайного смещения в пределах конуса. Это имитирует интегрирование по объему конуса
		//Генерируем случайные направления, перпендикулярные лучу
		auto rays_d_norm = safe_normalize(rays_d);
		//Создаем ортогональный базис
		auto n_rays = rays_d_norm.size(0);
		auto n_samples = z_vals.size(1);

		//Выбираем вспомогательный вектор, гарантированно не коллинеарный лучу
		auto abs_d = torch::abs(rays_d_norm);
		auto mask_x = (abs_d.index({ "...", 0 }) < abs_d.index({ "...", 1 })) & (abs_d.index({ "...", 0 }) < abs_d.index({ "...", 2 }));
		auto mask_y = (abs_d.index({ "...", 1 }) < abs_d.index({ "...", 0 })) & (abs_d.index({ "...", 1 }) < abs_d.index({ "...", 2 }));
		auto mask_z = torch::logical_not(torch::logical_or(mask_x, mask_y));  // Все остальные случаи
		mask_x = mask_x.unsqueeze(-1).expand({ n_rays, 3 });
		mask_y = mask_y.unsqueeze(-1).expand({ n_rays, 3 });
		mask_z = mask_z.unsqueeze(-1).expand({ n_rays, 3 });
		auto candidate_x = torch::tensor({ 1.0f, 0.0f, 0.0f }, device = device).unsqueeze(0).expand({ n_rays, 3 });
		auto candidate_y = torch::tensor({ 0.0f, 1.0f, 0.0f }, device = device).unsqueeze(0).expand({ n_rays, 3 });
		auto candidate_z = torch::tensor({ 0.0f, 0.0f, 1.0f }, device = device).unsqueeze(0).expand({ n_rays, 3 });
		auto up_candidate = torch::where(mask_x, candidate_x, torch::where(mask_y, candidate_y, candidate_z));
		//Перпендикулярные векторы
		auto tangent = torch::cross(rays_d_norm, up_candidate, /*dim=*/-1);
		tangent = safe_normalize(tangent);
		auto bitangent = torch::cross(rays_d_norm, tangent, /*dim=*/-1);
		bitangent = safe_normalize(bitangent);

		//Случайные смещения в плоскости
		//Генерируем равномерное распределение по радиусу (используя sqrt для равномерности по площади) и углу
		auto r = torch::sqrt(torch::clamp(torch::rand({ n_rays, n_samples, 1 }, device = device), 1e-8f, 1.0f - 1e-8f));
		auto theta = torch::fmod(torch::rand({ n_rays, n_samples, 1 }, device = device) * 2.0f * M_PI, 2.0f * M_PI);
		auto offset_x = r * torch::cos(theta);
		auto offset_y = r * torch::sin(theta);

		auto tangent_exp = tangent.unsqueeze(1);      // [n_rays, 1, 3]
		auto bitangent_exp = bitangent.unsqueeze(1);  // [n_rays, 1, 3]
		auto offset_global = tangent_exp * offset_x + bitangent_exp * offset_y;  // [n_rays, n_samples, 3]

		//Масштабируем смещения радиусами конуса и добавляем к точкам
		pts = pts + offset_global * cone_radii.unsqueeze(-1);
		if (bounding_box.defined())
		{
			std::vector<torch::Tensor> bounds = torch::split(bounding_box, { 3, 3 }, -1);
			auto min_bound = bounds[0].to(device);
			auto max_bound = bounds[1].to(device);
			pts = torch::clamp(pts, min_bound, max_bound);
		}
	}		//if (cone_angle > 0.0f)
	return pts;
};

///Volumetric rendering.
template <class TEmbedder, class TEmbedDirs, class TNeRF>
NeRFRenderResult NeRFRenderer<TEmbedder, TEmbedDirs, TNeRF> :: RenderRays(
	torch::Tensor ray_batch,		///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
	torch::Tensor cone_angle,
	const int n_samples,
	const bool return_raw /*= false*/,					///If True, include model's raw, unprocessed predictions.
	const bool lin_disp /*= false*/,						///If True, sample linearly in inverse depth rather than in depth.
	const float perturb /*= 0.f*/,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
	const int n_importance /*= 0*/,							///Number of additional times to sample along each ray.
	const bool white_bkgr /*= false*/,					///If True, assume a white background.
	const float raw_noise_std /*= 0.f*/,
	const float stochastic_preconditioning_alpha /*= 0.f*/,
	torch::Tensor bounding_box /* = torch::Tensor()*/,
	const bool return_weights /*= true*/
) {
	torch::Device device = ray_batch.device();		//Передать параметром??
	NeRFRenderResult result;
	///!!!Можно просто передавать структурку не парсить тензор
	int nrays = ray_batch.sizes()[0];
	auto rays_o = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3) });			//[N_rays, 3]		Origins
	auto rays_d = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(3, 6) });			//[N_rays, 3]		Directions
	torch::Tensor viewdirs;
	if (ray_batch.sizes().back() > 8)
		viewdirs = ray_batch.index({ torch::indexing::Slice(), torch::indexing::Slice(-3, torch::indexing::None) });
	auto ray_bounds = torch::reshape(ray_batch.index({ "...", torch::indexing::Slice(6, 8) }), { -1, 1, 2 });
	auto near = ray_bounds.index({ "...", 0 });
	auto far = ray_bounds.index({ "...", 1 });

	torch::Tensor t_vals = torch::linspace(0.f, 1.f, n_samples, torch::kFloat).to(device);
	torch::Tensor z_vals;
	if (!lin_disp)
	{
		z_vals = near * (1.f - t_vals) + far * (t_vals);
	} else {
		//z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals));
		auto safe_inv = [](torch::Tensor x, float eps = 1e-8f) {return torch::where(torch::abs(x) < eps, torch::ones_like(x) / eps, 1.0f / x);};
		z_vals = safe_inv(safe_inv(near) * (1.f - t_vals) + safe_inv(far) * t_vals);
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
	pts = TangentScatter(pts, z_vals, cone_angle, rays_d, device, bounding_box);

	torch::Tensor raw = RunNetwork(pts, viewdirs, /*torch::Tensor(),*/ NeRF, EmbedFn, EmbeddirsFn);
	auto outputs1 = RawToOutputs(raw, cone_angle, z_vals, rays_d, raw_noise_std, white_bkgr);

	if (n_importance > 0)
	{
		auto z_vals_mid = .5 * (z_vals.index({ "...", torch::indexing::Slice(1, torch::indexing::None) }) + z_vals.index({ "...", torch::indexing::Slice(torch::indexing::None, -1) }));
		auto z_samples = SamplePDF(z_vals_mid, outputs1.Weights.index({ "...", torch::indexing::Slice(1, -1) }), n_importance, perturb == 0.);
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
			auto noise =  torch::randn_like(pts) * stochastic_preconditioning_alpha;
			pts = pts + noise.to(device);
			pts = ReflectBoundary(pts, min_bound, max_bound);	//Обрабатываем границы отражением
		}

		pts = TangentScatter(pts, z_vals, cone_angle, rays_d, device, bounding_box);

		raw = RunNetwork(pts, viewdirs, NeRF, EmbedFn, EmbeddirsFn);
		result.Outputs = RawToOutputs(raw, cone_angle, z_vals, rays_d, raw_noise_std, white_bkgr);
		//result.ZStd = torch::std(z_samples, -1, false);  // [N_rays]
	}		//if (n_importance > 0)

	if (return_raw)
		result.Raw = raw;

	if (!return_weights)
		result.Outputs.Weights = torch::Tensor();

	return result;
}


///Render rays in smaller minibatches to save memory
///rays_flat.sizes()[0] должно быть кратно размеру chunk
template <class TEmbedder, class TEmbedDirs, class TNeRF>
NeRFRenderResult NeRFRenderer<TEmbedder, TEmbedDirs, TNeRF> :: BatchifyRays(
	torch::Tensor rays_flat,			///All information necessary for sampling along a ray, including : ray origin, ray direction, min dist, max dist, and unit - magnitude viewing direction.
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
	const bool return_weights /*= true*/
) {
	NeRFRenderResult result;
	std::vector<NeRFRenderResult> all_results;
	all_results.reserve(rays_flat.sizes()[0] / chunk);
	for (int i = 0; i < rays_flat.sizes()[0]; i += chunk)
	{
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
			return_weights
		));
	}
	//Слить all_results в один RenderResult используя torch::cat(torch::TensorList - at::ArrayRef ...
	//!!!make this part cleaner and shorter
	std::vector <torch::Tensor> out_rgb_map,
		out_disp_map,
		out_acc_map,
		out_weights,
		out_depth_map,
		raw;
	for (auto it : all_results)
	{
		if (it.Outputs.RGBMap.defined()) out_rgb_map.push_back(it.Outputs.RGBMap);
		if (it.Outputs.DispMap.defined()) out_disp_map.push_back(it.Outputs.DispMap);
		if (it.Outputs.AccMap.defined()) out_acc_map.push_back(it.Outputs.AccMap);
		if (it.Outputs.Weights.defined()) out_weights.push_back(it.Outputs.Weights);
		if (it.Outputs.DepthMap.defined()) out_depth_map.push_back(it.Outputs.DepthMap);
		if (it.Raw.defined()) raw.push_back(it.Raw);
	}
	if (!out_rgb_map.empty()) result.Outputs.RGBMap = torch::cat(out_rgb_map, 0);
	if (!out_disp_map.empty()) result.Outputs.DispMap = torch::cat(out_disp_map, 0);
	if (!out_acc_map.empty()) result.Outputs.AccMap = torch::cat(out_acc_map, 0);
	if (!out_weights.empty()) result.Outputs.Weights = torch::cat(out_weights, 0);
	if (!out_depth_map.empty()) result.Outputs.DepthMap = torch::cat(out_depth_map, 0);
	if (!raw.empty()) result.Raw = torch::cat(raw, 0);

	return result;
}


///Если определены позиции c2w то rays не нужен т.к.не используется (задавать либо pose c2w либо rays)
template <class TEmbedder, class TEmbedDirs, class TNeRF>
NeRFRenderResult NeRFRenderer<TEmbedder, TEmbedDirs, TNeRF> :: Render(
	const int h,					///Height of image in pixels.
	const int w,					///Width of image in pixels.
	torch::Tensor k,			///Сamera calibration
	const NeRFRenderParams &render_params,
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rays /*= { torch::Tensor(), torch::Tensor() }*/,			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	torch::Tensor c2w /*= torch::Tensor()*/,			///array of shape[3, 4].Camera - to - world transformation matrix.
	torch::Tensor c2w_staticcam /*= torch::Tensor()*/			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
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

	torch::Tensor viewdirs;
	if (render_params.UseViewdirs)
	{
		//provide ray directions as input
		viewdirs = rays_d;
		if (c2w_staticcam.defined() && c2w_staticcam.numel() != 0)
		{
			//special case to visualize effect of viewdirs
			std::tie(rays_o, rays_d, cone_angle) = GetRays(h, w, k, c2w_staticcam);
		}
		viewdirs = viewdirs / torch::norm(viewdirs, 2/*L2*/, -1/*dim*/, true);
		viewdirs = torch::reshape(viewdirs, { -1, 3 });//.float();
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
	
	if (render_params.UseViewdirs)
		rays_ = torch::cat({ rays_, viewdirs }, -1);

	//Renderand reshape
	NeRFRenderResult all_ret = std::move(BatchifyRays(
		rays_, render_params.ThinRay ? torch::Tensor() : cone_angle, render_params.NSamples, render_params.Chunk, render_params.ReturnRaw, render_params.LinDisp, render_params.Perturb,
		render_params.NImportance, render_params.WhiteBkgr, render_params.RawNoiseStd, render_params.StochasticPreconditioningAlpha, render_params.BoundingBox, render_params.ReturnWeights
	));

	if (all_ret.Outputs.RGBMap.numel() != 0)
		all_ret.Outputs.RGBMap = torch::reshape(all_ret.Outputs.RGBMap, sh);		//[640000, 3] -> [800, 800, 3]

	if (sh.size() > 2)			//не [4096, 3] а [800,800,3]
	{
		if (all_ret.Outputs.DispMap.numel() != 0)
			all_ret.Outputs.DispMap = torch::reshape(all_ret.Outputs.DispMap, { sh[0], sh[1] });	//[640000] -> [800,800]
		if (all_ret.Outputs.DepthMap.numel() != 0)
			all_ret.Outputs.DepthMap = torch::reshape(all_ret.Outputs.DepthMap, { sh[0], sh[1] });
	} else {
	}
	all_ret.Near = near_.min().item<float>();
	all_ret.Far = far_.max().item<float>();
	return all_ret;
}