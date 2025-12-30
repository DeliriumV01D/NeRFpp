#pragma once

#include "TorchHeader.h"

inline torch::Tensor GetDirections(const int h, const int w, torch::Tensor k)
{
	const auto device = k.device();
	const auto options = torch::TensorOptions().device(device);
	//Создаем сетку на целевом устройстве (без транспонирования)
	auto y_range = torch::linspace(0, h - 1, h, options).view({ h, 1 }).expand({ h, w });
	auto x_range = torch::linspace(0, w - 1, w, options).view({ 1, w }).expand({ h, w });
	const auto fx = k[0][0];
	const auto cx = k[0][2];
	const auto fy = k[1][1];
	const auto cy = k[1][2];
	//Вычисляем направления (векторизованные операции)
	auto dir_x = (x_range - cx) / fx;
	auto dir_y = -(y_range - cy) / fy;
	auto dir_z = -torch::ones_like(x_range);
	return torch::stack({ dir_x, dir_y, dir_z }, -1); // [h, w, 3]
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetRays(const int h, const int w, torch::Tensor k, torch::Tensor c2w)
{
	auto device = c2w.device();
	torch::Tensor dirs = GetDirections(h, w, k).to(device);
	//Rotate ray directions from camera frame to the world frame
	auto rays_d = torch::sum(
		dirs.index({ "...", torch::indexing::None, torch::indexing::Slice() })
		* c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 3) }),
		-1);  //dot product, equals to : [c2w.dot(dir) for dir in dirs]
	//Translate camera frame's origin to the world frame. It is the origin of all rays.
	auto rays_o = c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), -1 }).expand(rays_d.sizes());

	//Вычисляем угловой размер пикселя в мировых координатах. Производная радиуса конуса = (размер пикселя) / (фокусное расстояние)
	auto fx = k.index({0, 0});
	auto fy = k.index({1, 1});
	//Размер пикселя в мировых координатах на расстоянии 1
	auto pixel_size_x = 1.0 / fx;
	auto pixel_size_y = 1.0 / fy;
	auto avg_pixel_size = (pixel_size_x + pixel_size_y) / 2.0;
	//Производная радиуса конуса = угловой размер пикселя. При расстоянии d, радиус конуса cone_ridius = d * pixel_size
	auto cone_angle = avg_pixel_size /** torch::ones({ h, w, 1 }, rays_d.options())*/ * 1.1f;  //Небольшой запас для антиалиасинга;

	return std::make_tuple(rays_o, rays_d, cone_angle);
}

///from camera to normalized device coordinate(NDC) space
inline std::tuple<torch::Tensor, torch::Tensor,torch::Tensor > NDCRays(
	const int h,
	const int w,
	const float focal,
	const float near,
	torch::Tensor rays_o,
	torch::Tensor rays_d,
	torch::Tensor cone_angle
) {
	//Shift ray origins to near plane
	auto t = -(near + rays_o.index({ "...", 2 })) / rays_d.index({ "...", 2 });
	rays_o = rays_o + t.index({ "...", torch::indexing::None }) * rays_d;

	//Projection
	auto o0 = -1. / (w / (2. * focal)) * rays_o.index({ "...", 0 }) / rays_o.index({ "...", 2 });
	auto o1 = -1. / (h / (2. * focal)) * rays_o.index({ "...", 1 }) / rays_o.index({ "...", 2 });
	auto o2 = 1. + 2. * near / rays_o.index({ "...", 2 });

	auto d0 = -1. / (w / (2. * focal)) * (rays_d.index({ "...", 0 }) / rays_d.index({ "...", 2 }) - rays_o.index({ "...", 0 }) / rays_o.index({ "...", 2 }));
	auto d1 = -1. / (h / (2. * focal)) * (rays_d.index({ "...", 1 }) / rays_d.index({ "...", 2 }) - rays_o.index({ "...", 1 }) / rays_o.index({ "...", 2 }));
	auto d2 = -2. * near / rays_o.index({ "...", 2 });

	rays_o = torch::stack({ o0, o1, o2 }, -1);
	rays_d = torch::stack({ d0, d1, d2 }, -1);

	if (cone_angle.defined() && cone_angle.numel())//if (cone_angle > 0.0f)
	{
		//Для NDC пространства также нужно масштабировать радиусы, производная радиуса должна быть пересчитана
		auto scale_factor = torch::sqrt(torch::pow(d0, 2) + torch::pow(d1, 2) + torch::pow(d2, 2))
			/ torch::sqrt(torch::pow(rays_d.index({ "...", 0 }), 2) + torch::pow(rays_d.index({ "...", 1 }), 2) + torch::pow(rays_d.index({ "...", 2 }), 2));
		cone_angle = cone_angle * scale_factor.unsqueeze(-1);
	}

	return std::make_tuple(rays_o, rays_d, cone_angle);
}


///
inline std::pair<torch::Tensor, torch::Tensor> IntersectWithAABB(
	const torch::Tensor &rays_o,
	const torch::Tensor &rays_d,
	const torch::Tensor &bounding_box,
	float near_plane = 0.f
) {
	auto aabb = bounding_box.reshape({ 2, 3 });

	//Avoid divide by zero
	auto dir_fraction = 1.0 / (rays_d + 1e-6);

	//Calculate intersections for each axis
	//x axis
	auto t1 = (aabb.index({ 0, 0 }) - rays_o.index({ "...", 0 }).unsqueeze(-1)) * dir_fraction.index({ "...", 0 }).unsqueeze(-1);
	auto t2 = (aabb.index({ 1, 0 }) - rays_o.index({ "...", 0 }).unsqueeze(-1)) * dir_fraction.index({ "...", 0 }).unsqueeze(-1);
	//y axis  
	auto t3 = (aabb.index({ 0, 1 }) - rays_o.index({ "...", 1 }).unsqueeze(-1)) * dir_fraction.index({ "...", 1 }).unsqueeze(-1);
	auto t4 = (aabb.index({ 1, 1 }) - rays_o.index({ "...", 1 }).unsqueeze(-1)) * dir_fraction.index({ "...", 1 }).unsqueeze(-1);
	//z axis
	auto t5 = (aabb.index({ 0, 2 }) - rays_o.index({ "...", 2 }).unsqueeze(-1)) * dir_fraction.index({ "...", 2 }).unsqueeze(-1);
	auto t6 = (aabb.index({ 1, 2 }) - rays_o.index({ "...", 2 }).unsqueeze(-1)) * dir_fraction.index({ "...", 2 }).unsqueeze(-1);

	//Compute near and far values
	auto min_x = torch::min(t1, t2);
	auto min_y = torch::min(t3, t4);
	auto min_z = torch::min(t5, t6);
	auto max_x = torch::max(t1, t2);
	auto max_y = torch::max(t3, t4);
	auto max_z = torch::max(t5, t6);

	//Near is the maximum of minimum intersections, far is the minimum of maximum intersections
	auto [nears, nidx] = torch::max(torch::cat({ min_x, min_y, min_z }, 1), 1);
	auto [fars, fidx] = torch::min(torch::cat({ max_x, max_y, max_z }, 1), 1);

	//Clamp to near plane and ensure far > near
	nears = torch::clamp_min(nears, near_plane);
	fars = torch::max(fars, nears + 1e-6f);

	return std::make_pair(nears, fars);
}