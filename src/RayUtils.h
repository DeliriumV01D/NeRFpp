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

inline std::pair<torch::Tensor, torch::Tensor> GetRays(const int h, const int w, torch::Tensor k, torch::Tensor c2w)
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
	return std::make_pair(rays_o, rays_d);
}

///from camera to normalized device coordinate(NDC) space
inline std::pair<torch::Tensor, torch::Tensor> NDCRays(
	const int h,
	const int w,
	const float focal,
	const float near,
	torch::Tensor rays_o,
	torch::Tensor rays_d)
{
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
	return std::make_pair(rays_o, rays_d);
}