#pragma once

#include "TorchHeader.h"

inline torch::Tensor GetDirections(const int h, const int w, torch::Tensor k)
{
	auto device = k.device();
	//pytorch's meshgrid has indexing='ij'
	std::vector<torch::Tensor> ij = torch::meshgrid({ torch::linspace(0, w - 1, w), torch::linspace(0, h - 1, h) });
	auto i = ij[0].to(device).t();
	auto j = ij[1].to(device).t();
	torch::Tensor dirs = torch::stack({ (i - k[0][2]) / k[0][0], -(j - k[1][2]) / k[1][1], -torch::ones_like(i) }, -1);
	return dirs;
}

inline std::pair<torch::Tensor, torch::Tensor> GetRays(const int h, const int w, torch::Tensor k, torch::Tensor c2w)
{
	auto device = c2w.device();
	torch::Tensor dirs = GetDirections(h, w, k).to(device);

	//std::cout << (dirs.unsqueeze(-2) * c2w.slice(0, 0, 3).slice(1, 0, 3)).sizes() << std::endl;
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