#pragma once

#include "TorchHeader.h"

///Hierarchical sampling
inline torch::Tensor SamplePDF(torch::Tensor bins, torch::Tensor weights, const int nsamples, const bool det /*= false*/)
{
	torch::Device device = weights.device();
	//Get probability density function (PDF)
	weights = weights + 1e-8;
	auto pdf = weights / torch::sum(weights, -1, true);
	auto cdf = torch::cumsum(pdf, -1);
	cdf = torch::cat({ torch::zeros_like(cdf.index({ "...", torch::indexing::Slice(torch::indexing::None, 1)})), cdf }, -1);		//[batch, len(bins)]
	torch::Tensor u;
	//Take uniform samples
	std::vector<int64_t> sz(cdf.sizes().begin(), cdf.sizes().end());
	sz.back() = nsamples;
	if (det)
	{
		u = torch::linspace(0.f, 1.f, nsamples, torch::kFloat);
		u = u.expand(sz);
	}
	else {
		u = torch::rand(sz);
	}

	//Invert cumulative distribution function (CDF)
	u = u.contiguous().to(device);
	auto inds = torch::searchsorted(cdf, u, false, true);
	auto below = torch::max(torch::zeros_like(inds - 1), inds - 1);
	auto above = torch::min((cdf.sizes().back() - 1) * torch::ones_like(inds), inds);
	auto inds_g = torch::stack({ below, above }, -1);  //[batch, N_samples, 2];

	std::vector< int64_t> matched_shape{ inds_g.sizes()[0], inds_g.sizes()[1], cdf.sizes().back() };
	auto cdf_g = torch::gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g);
	auto bins_g = torch::gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g);

	auto denom = (cdf_g.index({ "...", 1 }) - cdf_g.index({ "...", 0 }));
	denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);
	auto t = (u - cdf_g.index({ "...", 0 })) / denom;
	auto samples = bins_g.index({ "...", 0 }) + t * (bins_g.index({ "...", 1 }) - bins_g.index({ "...", 0 }));

	return samples;
}