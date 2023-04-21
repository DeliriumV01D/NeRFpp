#pragma once

#include "TorchHeader.h"
#include "Trainable.h"

#include <set>


inline std::pair<torch::Tensor, torch::Tensor> GetRays(const int h, const int w, torch::Tensor k, torch::Tensor c2w)
{
	auto device = c2w.device();
	//pytorch's meshgrid has indexing='ij'
	std::vector<torch::Tensor> ij = torch::meshgrid({ torch::linspace(0, w - 1, w), torch::linspace(0, h - 1, h) });
	auto i = ij[0].to(device).t();
	auto j = ij[1].to(device).t();
	torch::Tensor dirs = torch::stack({ (i - k[0][2]) / k[0][0], - (j - k[1][2]) / k[1][1], - torch::ones_like(i) }, -1);

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

//Hierarchical sampling
inline torch::Tensor SamplePDF(torch::Tensor bins, torch::Tensor weights, const int nsamples, const bool det = false)
{
	//Get probability density function (PDF)
	weights = weights + 1e-5;
	auto pdf = weights / torch::sum(weights, -1, true);
	auto cdf = torch::cumsum(pdf, -1);
	cdf = torch::cat({ torch::zeros_like(cdf.index({ "...", torch::indexing::Slice(torch::indexing::None, 1)})), cdf }, -1);
	torch::Tensor u;

	//Take uniform samples
	std::vector<int64_t> sz(cdf.sizes().begin(), cdf.sizes().end());
	sz.push_back(nsamples);
	if (det)
	{
		u = torch::linspace(0., 1., nsamples);
		u = u.expand(c10::IntArrayRef(&(*sz.begin()), &(*sz.end())));
	}	else {
		u = torch::rand(c10::IntArrayRef(&(*sz.begin()), &(*sz.end())));
	}

	//Invert cumulative distribution function (CDF)
	u = u.contiguous();
	auto inds = torch::searchsorted(cdf, u, false, true);
	auto below = torch::max(torch::zeros_like(inds - 1), inds - 1);
	auto above = torch::min((cdf.sizes().back() - 1) * torch::ones_like(inds), inds);
	auto inds_g = torch::stack({ below, above }, -1);  //(batch, N_samples, 2);

	auto matched_shape = { inds_g.sizes()[0], inds_g.sizes()[1], cdf.sizes().back() };
	auto cdf_g = torch::gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g);
	auto bins_g = torch::gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g);

	auto denom = (cdf_g.index({ "...", 1 }) - cdf_g.index({ "...", 0 }));
	denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);
	auto t = (u - cdf_g.index({ "...", 0})) / denom;
	auto samples = bins_g.index({ "...", 0 }) + t * (bins_g.index({ "...", 1 }) - bins_g.index({ "...", 0}));

	return samples;
}

///Positional encoding
class EmbedderImpl : torch::nn::Module {
protected:
	int NumFreqs;
	float MaxFreq;
	bool IncludeInput;
	int InputDims,
		OutputDims = 0;;
	bool LogSampling;
	std::vector<float> FreqBands;
public:
	EmbedderImpl(const std::string &module_name, int multires)
		: EmbedderImpl(module_name, multires, multires - 1){}
	EmbedderImpl(const std::string& module_name, int num_freqs, float max_freq_log2, bool include_input = true, int input_dims = 3, bool log_sampling = true);
	int GetOutputDims() { return OutputDims; }
	torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Embedder);

struct NeRFImpl : public Trainable
{
	int D,
		W,
		InputCh,
		InputChViews,
		OutputCh;
	std::set<int> Skips;
	bool UseViewDirs;
	
	torch::nn::ModuleList PtsLinears,
		ViewsLinears;

	torch::nn::Linear FeatureLinear = nullptr,
		AlphaLinear = nullptr,
		RGBLinear = nullptr,
		OutputLinear = nullptr;

	NeRFImpl(
		const int d = 8,
		const int w = 256,
		const int input_ch = 3,
		const int input_ch_views = 3,
		const int output_ch = 4,
		const std::set<int> &skips = std::set<int>{ 4 },
		const bool use_viewdirs = false,
		const std::string module_name = "nerf"
	);
	
	virtual ~NeRFImpl() override {}

	virtual torch::Tensor forward(torch::Tensor x); //override

	///x.sizes()[0] должно быть кратно размеру chunk
	virtual torch::Tensor Batchify(torch::Tensor x, const int chunk);
};

TORCH_MODULE(NeRF);

