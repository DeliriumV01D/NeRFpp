#pragma once

#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "NeRF.h"
#include "BaseNeRFRenderer.h"
#include "TRandomInt.h"

#include <set>
#include <filesystem>
#include <chrono>

enum class DatasetType { LLFF, BLENDER, LINEMOD, DEEP_VOXELS };

struct NeRFExecutorTrainParams {
	DatasetType DatasetType = DatasetType::BLENDER;
	std::filesystem::path DataDir,	///input data directory
		BaseDir;											///where to store ckpts and logs
	bool HalfRes{ false },					///load blender synthetic data at 400x400 instead of 800x800
		TestSkip{ false },						///will load 1/N images from test/val sets, useful for large datasets like deepvoxels
		WhiteBkgr{ false },						///set to render synthetic data on a white bkgd (always use for dvoxels)
		RenderOnly{ false },					///do not optimize, reload weights and render out render_poses path
		Ndc{ true },									///use normalized device coordinates (set for non-forward facing scenes)
		LinDisp{ false },							///sampling linearly in disparity rather than depth
		NoBatching{ true };						///only take random rays from 1 image at a time
	int Chunk{ 1024 * 32 },					///number of rays processed in parallel, decrease if running out of memory
		NetChunk{ 1024 * 64 },				///number of pts sent through network in parallel, decrease if running out of memory
		NSamples{ 64 },								///number of coarse samples per ray
		NRand{ 32 * 32 * 4 },					///batch size (number of random rays per gradient step) must be < H * W
		PrecorpIters{ 0 },						///number of steps to train on central crops
		LRateDecay{ 250 },						///exponential learning rate decay (in 1000 steps)
		//logging / saving options
		IPrint{ 100 },			///frequency of console printout and metric loggin
		IImg{ 500 },				///frequency of tensorboard image logging
		IWeights{ 10000 },	///frequency of weight ckpt saving
		ITestset{ 50000 },	///frequency of testset saving
		IVideo{ 50000 };		///frequency of render_poses video saving
	bool ReturnRaw{ false };
	float RenderFactor{ 0 },
		PrecorpFrac{ 0.5f };
};


class NeRFExecutor {
protected:
	Embedder ExecutorEmbedder = nullptr,
		ExecutorEmbeddirs = nullptr;
	NeRF Model = nullptr ,
		ModelFine = nullptr;
	std::vector<torch::Tensor> GradVars;
	std::unique_ptr<torch::optim::Adam> Optimizer;
	int Start = 0,
		NImportance{ 0 };
	float LearningRate;
	torch::Device Device;
	bool UseViewDirs;									//use full 5D input instead of 3D
public:
	NeRFExecutor(
		const int net_depth = 8,				//layers in network
		const int net_width = 256,			//channels per layer
		const int multires = 10,
		const bool use_viewdirs = true,	//use full 5D input instead of 3D
		const int multires_views = 4,		//log2 of max freq for positional encoding (2D direction)
		const int n_importance = 0,			//number of additional fine samples per ray
		const int net_depth_fine = 8,		//layers in fine network
		const int net_width_fine = 256,	//channels per layer in fine network
		torch::Device device = torch::kCUDA,
		const float learning_rate = 5e-4,
		std::filesystem::path ft_path = ""
	);

	///
	std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> RenderPath(
		const std::vector <torch::Tensor>& render_poses,
		int h,
		int w,
		float focal,
		torch::Tensor k,
		const int n_samples,
		const int chunk = 1024 * 32,						///Maximum number of rays to process simultaneously.Used to control maximum memory usage.Does not affect final results.
		const int net_chunk = 1024 * 64,				///number of pts sent through network in parallel, decrease if running out of memory
		const bool return_raw = false,					///If True, include model's raw, unprocessed predictions.
		const bool lin_disp = false,						///If True, sample linearly in inverse depth rather than in depth.
		const float perturb = 0.f,							///0. or 1. If non - zero, each ray is sampled at stratified random points in time.
		const int n_importance = 0,							///Number of additional times to sample along each ray.
		const bool white_bkgr = false,					///If True, assume a white background.
		const float raw_noise_std = 0.,
		std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() },			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		//torch::Tensor c2w = torch::Tensor(),			///array of shape[3, 4].Camera - to - world transformation matrix.
		const bool ndc = true,						///If True, represent ray origin, direction in NDC coordinates.
		const float near = 0.,						///float or array of shape[batch_size].Nearest distance for a ray.
		const float far = 1.,							///float or array of shape[batch_size].Farthest distance for a ray.
		const bool use_viewdirs = false,	///If True, use viewing direction of a point in space in model.
		torch::Tensor c2w_staticcam = torch::Tensor(),			///array of shape[3, 4].If not None, use this transformation matrix for camera while using other c2w argument for viewing directions.
		//torch::Tensor gt_imgs = torch::Tensor(),
		const std::filesystem::path savedir = "",
		const float render_factor = 0
	);

	///
	void Train(NeRFExecutorTrainParams &params);
};