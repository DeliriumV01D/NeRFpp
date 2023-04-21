#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "NeRF.h"
#include "BaseNeRFRenderer.h"
#include "NeRFExecutor.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

const std::string DATA_DIR = "..//data//nerf_synthetic//lego";


void test()
{
	auto aa = torch::arange(9, torch::kFloat32) - 4;
	auto bb = aa.reshape({ 3, 3 });
	auto cc = torch::reshape(aa, { -1, 3 });
	std::cout << "aa: " << aa << torch::norm(aa) << std::endl; //tensor(7.7460)
	std::cout << "bb: " << bb << torch::norm(bb, 2/*L2*/,/*dim*/ -1) << std::endl; //tensor([5.3852, 1.4142, 5.3852])
	std::cout << "cc: " << cc << std::endl;

	auto a = torch::ones(1, torch::kFloat32) * 1e10;		//torch::full(...
	float dist_data[] = { 1, 2,
												1, 2 };
	auto dists = torch::from_blob(dist_data, { 2, 2, 1 });
	std::cout << "dists" << dists << " " << dists.sizes() << std::endl;
	dists = torch::cat({ dists, a.expand(dists.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }).sizes()) }, -1);
	std::cout << "dists" << dists << " " << dists.sizes() << std::endl;


	float data[] = { 1, 2, 3,
							 4, 5, 6 };
	torch::Tensor f = torch::from_blob(data, { 2, 3 }),
		f2;
	std::cout << "f2.defined():" << f2.defined() << std::endl;
	auto c = torch::cat({ f, f }, -1);
	std::cout << "torch::cat({f, f}, -1)" << c << std::endl;
	std::vector<torch::Tensor> fv;
	fv.push_back(f);
	fv.push_back(f);
	fv.push_back(f);
	auto c2 = torch::cat(fv, 0);
	std::cout << "c2: " << c2 << std::endl;
	std::cout << "torch::stack(fv)" << torch::stack(fv, 0) << std::endl;
	std::vector<torch::Tensor> splits = torch::split(c, { 3, 3 }, -1);
	std::cout << "torch::split(c, { 3, 3 }, -1)" << splits[0] << std::endl << splits[1] << std::endl;
	std::cout << "c2[:3,:3]" << c2.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 3) }) << std::endl;
	std::cout << "c2[1]" << c2.index({ 1 }) << std::endl << "c2[1]" << c2[1] << std::endl;

	std::vector<int64_t> sz(f.sizes().begin(), f.sizes().end());
	sz.push_back(10);
	c10::IntArrayRef rsz(&(*sz.begin()), &(*sz.cend()));
	std::cout << "rsz: " << rsz << std::endl;
	sz = f.sizes().vec();		//����� �������� ������
	sz.pop_back();
	sz.push_back(10);
	std::cout << "sz: " << sz << std::endl;

	c = torch::reshape(c, { 2, 3, 2 });
	auto c_flat = torch::reshape(c, { -1, c.sizes()[-1] });	//Error!
	std::cout << "c: " << c << " " << c.sizes() << std::endl;
	std::cout << "c_flat: " << c_flat << " " << c_flat.sizes() << std::endl;
	c_flat = torch::reshape(c, { -1, c.sizes().back() });
	std::cout << "c_flat: " << c_flat << " " << c_flat.sizes() << std::endl;

	torch::Tensor t2 = torch::tensor({ {1, 2}, {3, 4} });
	torch::Tensor t3 = torch::tensor({ {5, 6}, {7, 8} });
	torch::Tensor t1 = t2 * t3;			//������������ ���������
	std::cout << t1 << std::endl;
	// Output: tensor([[ 5, 12],
	//                 [21, 32]])



	float x_data[] = { 1, 2, 3 };
	torch::Tensor x = torch::from_blob(data, { 3, 1 });
	std::cout << "x" << x << std::endl;
	Embedder embedder("embedder", 5);
	auto embed_x = embedder->forward(x);
	std::cout << "embedder(x)" << embed_x << std::endl;


	//torch::nn::ModuleList nmlist{
	//	torch::nn::Linear(3, 4),
	//	torch::nn::BatchNorm1d(4),
	//	torch::nn::Dropout(0.5),
	//};

	//for (auto k : nmlist->named_parameters())
	//	std::cout << k.key() << std::endl;

	//std::cout << "params count: " << Trainable::ParamsCount(nmlist) << std::endl;

	//Trainable::Initialize(nmlist);


	// Create the device we pass around based on whether CUDA is available.
	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Training on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	} else {
		std::cout << "CUDA is not available! Training on CPU." << std::endl;
	}

	NeRF nerf(8, 256, 3, 3, 4, std::set<int>{4}, false, "nerf");
	nerf->to(device);
	Trainable::Initialize(nerf);

	for (auto &k : nerf->named_parameters())
		std::cout << k.key() << std::endl;

	std::cout << "params count: " << Trainable::ParamsCount(nerf) << std::endl;

	auto cd = load_blender_data(DATA_DIR, false, true);
	for (auto it : cd.Splits)
		std::cout << it << std::endl;

	for (auto it : cd.SplitsIdx)
		std::cout << it << std::endl;
}


int main(int argc, const char* argv[])
{
	torch::manual_seed(42);

	//test();

	NeRFExecutor nerf_executor(
		/*net_depth =*/ 8,				//layers in network
		/*net_width =*/ 256,			//channels per layer
		/*multires =*/ 10,
		/*use_viewdirs =*/ true,	//use full 5D input instead of 3D
		/*multires_views =*/ 4,		//log2 of max freq for positional encoding (2D direction)
		/*n_importance =*/ 0,			//number of additional fine samples per ray
		/*net_depth_fine =*/ 8,		//layers in fine network
		/*net_width_fine =*/ 256,	//channels per layer in fine network
		/*device =*/ torch::kCUDA,
		/*learning_rate =*/ 5e-4,
		/*ft_path =*/ "output"
	);
	NeRFExecutorTrainParams params;
	params.DatasetType = DatasetType::BLENDER;
	params.DataDir = DATA_DIR;			//input data directory
	params.BaseDir = "output";			//where to store ckpts and logs
	params.HalfRes = false;					////load blender synthetic data at 400x400 instead of 800x800
	params.WhiteBkgr = false;				//set to render synthetic data on a white bkgd (always use for dvoxels)
	params.RenderOnly = false;			//do not optimize, reload weights and render out render_poses path
	params.Ndc = true;							//use normalized device coordinates (set for non-forward facing scenes)
	params.LinDisp = false;					//sampling linearly in disparity rather than depth
	params.NoBatching = true;				//only take random rays from 1 image at a time
	params.Chunk = 1024 * 32;				//number of rays processed in parallel, decrease if running out of memory
	params.NSamples = 64;						//number of coarse samples per ray
	params.NRand = 32 * 32 * 4;			//batch size (number of random rays per gradient step)
	params.PrecorpIters = 0;				//number of steps to train on central crops
	params.LRateDecay = 250;				//exponential learning rate decay (in 1000 steps)
	//logging / saving options
	params.IPrint = 100;						//frequency of console printout and metric loggin
	params.IImg = 500;							//frequency of tensorboard image logging
	params.IWeights = 10000;				//frequency of weight ckpt saving
	params.ITestset = 50000;				//frequency of testset saving
	params.IVideo = 50000;					//frequency of render_poses video saving
	params.ReturnRaw = false;
	params.RenderFactor = 0;
	params.PrecorpFrac = 0.5f;

	nerf_executor.Train(params);
}