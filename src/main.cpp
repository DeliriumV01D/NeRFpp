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
	auto t = torch::tensor({ {{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1}} });
	std::cout << "t: " << t << std::endl;

	// Создание тензоров cdf и u
	torch::Tensor cdf = torch::rand({ 96, 3 });
	torch::Tensor u = torch::rand({ 96, 10 });

	//searchsorted работает только с тензорами одинакового размера вот мы и сводим задачу к этому
	auto inds = torch::searchsorted(cdf, u, false, true);
	std::cout << inds << std::endl;

	//// Создание тензоров cdf и u
	//torch::Tensor cdf = torch::rand({ 96, 3 });
	//torch::Tensor u = torch::rand({ 96, 3, 10 });

	////searchsorted работает только с тензорами одинакового размера вот мы и сводим задачу к этому
	//torch::Tensor inds = torch::zeros(u.sizes(), torch::kLong);
	//for (int c = 0; c < u.sizes().back(); c++)
	//{
	//	auto y = u.index({ "...", c });
	//	auto ynds = torch::searchsorted(cdf, y, /*out_int32*/false, /*right=*/true);
	//	std::cout << "k " << y.sizes() << " " << y.type() << std::endl;
	//	std::cout << "ynds " << ynds.sizes() << " " << ynds.type() << std::endl;
	//	inds.index_put_({ "...", c }, ynds);
	//}
	//std::cout << inds << std::endl;


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
	sz = f.sizes().vec();		//Более красивый способ
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
	torch::Tensor t1 = t2 * t3;			//Поэлементное умножение
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

	NeRFExecutor <HashEmbedder, SHEncoder, NeRFSmall> nerf_executor(
		/*net_depth =*/ 2,				//layers in network 8 for classic NeRF, 2/3 for HashNeRF
		/*net_width =*/ 64,				//channels per layer 256 for classic NeRF, 64 for HashNeRF
		/*multires =*/ 10,
		/*use_viewdirs =*/ false,	//use full 5D input instead of 3D Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
		/*multires_views =*/ 4,		//log2 of max freq for positional encoding (2D direction)
		/*n_importance =*/ 192,		//number of additional fine samples per ray
		/*net_depth_fine =*/ 2,		//layers in fine network 8 for classic NeRF, 2/3 for HashNeRF
		/*net_width_fine =*/ 64,	//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
		/*num_layers_color =*/ 3,				//for color part of the HashNeRF
		/*hidden_dim_color =*/ 64,			//for color part of the HashNeRF
		/*num_layers_color_fine =*/ 3,	//for color part of the HashNeRF
		/*hidden_dim_color_fine =*/ 64,	//for color part of the HashNeRF
		/*bounding_box =*/ torch::tensor({-4.f, -4.f, -4.f, 4.f, 4.f, 4.f})/*.to(device)*/,
		/*n_levels =*/ 16,
		/*n_features_per_level =*/ 2,
		/*log2_hashmap_size =*/ 19,		//19
		/*base_resolution =*/ 16,
		/*finest_resolution =*/ 512,
		/*device =*/ torch::kCUDA,
		/*learning_rate =*/ 1e-2,		//5e-4 for classic NeRF
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
	params.Chunk = 1024 * 8/*16*/;				//number of rays processed in parallel, decrease if running out of memory
	params.NetChunk = 1024 * 32/*32*/,		//number of pts sent through network in parallel, decrease if running out of memory
	params.NSamples = 64;						//number of coarse samples per ray
	params.NRand = 32 * 32 * 1/*4*/;			//batch size (number of random rays per gradient step)
	params.PrecorpIters = 0;				//number of steps to train on central crops
	params.NIters = 15100;
	params.LRateDecay = 7;				//exponential learning rate decay (in 1000 steps)  например: 150 - каждые 150000 итераций скорость обучения будет падать в 10 раз
	//logging / saving options
	params.IPrint = 100;						//frequency of console printout and metric loggin
	params.IImg = 500;							//frequency of tensorboard image logging
	params.IWeights = 15000;				//frequency of weight ckpt saving
	params.ITestset = 15000;				//frequency of testset saving
	params.IVideo = 15200;					//frequency of render_poses video saving
	params.ReturnRaw = false;
	params.RenderFactor = 0;
	params.PrecorpFrac = 0.5f;

	nerf_executor.Train(params);
}