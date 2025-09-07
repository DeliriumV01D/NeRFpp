#include "TorchHeader.h"
#include "load_blender.h"
#include "Trainable.h"
#include "CuSHEncoder.h"
#include "CuHashEmbedder.h"
#include "NeRF.h"
#include "NeRFRenderer.h"
#include "NeRFExecutor.h"
#include "LeRF.h"
#include "LeRFRenderer.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

const std::string DATA_DIR = "..//data//nerf_synthetic//drums";


//void test()
//{
//	auto t = torch::tensor({ {{0, 0, 0},
//		{0, 0, 1},
//		{0, 1, 0},
//		{0, 1, 1},
//		{1, 0, 0},
//		{1, 0, 1},
//		{1, 1, 0},
//		{1, 1, 1}} });
//	std::cout << "t: " << t << std::endl;
//
//	// Создание тензоров cdf и u
//	torch::Tensor cdf = torch::rand({ 96, 3 });
//	torch::Tensor u = torch::rand({ 96, 10 });
//
//	//searchsorted работает только с тензорами одинакового размера вот мы и сводим задачу к этому
//	auto inds = torch::searchsorted(cdf, u, false, true);
//	std::cout << inds << std::endl;
//
//	//// Создание тензоров cdf и u
//	//torch::Tensor cdf = torch::rand({ 96, 3 });
//	//torch::Tensor u = torch::rand({ 96, 3, 10 });
//
//	////searchsorted работает только с тензорами одинакового размера вот мы и сводим задачу к этому
//	//torch::Tensor inds = torch::zeros(u.sizes(), torch::kLong);
//	//for (int c = 0; c < u.sizes().back(); c++)
//	//{
//	//	auto y = u.index({ "...", c });
//	//	auto ynds = torch::searchsorted(cdf, y, /*out_int32*/false, /*right=*/true);
//	//	std::cout << "k " << y.sizes() << " " << y.type() << std::endl;
//	//	std::cout << "ynds " << ynds.sizes() << " " << ynds.type() << std::endl;
//	//	inds.index_put_({ "...", c }, ynds);
//	//}
//	//std::cout << inds << std::endl;
//
//
//	auto aa = torch::arange(9, torch::kFloat32) - 4;
//	auto bb = aa.reshape({ 3, 3 });
//	auto cc = torch::reshape(aa, { -1, 3 });
//	std::cout << "aa: " << aa << torch::norm(aa) << std::endl; //tensor(7.7460)
//	std::cout << "bb: " << bb << torch::norm(bb, 2/*L2*/,/*dim*/ -1) << std::endl; //tensor([5.3852, 1.4142, 5.3852])
//	std::cout << "cc: " << cc << std::endl;
//
//	auto a = torch::ones(1, torch::kFloat32) * 1e10;		//torch::full(...
//	float dist_data[] = { 1, 2,
//												1, 2 };
//	auto dists = torch::from_blob(dist_data, { 2, 2, 1 });
//	std::cout << "dists" << dists << " " << dists.sizes() << std::endl;
//	dists = torch::cat({ dists, a.expand(dists.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) }).sizes()) }, -1);
//	std::cout << "dists" << dists << " " << dists.sizes() << std::endl;
//
//
//	float data[] = { 1, 2, 3,
//							 4, 5, 6 };
//	torch::Tensor f = torch::from_blob(data, { 2, 3 }),
//		f2;
//	std::cout << "f2.defined():" << f2.defined() << std::endl;
//	auto c = torch::cat({ f, f }, -1);
//	std::cout << "torch::cat({f, f}, -1)" << c << std::endl;
//	std::vector<torch::Tensor> fv;
//	fv.push_back(f);
//	fv.push_back(f);
//	fv.push_back(f);
//	auto c2 = torch::cat(fv, 0);
//	std::cout << "c2: " << c2 << std::endl;
//	std::cout << "torch::stack(fv)" << torch::stack(fv, 0) << std::endl;
//	std::vector<torch::Tensor> splits = torch::split(c, { 3, 3 }, -1);
//	std::cout << "torch::split(c, { 3, 3 }, -1)" << splits[0] << std::endl << splits[1] << std::endl;
//	std::cout << "c2[:3,:3]" << c2.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 3) }) << std::endl;
//	std::cout << "c2[1]" << c2.index({ 1 }) << std::endl << "c2[1]" << c2[1] << std::endl;
//
//	std::vector<int64_t> sz(f.sizes().begin(), f.sizes().end());
//	sz.push_back(10);
//	c10::IntArrayRef rsz(&(*sz.begin()), &(*sz.cend()));
//	std::cout << "rsz: " << rsz << std::endl;
//	sz = f.sizes().vec();		//Более красивый способ
//	sz.pop_back();
//	sz.push_back(10);
//	std::cout << "sz: " << sz << std::endl;
//
//	c = torch::reshape(c, { 2, 3, 2 });
//	auto c_flat = torch::reshape(c, { -1, c.sizes()[-1] });	//Error!
//	std::cout << "c: " << c << " " << c.sizes() << std::endl;
//	std::cout << "c_flat: " << c_flat << " " << c_flat.sizes() << std::endl;
//	c_flat = torch::reshape(c, { -1, c.sizes().back() });
//	std::cout << "c_flat: " << c_flat << " " << c_flat.sizes() << std::endl;
//
//	torch::Tensor t2 = torch::tensor({ {1, 2}, {3, 4} });
//	torch::Tensor t3 = torch::tensor({ {5, 6}, {7, 8} });
//	torch::Tensor t1 = t2 * t3;			//Поэлементное умножение
//	std::cout << t1 << std::endl;
//	// Output: tensor([[ 5, 12],
//	//                 [21, 32]])
//
//
//
//	float x_data[] = { 1, 2, 3 };
//	torch::Tensor x = torch::from_blob(data, { 3, 1 });
//	std::cout << "x" << x << std::endl;
//	Embedder embedder("embedder", 5);
//	auto embed_x = embedder->forward(x);
//	std::cout << "embedder(x)" << embed_x << std::endl;
//
//
//	//torch::nn::ModuleList nmlist{
//	//	torch::nn::Linear(3, 4),
//	//	torch::nn::BatchNorm1d(4),
//	//	torch::nn::Dropout(0.5),
//	//};
//
//	//for (auto k : nmlist->named_parameters())
//	//	std::cout << k.key() << std::endl;
//
//	//std::cout << "params count: " << Trainable::ParamsCount(nmlist) << std::endl;
//
//	//Trainable::Initialize(nmlist);
//
//
//	// Create the device we pass around based on whether CUDA is available.
//	torch::Device device(torch::kCPU);
//	if (torch::cuda::is_available())
//	{
//		std::cout << "CUDA is available! Training on GPU." << std::endl;
//		device = torch::Device(torch::kCUDA);
//	} else {
//		std::cout << "CUDA is not available! Training on CPU." << std::endl;
//	}
//
//	NeRF nerf(8, 256, 3, 3, 4, std::set<int>{4}, false, "nerf");
//	nerf->to(device);
//	Trainable::Initialize(nerf);
//
//	for (auto &k : nerf->named_parameters())
//		std::cout << k.key() << std::endl;
//
//	std::cout << "params count: " << Trainable::ParamsCount(nerf) << std::endl;
//
//	auto cd = load_blender_data(DATA_DIR, 0.f, 0.f, false, true);
//	for (auto it : cd.Splits)
//		std::cout << it << std::endl;
//
//	for (auto it : cd.SplitsIdx)
//		std::cout << it << std::endl;
//}


int main(int argc, const char* argv[])
{
	torch::manual_seed(42);

	//test();

	NeRFExecutorParams exparams;
	exparams.net_depth = 2;				//layers in network 8 for classic NeRF, 2/3 for HashNeRF
	exparams.net_width = 64;				//channels per layer 256 for classic NeRF, 64 for HashNeRF
	exparams.multires = 10;
	exparams.use_nerf = true;
	exparams.use_viewdirs = true;	//use full 5D input instead of 3D Не всегда нужна зависимость от направления обзора + обучение быстрее процентов на 30.
	exparams.calculate_normals = false;
	exparams.use_pred_normal = false;	//whether to use predicted normals
	exparams.use_lerf = false;				//use language embedded radiance fields
	exparams.multires_views = 8;		//log2 of max freq for positional encoding (2D direction)
	exparams.n_importance = 192;//192;		//number of additional fine samples per ray
	exparams.net_depth_fine = 3;		//layers in fine network 8 for classic NeRF, 2/3 for HashNeRF
	exparams.net_width_fine = 64;		//channels per layer in fine network 256 for classic NeRF, 64 for HashNeRF
	exparams.num_layers_color = 2;				//for color part of the HashNeRF
	exparams.hidden_dim_color = 64;			//for color part of the HashNeRF
	exparams.num_layers_color_fine = 3;	//for color part of the HashNeRF
	exparams.hidden_dim_color_fine = 64;	//for color part of the HashNeRF
	exparams.num_layers_normals = 2;			//!!!->2
	exparams.hidden_dim_normals = 64;
	exparams.geo_feat_dim = 15;
	exparams.n_levels = 18;
	exparams.n_features_per_level = 2;
	exparams.log2_hashmap_size = 21;		//19
	exparams.base_resolution = 16;
	exparams.finest_resolution = 1024;
	exparams.device = torch::kCUDA;
	exparams.learning_rate = 1e-2;		//5e-4 for classic NeRF
	exparams.ft_path = "output";
	exparams.n_levels_le = exparams.n_levels/*32*/,																		//for language embedder
	exparams.n_features_per_level_le = 8/*8*/,								//for language embedder
	exparams.log2_hashmap_size_le = 19,									//for language embedder
	exparams.base_resolution_le = exparams.base_resolution,													//for language embedder
	exparams.finest_resolution_le = exparams.finest_resolution,										//for language embedder
	exparams.pyr_embedder_overlap = 0.75f;
	exparams.clip_input_img_size = 336;	//Input RuClip model size
	exparams.num_layers_le = 2;					//Language embedder head params
	exparams.hidden_dim_le = 256;				//Language embedder head params
	exparams.lang_embed_dim = 768;			//Language embedder head params
	exparams.geo_feat_dim_le = 32;			//Language embedder head params
	exparams.path_to_clip = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336";									//Path to RuClip model
	exparams.path_to_bpe = "..//..//RuCLIP//data//ruclip-vit-large-patch14-336//bpe.model";			//Path to tokenizer
	exparams.lerf_positives = "металлическая тарелка";//"металлическая тарелка";//"красный барабан";
	exparams.lerf_negatives = {"объект", "предметы", "текстура"};
	NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall, NeRFRenderer<CuHashEmbedder, CuSHEncoder, NeRFSmall>,
		CuHashEmbedder/*LeRFEmbedder<CuHashEmbedder>*/, LeRF, LeRFRenderer> nerf_executor(exparams);
	//NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall, NeRFRenderer<CuHashEmbedder, CuSHEncoder, NeRFSmall>> nerf_executor(exparams);

	NeRFExecutorTrainParams params;
	params.BaseDir = "output";			//where to store ckpts and logs
	params.RenderOnly = false;			//do not optimize, reload weights and render out render_poses path
	params.Ndc = false;							//use normalized device coordinates (set for non-forward facing scenes)
	params.LinDisp = false;					//sampling linearly in disparity rather than depth
	params.TestSkip = false;
	params.Chunk = 1024 * (exparams.use_lerf ? 1 : 4);				//number of rays processed in parallel, decrease if running out of memory <= NRand
	params.NSamples = 64;						//number of coarse samples per ray
	params.NRand = 32 * 32 * (exparams.use_lerf ? 1 : 16);		//batch size (number of random rays per gradient step), decrease if running out of memory >= Chunk, n*Chunk
	params.PrecorpIters = 0;				//number of steps to train on central crops
	params.NIters = 6100;
	params.LRateDecay = 4;				//exponential learning rate decay (in 1000 steps)  например: 150 - каждые 150000 итераций скорость обучения будет падать в 10 раз
	//logging / saving options
	params.IPrint = 100;						//frequency of console printout and metric loggin
	params.IImg = 500;							//frequency of tensorboard image logging
	params.IWeights = 6000;				//frequency of weight ckpt saving
	params.ITestset = 6000;				//frequency of testset saving
	params.IVideo = 6200;					//frequency of render_poses video saving
	params.ReturnRaw = false;
	params.RenderFactor = 0;
	params.PrecorpFrac = 0.5f;
	params.PyramidClipEmbeddingSaveDir = DATA_DIR;			//

	NeRFDatasetParams data = LoadDatasetParams(
		DATA_DIR,
		exparams.device,
		DatasetType::BLENDER, 
		false,			///load blender synthetic data at 400x400 instead of 800x800
		params.TestSkip,
		false				///set to render synthetic data on a white bkgd (always use for dvoxels)
	);

	nerf_executor.Train(data, params);

	exparams.SaveToFile(params.BaseDir / "executor_params.json");
	params.SaveToFile(params.BaseDir / "executor_train_params.json");
	data.SaveToFile(params.BaseDir / "data.json");
}