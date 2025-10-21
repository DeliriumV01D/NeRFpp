#include "NeRFDataset.h"

NeRFDataset :: NeRFDataset(
	const NeRFDatasetParams &params,
	const LeRFDatasetParams &lerf_params,
	const int batch_size,
	const int precorp_iters,
	const float precorp_frac,
	torch::Device device,
	const CLIP clip,
	const std::shared_ptr<RuCLIPProcessor> clip_processor
) : Params(params), LeRFParams(lerf_params), BatchSize(batch_size), PrecorpIters(precorp_iters), PrecorpFrac(precorp_frac), Device(device),
Clip(clip), ClipProcessor(clip_processor), Rng(std::random_device{}())
{
	InitializePyramidClipEmbedding();
	//Загрузка первого изображения
	CurrentImageIdx = GetRandomTrainIdx();
	CurrentImage = LoadImage(CurrentImageIdx);
	//Предзагрузка следующего изображения
	PrefetchNextImage();
}

int NeRFDataset :: GetRandomTrainIdx()
{
	std::uniform_int_distribution<int> dist(0, Params.SplitsIdx[0] - 1);
	return dist(Rng);
}

torch::Tensor NeRFDataset :: LoadImage(const int idx) const
{
	const auto &path = Params.ImagePaths[idx];
	cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR/*cv::IMREAD_UNCHANGED*/);		//keep all 4 channels(RGBA)
	if (img.empty())
		throw std::runtime_error("NeRFDataset :: LoadImage error: Failed to load image: " + path.string());
	return CVMatToTorchTensor(img).squeeze(0).to(Device);		//1, 800, 800, 3(4) -> 800, 800, 3(4)
}

void NeRFDataset :: PrefetchNextImage()
{
	NextImageIdx = GetRandomTrainIdx();
	LoadingFuture = std::async(std::launch::async, [this] { NextImage = LoadImage(NextImageIdx); });
}

std::tuple<int, int, int, int> NeRFDataset :: CalculateBounds() const
{
	int h_start, h_end, w_start, w_end;
	if (CurrentIter < PrecorpIters)
	{
		int dh = static_cast<int>(Params.H / 2 * PrecorpFrac);
		int dw = static_cast<int>(Params.W / 2 * PrecorpFrac);
		h_start = Params.H / 2 - dh;
		h_end = Params.H / 2 + dh - 1;
		w_start = Params.W / 2 - dw;
		w_end = Params.W / 2 + dw - 1;
	}
	else {
		h_start = 0;
		h_end = Params.H - 1;
		w_start = 0;
		w_end = Params.W - 1;
	}
	return { h_start, h_end, w_start, w_end };
}

void NeRFDataset :: InitializePyramidClipEmbedding()
{
	try {
		if (LeRFParams.UseLerf)
		{
			PyramidEmbedderProperties pyramid_embedder_properties;
			pyramid_embedder_properties.ImgSize = { LeRFParams.clip_input_img_size, LeRFParams.clip_input_img_size };	//Входной размер изображения сети
			pyramid_embedder_properties.Overlap = LeRFParams.pyr_embedder_overlap;										///Доля перекрытия
			pyramid_embedder_properties.MinZoomOut = LeRFParams.MinZoomOut;		//0 or -1
			///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0 , 1, 2...
			pyramid_embedder_properties.MaxZoomOut = std::min(log2f(Params.W / LeRFParams.clip_input_img_size), log2f(Params.H / LeRFParams.clip_input_img_size));
			PyramidEmbedder PyramidClipEmbedder(Clip, ClipProcessor, pyramid_embedder_properties);

			if (!std::filesystem::exists(LeRFParams.PyramidClipEmbeddingSaveDir / "pyramid_embeddings.pt"))
			{
				std::cout << "calculating pyramid embeddings..." << std::endl;
				///Разбить на патчи с перекрытием  +  парочку масштабов (zoomout) и кэшировать эмбеддинги от них
				PyramidClipEmbedding = PyramidClipEmbedder(Params);
				PyramidClipEmbedding.Save(LeRFParams.PyramidClipEmbeddingSaveDir / "pyramid_embeddings.pt");
			}
			else {
				std::cout << "loading pyramid embeddings..." << std::endl;
				PyramidClipEmbedding.Load(LeRFParams.PyramidClipEmbeddingSaveDir / "pyramid_embeddings.pt");
			}
		}
	}
	catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
}


///
std::pair<torch::Tensor, torch::Tensor> NeRFDataset :: GetRayBatch(
	const torch::Tensor &rand_h,
	const torch::Tensor &rand_w,
	int H,
	int W,
	const torch::Tensor &K,
	const torch::Tensor &c2w
) {
	torch::Tensor fx = K[0][0];
	torch::Tensor fy = K[1][1];
	torch::Tensor cx = K[0][2];
	torch::Tensor cy = K[1][2];

	torch::Tensor dirsx = (rand_w.to(torch::kFloat32) - cx) / fx;
	torch::Tensor dirsy = -(rand_h.to(torch::kFloat32) - cy) / fy;
	torch::Tensor dirsz = -torch::ones_like(dirsx);
	//Get directions
	torch::Tensor dirs = torch::stack({ dirsx, dirsy, dirsz }, -1);		 // [h, w, 3]
	//Rotate ray directions from camera frame to the world frame
	auto rays_d = torch::sum(
		dirs.index({ "...", torch::indexing::None, torch::indexing::Slice() })
		* c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(torch::indexing::None, 3) }),
		-1);  //dot product, equals to : [c2w.dot(dir) for dir in dirs]
	//Translate camera frame's origin to the world frame. It is the origin of all rays.
	auto rays_o = c2w.index({ torch::indexing::Slice(torch::indexing::None, 3), -1 }).expand(rays_d.sizes());
	return { rays_o, rays_d };
}


///
NeRFDataExample  NeRFDataset :: get_batch(std::vector<size_t> request/*Не используется*/)
{
	torch::Tensor pose = Params.Poses[CurrentImageIdx].to(Device);		//Получаем позу для текущего изображения
	auto [h_start, h_end, w_start, w_end] = CalculateBounds();		//Вычисляем границы кадрирования
	//Прямая генерация случайных координат
	auto options = torch::TensorOptions().dtype(torch::kLong).device(Device);
	torch::Tensor rand_h = torch::randint(h_start, h_end + 1, { BatchSize }, options);
	torch::Tensor rand_w = torch::randint(w_start, w_end + 1, { BatchSize }, options);
	torch::Tensor target_s = CurrentImage.index({ rand_h, rand_w });		//Извлекаем цвета пикселей
	auto [rays_o, rays_d] = GetRayBatch(rand_h, rand_w, Params.H, Params.W, Params.K.to(Device), pose);		//Вычисляем лучи

	//!!!->class LeRFDataset
	///Вычислим CLIP эмбеддинги в точках изображения которые попали в батч
	torch::Tensor target_lang_embedding;
	if (LeRFParams.UseLerf)
	{
		PyramidEmbedderProperties pyramid_embedder_properties;
		pyramid_embedder_properties.ImgSize = { LeRFParams.clip_input_img_size, LeRFParams.clip_input_img_size };	//Входной размер изображения сети
		pyramid_embedder_properties.Overlap = LeRFParams.pyr_embedder_overlap;										///Доля перекрытия
		pyramid_embedder_properties.MinZoomOut = LeRFParams.MinZoomOut;
		///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0, 1, 2...
		pyramid_embedder_properties.MaxZoomOut = std::min(log2f(Params.W / LeRFParams.clip_input_img_size), log2f(Params.H / LeRFParams.clip_input_img_size));
		//auto select_coords_cpu = select_coords.to(torch::kCPU).to(torch::kFloat);		//!!!.item<long>()почему то не находит поэтому преобразуем во float
		target_lang_embedding = torch::ones({ BatchSize, LeRFParams.lang_embed_dim }, torch::kFloat32);

		#pragma omp parallel for
		for (int idx = 0; idx < rand_h.size(0)/*NRand*/; idx++)
		{
			target_lang_embedding.index_put_({ idx/*, torch::indexing::Slice()*/ }, PyramidClipEmbedding.GetPixelValue(
				rand_h.index({ idx }).to(torch::kCPU).to(torch::kFloat).item<float>(),
				rand_w.index({ idx }).to(torch::kCPU).to(torch::kFloat).item<float>(),
				0.5f,		///!!!ПРИВЯЗАТЬСЯ К МАСШТАБУ для этого перенести в RunNetwork по аналогии с calculated_normals			//Get zoom_out_idx from scale //-1, 0, 1, 2 ... <- 1/2, 1, 2, 4 ...
				CurrentImageIdx,
				pyramid_embedder_properties,
				cv::Size(Params.W, Params.H)
			));
		}
	}		//if use_lerf

	//Проверяем завершение предзагрузки
	if (LoadingFuture.valid())
	{
		if (LoadingFuture.wait_for(std::chrono::duration<int>::zero()) == std::future_status::ready)
		{
			CurrentImage = NextImage;
			CurrentImageIdx = NextImageIdx;
			PrefetchNextImage();		//Заново стартуем предзагрузку
		}
	}

	return { {rays_o, rays_d}, {target_s, target_lang_embedding} };
}