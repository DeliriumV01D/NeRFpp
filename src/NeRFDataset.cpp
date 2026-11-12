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
) : Params(params), LeRFParams(lerf_params), BatchSize(batch_size), PrecorpIters(precorp_iters), PrecorpFrac(precorp_frac), Device(device), Rng(std::random_device{}())
{
	InitializePyramidClipEmbedding(clip, clip_processor);
	//Load the first image and start async prefetch of the next one
	CurrentImageIdx = GetRandomTrainIdx();
	CurrentImage = LoadImage(CurrentImageIdx);
	PrefetchNextImage();
}

int NeRFDataset :: GetRandomTrainIdx()
{
	std::uniform_int_distribution<int> dist(0, Params.SplitsIdx[0] - 1);
	return dist(Rng);
}

torch::Tensor NeRFDataset :: LoadImage(const int idx) const
{
	const auto &path = Params.Views[idx].ImagePath;
	cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR/*cv::IMREAD_UNCHANGED*/); //keep all 4 channels(RGBA)
	if (img.empty())
		throw std::runtime_error("NeRFDataset :: LoadImage error: Failed to load image: " + path.string());
	return CVMatToTorchTensor(img).squeeze(0).clone().to(Device); // (1, H, W, 3) -> (H, W, 3)
}

void NeRFDataset :: PrefetchNextImage()
{
	NextImageIdx = GetRandomTrainIdx();
	LoadingFuture = std::async(std::launch::async, [this] { NextImage = LoadImage(NextImageIdx); });
}

std::tuple<int, int, int, int> NeRFDataset :: CalculateBounds() const
{
	int h_start, h_end, w_start, w_end,
		h = Params.Views[CurrentImageIdx].H,
		w = Params.Views[CurrentImageIdx].W;
	if (CurrentIter < PrecorpIters)
	{
		int dh = static_cast<int>(h / 2 * PrecorpFrac);
		int dw = static_cast<int>(w / 2 * PrecorpFrac);
		h_start = h / 2 - dh;
		h_end = h / 2 + dh - 1;
		w_start = w / 2 - dw;
		w_end = w / 2 + dw - 1;
	}
	else {
		h_start = 0;
		h_end = h - 1;
		w_start = 0;
		w_end = w - 1;
	}
	return { h_start, h_end, w_start, w_end };
}




void NeRFDataset :: InitializePyramidClipEmbedding(const CLIP clip, const std::shared_ptr<RuCLIPProcessor> clip_processor)
{

	if (!LeRFParams.UseLerf) {
		PyramidClipEmbedding = nullptr;
		return;
	}

	//Собираем конфиг для PyramidEmbeddingDataloader
	int wmax = 0, hmax = 0;
	for (const auto &view : Params.Views)
	{
		wmax = std::max(wmax, view.W);
		hmax = std::max(hmax, view.H);
	}
	std::vector<int64_t> image_shape = { hmax, wmax };

	PyramidEmbeddingDataloaderProperties pyramid_cfg;
	pyramid_cfg.TileSizeRange = { LeRFParams.pyr_tile_size_min, LeRFParams.pyr_tile_size_max };	//Pyramid scales follow the original LeRF defaults (kerrj/lerf): absolute tile_size_range as a fraction of image size
	pyramid_cfg.TileSizeRes = LeRFParams.pyr_tile_size_res;			//tile_size_res linear levels; values come from NeRFExecutorParams config via LeRFDatasetParams
	pyramid_cfg.StrideScaler = LeRFParams.pyr_stride_scaler;	//Per-level stride_ratio is interpolated by PyramidEmbeddingDataloader::StrideScaler() over ratio range [0.05, 0.15] -> [1.0, StrideScaler].
	pyramid_cfg.ImageShape = image_shape;
	pyramid_cfg.ModelName = "pyramid";

	try {
		PyramidClipEmbedding = std::make_unique<PyramidEmbeddingDataloader>(pyramid_cfg, Device, clip, clip_processor, Params, LeRFParams.PyramidClipEmbeddingSaveDir);
		std::cout << "PyramidEmbeddingDataloader initialized successfully." << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "Failed to initialize PyramidEmbeddingDataloader: " << e.what() << std::endl;
		throw;
	}
}


///
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NeRFDataset :: GetRayBatch(
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

	//Вычисляем угловой размер пикселя в мировых координатах. Производная радиуса конуса = (размер пикселя) / (фокусное расстояние)
	//Размер пикселя в мировых координатах на расстоянии 1
	auto pixel_size_x = 1.0 / fx;
	auto pixel_size_y = 1.0 / fy;
	auto avg_pixel_size = (pixel_size_x + pixel_size_y) / 2.0;
	//Производная радиуса конуса = угловой размер пикселя. При расстоянии d, радиус конуса cone_ridius = d * pixel_size
	auto cone_angle = avg_pixel_size /** torch::ones({ h, w, 1 }, rays_d.options())*/;

	return { rays_o, rays_d, cone_angle };
}


///
NeRFDataExample NeRFDataset :: get_batch(std::vector<size_t> request/*Не используется*/)
{
	torch::Tensor pose = Params.Views[CurrentImageIdx].Pose.to(Device);		//Получаем позу для текущего изображения
	auto [h_start, h_end, w_start, w_end] = CalculateBounds();		//Вычисляем границы кадрирования
	//Прямая генерация случайных координат
	auto options = torch::TensorOptions().dtype(torch::kLong).device(Device);
	torch::Tensor rand_h = torch::randint(h_start, h_end + 1, { BatchSize }, options);
	torch::Tensor rand_w = torch::randint(w_start, w_end + 1, { BatchSize }, options);
	torch::Tensor target_s = CurrentImage.index({ rand_h, rand_w });		//Извлекаем цвета пикселей
	auto [rays_o, rays_d, cone_angle] = GetRayBatch(rand_h, rand_w, Params.Views[CurrentImageIdx].H, Params.Views[CurrentImageIdx].W, Params.Views[CurrentImageIdx].K.to(Device), pose);		//Вычисляем лучи

	//!!!->class LeRFDataset
	///Compute CLIP embeddings at the image points that landed in this batch
	torch::Tensor target_lang_embedding,
		target_lerf_scale;
	if (LeRFParams.UseLerf && PyramidClipEmbedding)
	{
		//All views are resized to Config.ImageShape when the pyramid is built, so native view coords must be rescaled accordingly
		const auto &pyr_shape = PyramidClipEmbedding->GetImageShape();
		float scale_h = static_cast<float>(pyr_shape[0]) / static_cast<float>(Params.Views[CurrentImageIdx].H);
		float scale_w = static_cast<float>(pyr_shape[1]) / static_cast<float>(Params.Views[CurrentImageIdx].W);

		//Points for the pyramid: [B, 3] = (img_id, x=row, y=col) - same order as in PatchEmbeddingDataloader::operator()
		auto coords = torch::stack({torch::full({ BatchSize }, static_cast<float>(CurrentImageIdx), torch::kFloat32).to(Device), rand_h.to(torch::kFloat32) * scale_h, rand_w.to(torch::kFloat32) * scale_w }, -1);

		//Scale is sampled randomly per point (as in the original LeRF): an undefined tensor routes to RandomScales
		auto [lang_embed_batch, scales_out] = (*PyramidClipEmbedding)(coords, torch::Tensor());
		//scales_out holds the per-point tile-ratio scale; normalize it to [0,1] across the pyramid range so the field and renderer work in normalized pyramid-scale units
		const auto &tile_sizes = PyramidClipEmbedding->GetTileSizesCPU();
		const float t_min = tile_sizes[0].item<float>();
		const float t_max = tile_sizes[tile_sizes.size(0) - 1].item<float>();
		target_lerf_scale = (scales_out - t_min) / std::max(t_max - t_min, 1e-6f);

		target_lang_embedding = lang_embed_batch/*.to(torch::kFloat32) done in PyramidClipEmbedding*/;
	} //if use_lerf

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
	return { {rays_o, rays_d, cone_angle, Params.Views[CurrentImageIdx].Near, Params.Views[CurrentImageIdx].Far}, {target_s, target_lang_embedding, target_lerf_scale,
		static_cast<float>(Params.Views[CurrentImageIdx].H), static_cast<float>(Params.Views[CurrentImageIdx].K[1][1].item<float>())} };
}