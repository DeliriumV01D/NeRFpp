#include "PatchEmbeddingDataloader.h"
#include <future>

///Интерполяция признаков
torch::Tensor PatchEmbeddingDataloader :: InterpInds(
	const torch::Tensor &img_ind,
	const torch::Tensor &x_ind,
	const torch::Tensor &y_ind,
	const torch::Tensor &img_points_x,
	const torch::Tensor &img_points_y
){
	auto img_ind_dev = img_ind.to(data_.device()); //data на CPU

	//Индексация: data_[img_ind, x_ind, y_ind] и соседи
	auto topleft = data_.index({ img_ind_dev, x_ind, y_ind }); // (B, embed_size)
	auto topright = data_.index({ img_ind_dev, x_ind + 1, y_ind });
	auto botleft = data_.index({ img_ind_dev, x_ind, y_ind + 1 });
	auto botright = data_.index({ img_ind_dev, x_ind + 1, y_ind + 1 });

	//Перемещаем на Device (GPU) для дальнейших вычислений
	topleft = topleft.to(Device);
	topright = topright.to(Device);
	botleft = botleft.to(Device);
	botright = botright.to(Device);

	//Веса
	auto x_stride = stride_;
	auto y_stride = stride_;
	auto right_w = ((img_points_x - center_x_.index({ x_ind })) / x_stride).to(Device); // (B)
	auto bot_w = ((img_points_y - center_y_.index({ y_ind })) / y_stride).to(Device); // (B)

	//Горизонтальная интерполяция
	auto top = torch::lerp(topleft, topright, right_w.unsqueeze(1).to(topleft.scalar_type()));
	auto bot = torch::lerp(botleft, botright, right_w.unsqueeze(1).to(botleft.scalar_type()));

	//Вертикальная интерполяция
	return torch::lerp(top, bot, bot_w.unsqueeze(1).to(top.scalar_type()));	//weight must match start/end dtype (Half), same cast as right_w above
}


///Извлечение и кодирование патчей одного изображения
torch::Tensor PatchEmbeddingDataloader :: EmbedClipTiles(
	const torch::Tensor &image, // (1, 3, H, W) на CPU?
	const torch::nn::functional::UnfoldFuncOptions &unfold_options
){
	auto patches = torch::nn::functional::unfold(image, unfold_options); // (1, 3*k*k, num_patches)
	patches = patches.squeeze(0).transpose(0, 1); // (num_patches, 3*k*k)

	//Преобразуем в (num_patches, 3, k, k)
	int64_t num_patches = patches.size(0);
	patches = patches.reshape({ num_patches, 3, kernel_size_, kernel_size_ });

	const int64_t clip_res = ClipProcessor->GetImageSize();
	if (kernel_size_ != clip_res)
		patches = torch::nn::functional::interpolate(patches, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{ clip_res, clip_res }).mode(torch::kBilinear).align_corners(false));

	//Переносим на GPU для модели
	patches = patches.to(Device);

	//Прогоняем через модель
	torch::NoGradGuard no_grad;

	auto input = ClipProcessor->operator()(std::vector <std::string>(), patches);
	auto clip_embeds = Clip->EncodeImage(input.second/*.to(Device)*/); // (num_patches, embed_size)
	//Нормализация
	clip_embeds = clip_embeds / clip_embeds.norm(2/*L2*/, 1, true);

	//Преобразование в сетку (num_centers_x, num_centers_y, embed_size)
	int64_t nx = center_x_.size(0);
	int64_t ny = center_y_.size(0);
	clip_embeds = clip_embeds.reshape({ nx, ny, -1 });

	//Добавляем дополнительную строку и столбец (копированием последних) extra row и extra column. Это стандартный трюк: при bilinear‑интерполяции нам нужно иметь доступ к ячейкам (x+1, y) и (x, y+1) даже для крайних патчей; дублирование гарантирует, что индексы никогда не выйдут за границы.
	auto last_col = clip_embeds.slice(1, ny - 1, ny); // (nx, 1, embed_size)
	clip_embeds = torch::cat({ clip_embeds, last_col }, 1); // (nx, ny+1, embed_size)

	auto last_row = clip_embeds.slice(0, nx - 1, nx); // (1, ny+1, embed_size)
	clip_embeds = torch::cat({ clip_embeds, last_row }, 0); // (nx+1, ny+1, embed_size)

	return clip_embeds.detach().cpu();
}


PatchEmbeddingDataloader :: PatchEmbeddingDataloader(
	const PatchEmbeddingDataloaderProperties &cfg,
	torch::Device device,
	CLIP model,
	std::shared_ptr<RuCLIPProcessor> clip_processor,
	const NeRFDatasetParams &dataset_params,
	const std::filesystem::path &cache_path
) : Device(device), Clip(model), ClipProcessor(clip_processor), Cfg(cfg), CachePath(cache_path), DatasetParams(dataset_params)
{
	kernel_size_ = static_cast<int64_t>(Cfg.ImageShape[0] * Cfg.TileRatio);
	stride_ = static_cast<int64_t>(kernel_size_ * Cfg.StrideRatio);
	padding_ = kernel_size_ / 2;

	//Вычисление количества центров по каждой оси
	int64_t num_centers_x = static_cast<int64_t>(std::floor(
		(Cfg.ImageShape[0] + 2 * padding_ - (kernel_size_ - 1) - 1) / static_cast<double>(stride_) + 1
	));
	int64_t num_centers_y = static_cast<int64_t>(std::floor(
		(Cfg.ImageShape[1] + 2 * padding_ - (kernel_size_ - 1) - 1) / static_cast<double>(stride_) + 1
	));

	//Генерация координат центров
	auto center_x = torch::arange(num_centers_x, torch::kFloat64) * stride_ + (kernel_size_ - 1) / 2.0 - padding_;
	auto center_y = torch::arange(num_centers_y, torch::kFloat64) * stride_ + (kernel_size_ - 1) / 2.0 - padding_;

	center_x_ = center_x.to(torch::kFloat32);
	center_y_ = center_y.to(torch::kFloat32);
	start_x_ = center_x_[0].to(torch::kFloat).item<float>();
	start_y_ = center_y_[0].to(torch::kFloat).item<float>();

	//Если есть путь к кэшу и файл существует, загружаем данные
	if (!cache_path.empty() && std::filesystem::exists(cache_path))
	{
		std::cout << "PatchEmbeddingDataloader: tring to load embeddings from cache " << cache_path << std::endl;
		Load();
	} else {
		std::cout << "PatchEmbeddingDataloader: calculating embeddings" << std::endl;
		//Иначе создаем признаки из image_list
		if (DatasetParams.Views.empty())
			throw std::runtime_error("PatchEmbeddingDataloader error: Either cache path with existing file or image list must be provided");
		Create();
	}
}			//PatchEmbeddingDataloader


///Загрузка из кэша на диске
void PatchEmbeddingDataloader :: Load()
{
	std::filesystem::path cache_info_path = CachePath;
	cache_info_path.replace_extension(".info");
	if (!std::filesystem::exists(cache_info_path))
		throw std::runtime_error("PatchEmbeddingDataloader::Load() error: Cache info file not found");

	std::ifstream f(cache_info_path);
	nlohmann::json cfg = nlohmann::json::parse(f);
	if (cfg != Cfg.ToJson())
		throw std::runtime_error("PatchEmbeddingDataloader::Load() error: Config mismatch");

	torch::load(data_, CachePath.string());
	if (data_.dtype() != torch::kHalf)
		data_ = data_.to(torch::kHalf);
	//Перемещаем на CPU (уже должно быть на CPU после загрузки)
	data_ = data_.cpu();
}			//Load


///Создание признаков изображений
void PatchEmbeddingDataloader :: Create()
{
	if (!Clip)
		throw std::runtime_error("PatchEmbeddingDataloader::Create() error: Model must be provided to generate features");

	if (!ClipProcessor)
		throw std::runtime_error("PatchEmbeddingDataloader::Create() error: Processor must be provided to generate features");

	if (DatasetParams.Views.empty())
		throw std::runtime_error("PatchEmbeddingDataloader::Create() error: image list must be provided to generate features");

	auto unfold_options = torch::nn::functional::UnfoldFuncOptions({ kernel_size_, kernel_size_ }).stride({ stride_, stride_ }).padding({ padding_, padding_ });

	const int64_t N = static_cast<int64_t>(DatasetParams.Views.size());
	std::vector<torch::Tensor> img_embeds;
	img_embeds.reserve(N);

	//Асинхронная предвыборка: декодирование и ресайз следующего изображения идут параллельно с кодированием текущего на GPU
	auto load_async = [this](int64_t idx) {
		return std::async(std::launch::async, [this, idx]() -> cv::Mat {
			cv::Mat m = cv::imread(DatasetParams.Views[idx].ImagePath.string(), cv::IMREAD_COLOR);
			if (m.empty())
				throw std::runtime_error("PatchEmbeddingDataloader::Create(): failed to load image " + DatasetParams.Views[idx].ImagePath.string());

			//Все вью должны совпадать с ImageShape пирамиды — иначе сетки центров различаются и stack не соберётся
			cv::Size target(static_cast<int>(Cfg.ImageShape[1]), static_cast<int>(Cfg.ImageShape[0])); //cv::Size(w, h)
			if (m.size() != target)
				cv::resize(m, m, target, 0, 0, cv::INTER_AREA);
			return m;
		});
	};

	std::future<cv::Mat> next = load_async(0);
	for (int64_t i = 0; i < N; ++i)
	{
		cv::Mat img_mat = next.get();
		if (i + 1 < N)
			next = load_async(i + 1); //пока текущее изображение кодируется CLIP'ом, уже грузится следующее

		//CVMatToTorchTensor already adds the batch dimension -> (1, 3, H, W); a second unsqueeze would give a 5D tensor and break unfold/reshape
		//convert_to_rgb = true: CLIP normalization constants are RGB-ordered while OpenCV decodes to BGR
		torch::Tensor img = CVMatToTorchTensor(img_mat, true, true); //stays on CPU; the move to GPU happens inside EmbedClipTiles
		auto embed = EmbedClipTiles(img, unfold_options);

		img_embeds.push_back(embed);
	}
	//Стек в один тензор (N, num_centers_x+1, num_centers_y+1, embed_size)
	data_ = torch::stack(img_embeds, 0).to(torch::kHalf).cpu();

	//Сохраняем в кэш, если указан путь
	if (!CachePath.empty())
	{
		torch::save(data_, CachePath.string());
		std::filesystem::path cache_info_path = CachePath;
		cache_info_path.replace_extension(".info");
		std::ofstream f(cache_info_path);
		f << Cfg.ToJson().dump(4);
	}
}			//Create


///(B, 3) -> (B, embed_size)
torch::Tensor PatchEmbeddingDataloader :: operator()(const torch::Tensor &img_points)
{
	//img_points: (B, 3) = [img_ind, x, y]
	auto img_points_cpu = img_points.cpu();
	auto img_ind = img_points_cpu.select(1, 0).to(torch::kLong);	//image index must be integer
	auto img_points_x = img_points_cpu.select(1, 1);
	auto img_points_y = img_points_cpu.select(1, 2);

	auto x_ind = torch::floor((img_points_x - start_x_) / stride_).to(torch::kLong);
	auto y_ind = torch::floor((img_points_y - start_y_) / stride_).to(torch::kLong);

	x_ind = x_ind.clamp(0, center_x_.size(0) - 1);
	y_ind = y_ind.clamp(0, center_y_.size(0) - 1);

	return InterpInds(img_ind, x_ind, y_ind, img_points_x, img_points_y);
}			//operator()