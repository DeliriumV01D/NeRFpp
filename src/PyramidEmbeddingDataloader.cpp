#include "PyramidEmbeddingDataloader.h"
#include <torch/nn/functional/pooling.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>


///Построить массив масштабов: linspace(min, max, num_levels)
std::vector<float> PyramidEmbeddingDataloader :: GenerateScaleLevels(const float min_scale, const float max_scale, const int64_t num_levels)
{
	std::vector<float> levels;
	if (num_levels <= 0)
		return levels;
	levels.reserve(static_cast<size_t>(num_levels));
	if (num_levels == 1)
	{
		levels.push_back(min_scale);
		return levels;
	}
	for (int64_t i = 0; i < num_levels; ++i)
		levels.push_back(min_scale + (max_scale - min_scale) * static_cast<float>(i) / static_cast<float>(num_levels - 1));
	return levels;
}

///
PyramidEmbeddingDataloader :: PyramidEmbeddingDataloader(
	const PyramidEmbeddingDataloaderProperties &config,
	torch::Device device,
	CLIP model,
	std::shared_ptr<RuCLIPProcessor> clip_processor,
	const NeRFDatasetParams &dataset_params,
	const std::filesystem::path &cache_path
) : Device(device), Clip(model), ClipProcessor(clip_processor), CachePath(cache_path), Config(config), DatasetParams(dataset_params)
{
	//Построить массив масштабов: linspace(min, max, num_levels)
	std::vector<float> level_list = GenerateScaleLevels(config.TileSizeRange[0], config.TileSizeRange[1], config.TileSizeRes);
	TileSizes = torch::from_blob(level_list.data(), { static_cast<int64_t>(level_list.size()) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone(); // [num_levels]

	NumLevels = static_cast<int64_t>(TileSizes.size(0));
	MinScale = TileSizes[0].item<float>();
	MaxScale = TileSizes[TileSizes.size(0) - 1].item<float>();
	//No scale step for a single level - otherwise division by zero
	ScaleStep = NumLevels > 1 ? (MaxScale - MinScale) / static_cast<float>(NumLevels - 1) : 0.f;

	//Вычислить stride_scaler для каждого уровня
	StriderScalerList.reserve(NumLevels);
	for (int64_t i = 0; i < NumLevels; ++i)
	{
		float tr = TileSizes[i].item<float>();
		StriderScalerList.push_back(StrideScaler(tr));
	}

	//Попытаться загрузить или создать
	if (!CachePath.empty() && std::filesystem::exists(CachePath / "info.json"))
	{
		try {
			Load();
		} catch (...) {
			//Если загрузка не удалась — пересоздать
			Levels.clear();
			Create();
		}
	} else {
		if (DatasetParams.Views.empty())
			throw std::runtime_error("PyramidEmbeddingDataloader: either cache or image list required");
		Create();
	}
}


///
float PyramidEmbeddingDataloader :: StrideScaler(float tile_ratio) const
{
	float t_min = 0.05f, t_max = 0.15f;
	if (tile_ratio <= t_min) return 1.0f;
	if (tile_ratio >= t_max) return Config.StrideScaler;
	float t_norm = (tile_ratio - t_min) / (t_max - t_min);
	return 1.0f + t_norm * (Config.StrideScaler - 1.0f);
}


///
void PyramidEmbeddingDataloader :: Create()
{
	if (!Clip) throw std::runtime_error("PyramidEmbeddingDataloader::Create: model not set");
	if (!ClipProcessor) throw std::runtime_error("PyramidEmbeddingDataloader::Create: processor not set");
	if (DatasetParams.Views.empty()) throw std::runtime_error("PyramidEmbeddingDataloader::Create: no images");

	//Создаем папку кэша
	if (!CachePath.empty())
		std::filesystem::create_directories(CachePath);

	Levels.clear();
	Levels.reserve(NumLevels);

	for (int64_t i = 0; i < NumLevels; i++)
	{
		float tr = TileSizes[i].item<float>();
		float stride_ratio = StriderScalerList[i];

		//Формируем конфиг для уровня
		PatchEmbeddingDataloaderProperties level_cfg;
		level_cfg.TileRatio = tr;
		level_cfg.StrideRatio = stride_ratio;
		level_cfg.ImageShape = Config.ImageShape;
		level_cfg.ModelName = Config.ModelName;

		std::filesystem::path level_path;
		if (!CachePath.empty())
			level_path = CachePath / ("level_" + std::to_string(i) + ".npy");

		auto embed_loader = std::make_unique<PatchEmbeddingDataloader>(level_cfg, Device, Clip, ClipProcessor, DatasetParams, level_path);

		Levels.push_back(std::move(embed_loader));
	}

	//Сохраняем metadata
	Save();
}


///
void PyramidEmbeddingDataloader :: Save()
{
	if (CachePath.empty()) return;
	std::filesystem::path info_path = CachePath / "info.json";
	std::ofstream f(info_path);
	f << Config.ToJson().dump(4);
}

///
void PyramidEmbeddingDataloader :: Load()
{
	if (CachePath.empty()) throw std::runtime_error("PyramidEmbeddingDataloader::Load: no cache path");

	std::filesystem::path info_path = CachePath / "info.json";
	if (!std::filesystem::exists(info_path))
		throw std::runtime_error("PyramidEmbeddingDataloader::Load: info.json not found");

	//Проверяем конфиг
	std::ifstream f(info_path);
	nlohmann::json cfg_json = nlohmann::json::parse(f);
	if (cfg_json != Config.ToJson())
	{
		//Конфиг отличается — удаляем всё и бросаем исключение
		for (const auto &entry : std::filesystem::directory_iterator(CachePath))
		{
			if (entry.is_regular_file() && entry.path().extension() == ".npy")
				std::filesystem::remove(entry.path());
		}
		throw std::runtime_error("PyramidEmbeddingDataloader::Load: config mismatch, cache cleared");
	}

	//Создаём PatchEmbeddingDataloader'ы "вручную", чтобы избежать повторного Create
	Levels.clear();
	Levels.reserve(NumLevels);

	for (int64_t i = 0; i < NumLevels; ++i)
	{
		float tr = TileSizes[i].item<float>();
		float stride_ratio = StriderScalerList[i];

		PatchEmbeddingDataloaderProperties level_cfg;
		level_cfg.TileRatio = tr;
		level_cfg.StrideRatio = stride_ratio;
		level_cfg.ImageShape = Config.ImageShape;
		level_cfg.ModelName = Config.ModelName;

		std::filesystem::path level_path = CachePath / ("level_" + std::to_string(i) + ".npy");

		auto embed_loader = std::make_unique<PatchEmbeddingDataloader>(level_cfg, Device, Clip, ClipProcessor, DatasetParams, level_path);

		////принудительно вызываем Load() внутри, чтобы убедиться в наличии файла
		//try {
		//	// Внутри PatchEmbeddingDataloader::PatchEmbeddingDataloader уже есть логика Load/Create - если файл есть - load() - если нет - создаст заново
		//} catch (...) {
		//	throw std::runtime_error("PyramidEmbeddingDataloader::Load: failed to initialize level " + std::to_string(i));
		//}

		Levels.push_back(std::move(embed_loader));
	}
}


///
std::pair<torch::Tensor, torch::Tensor> PyramidEmbeddingDataloader :: operator()(const torch::Tensor &img_points, const torch::Tensor &scale)
{
	if (scale.defined() && scale.numel() != 0)
		return UniformScales(img_points, scale);
	/*else*/
	return RandomScales(img_points);
}

///(x, y, scale, img_id) -> (embed_dim,)
///x is the coordinate along ImageShape[0] (row), y along ImageShape[1] (column); scale is a real scale from [MinScale, MaxScale]
torch::Tensor PyramidEmbeddingDataloader :: GetPixelValue(float x, float y, float scale, int img_id)
{
	//Coordinate order matches PatchEmbeddingDataloader: (img_id, x=row, y=col)
	//The whole query pipeline expects a 2-D (B, 3) batch - a bare 1-D [3] point breaks select(1, k) downstream
	auto img_point = torch::tensor({ { static_cast<float>(img_id), x, y } }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	auto [embed, scales_out] = this->operator()(img_point, torch::scalar_tensor(scale, torch::kFloat32));
	//The pyramid stores features in half; callers (Relevancy / LeRFRenderer) work with float32
	//Keep the batch dim: Relevancy's torch::mm requires a 2-D (1, D) image embedding
	return embed.to(torch::kFloat32);
}

///
std::pair<torch::Tensor, torch::Tensor> PyramidEmbeddingDataloader :: RandomScales(const torch::Tensor &img_points)
{
	//Специальный случай: только один уровень
	if (NumLevels == 1)
	{
		auto embed = Levels[0]->operator()(img_points.to(Device));
		torch::Tensor scales_out = torch::full({ img_points.size(0), 1 }, MinScale, torch::TensorOptions().dtype(torch::kFloat32).device(Device));
		return { embed, scales_out };
	}

	auto B = img_points.size(0);
	auto D = Levels[0]->GetData().size(-1); //embed_size

	//Normalize input device once: ids below are CUDA, indexing a CPU img_points would throw
	torch::Tensor pts = img_points.to(Device);

	//Случайный bin: int64 [B], от 0 до NumLevels - 2
	torch::Tensor random_bin = torch::randint(/*low=*/0, /*high=*/NumLevels - 1, { B }, torch::TensorOptions().dtype(torch::kInt64).device(Device));

	//Случайный вес: float16 [B]
	torch::Tensor random_w = torch::rand({ B }, torch::TensorOptions().dtype(torch::kFloat16).device(Device));

	//Подготовка выходов
	torch::Tensor bottom_interp = torch::zeros({ B, D }, torch::TensorOptions().dtype(torch::kHalf).device(Device));
	torch::Tensor top_interp = torch::zeros({ B, D }, torch::TensorOptions().dtype(torch::kHalf).device(Device));

	//Цикл по уровням (нужно оптимизировать в будущем)
	for (int64_t level = 0; level < NumLevels - 1; level++) 
	{
		//Индексы точек, для которых этот уровень "нижний"
		torch::Tensor mask = (random_bin == level);
		if (!mask.any().item<bool>()) continue;

		//Получаем индексы подмножества
		torch::Tensor ids = mask.nonzero().squeeze(1); // [M]

		//Берём точки, соответствующие этим индексам (pts is already on Device - ids are CUDA)
		torch::Tensor points_slice = pts.index({ ids });

		//Вычисляем эмбеддинги
		auto bottom_embed = Levels[level]->operator()(points_slice).to(Device);
		auto top_embed = Levels[level + 1]->operator()(points_slice).to(Device);

		//Присваиваем в выходные тензоры
		bottom_interp.index_put_({ ids }, bottom_embed);
		top_interp.index_put_({ ids }, top_embed);
	}

	//Линейная интерполяция: lerp(bottom, top, w)
	torch::Tensor result = torch::lerp(bottom_interp, top_interp, random_w.unsqueeze(1)).to(torch::kFloat32);

	//Вычисляем scale для возврата: scale = MinScale + (bin + w) * ScaleStep
	torch::Tensor scales_out = (MinScale + (random_bin.to(torch::kFloat32) + random_w.to(torch::kFloat32)) * ScaleStep).unsqueeze(1);

	//Нормализация — в оригинале только для UniformScales, но добавим и сюда
	result = result / result.norm(/*p=*/2, /*dim=*/-1, /*keepdim=*/true).clamp_min(1e-8f);

	return { result, scales_out };
}


///
std::pair<torch::Tensor, torch::Tensor> PyramidEmbeddingDataloader :: UniformScales(const torch::Tensor &img_points, const torch::Tensor &scale)
{
	//Специальный случай: только один уровень
	if (NumLevels == 1)
	{
		auto embed = Levels[0]->operator()(img_points.to(Device));
		torch::Tensor scales_out = torch::full({ img_points.size(0), 1 }, MinScale, torch::TensorOptions().dtype(torch::kFloat32).device(Device));
		return { embed, scales_out };
	}

	auto B = img_points.size(0);
	if (scale.numel() != 1 && scale.numel() != B)
		throw std::runtime_error("PyramidEmbeddingDataloader: uniform_scales scale must be scalar or [B]");

	//Приводим scales к [B] (если скаляр)
	torch::Tensor s = scale;
	if (s.numel() == 1)
		s = s.expand({ B });
	s = s.to(Device).to(torch::kFloat32);

	torch::Tensor pts = img_points.to(Device);

	//Normalize scale into [0, num_levels-1]
	torch::Tensor norm_scales = (s - MinScale) / std::max(MaxScale - MinScale, 1e-8f);
	torch::Tensor level_f = norm_scales * (NumLevels - 1);

	//floor -> int64 and clamp to [0, num_levels-2] (otherwise level+1 goes out of range)
	torch::Tensor bin = torch::floor(level_f).to(torch::kInt64);
	bin = torch::clamp(bin, 0, NumLevels - 2);

	//Weight inside the level: (scale - level_start) / (level_end - level_start)
	torch::Tensor level_vals = TileSizes.to(Device).to(torch::kFloat32);
	torch::Tensor l_start = level_vals.index({ bin });
	torch::Tensor l_end = level_vals.index({ bin + 1 });
	torch::Tensor weight = (s - l_start) / (l_end - l_start).clamp_min(1e-8f);

	int64_t D = Levels[0]->GetData().size(-1); //embed_size
	torch::Tensor result = torch::zeros({ B, D }, torch::TensorOptions().dtype(torch::kFloat32).device(Device));

	//Group points by bin: one call per level instead of B GPU syncs (same strategy as RandomScales)
	for (int64_t level = 0; level < NumLevels - 1; ++level)
	{
		torch::Tensor mask = (bin == level);
		if (!mask.any().item<bool>()) continue;

		torch::Tensor ids = mask.nonzero().squeeze(1); // [M]
		torch::Tensor points_slice = pts.index({ ids });

		auto e0 = Levels[level]->operator()(points_slice).to(torch::kFloat32);
		auto e1 = Levels[level + 1]->operator()(points_slice).to(torch::kFloat32);
		result.index_put_({ ids }, torch::lerp(e0, e1, weight.index({ ids }).unsqueeze(1)));
	}

	//Normalize (clamp_min guards against NaN on zero vectors)
	result = result / result.norm(/*p=*/2, /*dim=*/-1, /*keepdim=*/true).clamp_min(1e-8f);

	return { result.to(torch::kHalf), s };
}