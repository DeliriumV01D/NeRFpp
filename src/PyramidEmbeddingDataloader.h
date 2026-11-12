#pragma once
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <filesystem>
#include <string>
#include "PatchEmbeddingDataloader.h"
#include "NeRFDatasetParams.h"

struct PyramidEmbeddingDataloaderProperties {
	std::array<float, 2> TileSizeRange; // [min_ratio, max_ratio]
	int64_t TileSizeRes;
	float StrideScaler;
	std::vector<int64_t> ImageShape;
	std::string ModelName;

	nlohmann::json ToJson() const
	{
		nlohmann::json cfg;
		cfg["tile_size_range"] = { TileSizeRange[0], TileSizeRange[1] };
		cfg["tile_size_res"] = TileSizeRes;
		cfg["stride_scaler"] = StrideScaler;
		cfg["image_shape"] = ImageShape;
		cfg["model_name"] = ModelName;
		return cfg;
	}
};


///
class PyramidEmbeddingDataloader {
protected:
	torch::Device Device;
	CLIP Clip;
	std::shared_ptr<RuCLIPProcessor> ClipProcessor;
	NeRFDatasetParams DatasetParams;
	PyramidEmbeddingDataloaderProperties Config;
	std::filesystem::path CachePath;
	std::vector<std::shared_ptr<PatchEmbeddingDataloader>> Levels;     //Пирамида уровней (lazy loading)
	//std::unordered_map<int64_t, torch::Tensor> FeatureCache;         //Кэш загруженных данных признаков для текущего батча
	torch::Tensor TileSizes;                                           // [num_levels] //Масштабы: torch::Tensor, float32, на CPU

	//Вычисленные stride_scalers для каждого уровня
	std::vector<float> StriderScalerList;

	int64_t NumLevels;
	float MinScale,
		MaxScale,
		ScaleStep; // для интерполяции scale -> bin

	//Вычисление stride_ratio по tile_ratio: np.interp([0.05, 0.15], [1.0, stride_scaler])
	float StrideScaler(float tile_ratio) const;

	////Базовая интерполяция (без нормализации и scale output)
	//std::pair<torch::Tensor, torch::Tensor> InterpImpl(const torch::Tensor &img_points, const torch::Tensor &scale_or_null = {});
	std::pair<torch::Tensor, torch::Tensor> RandomScales(const torch::Tensor &img_points);
	std::pair<torch::Tensor, torch::Tensor> UniformScales(const torch::Tensor &img_points, const torch::Tensor &scales);
public:
	PyramidEmbeddingDataloader(
		const PyramidEmbeddingDataloaderProperties &cfg,
		torch::Device device,
		CLIP model,
		std::shared_ptr<RuCLIPProcessor> clip_processor,
		const NeRFDatasetParams &dataset_params,
		const std::filesystem::path &cache_path
	);

	///(B, 3) -> (B, embed_size), scale=randomly selected ИЛИ (B, 3), scale: scalar or [B] -> (B, embed_size)
	std::pair<torch::Tensor, torch::Tensor> operator()(const torch::Tensor &img_points, const torch::Tensor &scale);

	///(x, y, scale, img_id) -> (embed_dim,)
	///x — координата вдоль ImageShape[0] (строка), y — вдоль ImageShape[1] (столбец); scale — реальный масштаб из [MinScale, MaxScale]
	torch::Tensor GetPixelValue(
		float x,
		float y,
		float scale,
		int img_id
	);

	void Load();
	void Create();
	void Save();

	int64_t GetNumLevels() const { return NumLevels; }
	const torch::Tensor &GetTileSizesCPU() const { return TileSizes; }
	///Форма изображения, к которой приведены все вью при построении уровней: (H, W)
	const std::vector<int64_t> & GetImageShape() const { return Config.ImageShape; }

	///Построить массив масштабов: linspace(min, max, num_levels)
	static std::vector<float> GenerateScaleLevels(const float min_scale, const float max_scale, const int64_t num_levels);
};
