#pragma once

#include <opencv2/opencv.hpp>

#include "RuCLIP.h"
#include "RuCLIPProcessor.h"
#include "NeRFDatasetParams.h"
#include "NeRFRenderer.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include "json.hpp"

struct PatchEmbeddingDataloaderProperties {
	float TileRatio{};                      //tile_size/image_size
	float StrideRatio{};                    //0.5
	std::vector<int64_t> ImageShape;        //H,W
	std::string ModelName;

	nlohmann::json ToJson() const
	{
		nlohmann::json cfg;
		cfg["TileRatio"] = TileRatio;
		cfg["StrideRatio"] = StrideRatio;
		cfg["ImageShape"] = ImageShape;
		cfg["ModelName"] = ModelName;
		return cfg;
	}
};

///
class PatchEmbeddingDataloader {
protected:
	torch::Device Device{ torch::kCUDA };

	CLIP Clip = nullptr;
	std::shared_ptr<RuCLIPProcessor> ClipProcessor = nullptr;

	//torch::Tensor image_list_;  // (N, 3, H, W) на CPU?
	std::filesystem::path CachePath;
	PatchEmbeddingDataloaderProperties Cfg;
	NeRFDatasetParams DatasetParams;

	int64_t kernel_size_,
		stride_,
		padding_;
	torch::Tensor center_x_, //(num_centers_x,) half
		center_y_; //(num_centers_y,) half
	float start_x_,  //float
		start_y_;  //float

	torch::Tensor data_; // (num_images, num_centers_x+1, num_centers_y+1, embed_size) half, cpu


	///Интерполяция признаков
	torch::Tensor InterpInds(
		const torch::Tensor &img_ind,
		const torch::Tensor &x_ind,
		const torch::Tensor &y_ind,
		const torch::Tensor &img_points_x,
		const torch::Tensor &img_points_y
	);

	///Извлечение и кодирование патчей одного изображения
	torch::Tensor EmbedClipTiles(
		const torch::Tensor &image, // (1, 3, H, W) на CPU?
		const torch::nn::functional::UnfoldFuncOptions &unfold_options
	);

public:
	PatchEmbeddingDataloader(
		const PatchEmbeddingDataloaderProperties &cfg,
		torch::Device device,
		CLIP model,
		std::shared_ptr<RuCLIPProcessor> clip_processor,
		const NeRFDatasetParams &dataset_params,
		const std::filesystem::path &cache_path
	);

	///Загрузка из кэша на диске
	void Load();

	///Создание признаков изображений
	void Create();

	///(B, 3) -> (B, embed_size)
	torch::Tensor operator()(const torch::Tensor &img_points);

	///
	torch::Tensor GetData() const { return data_; }
};