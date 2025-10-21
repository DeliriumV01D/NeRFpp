#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "RuCLIP.h"
#include "RuCLIPProcessor.h"
#include "TRandomInt.h"

#include "NeRFDatasetParams.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <tuple>

///
struct PyramidEmbedderProperties 
{
	cv::Size ImgSize {0, 0};	//Входной размер изображения сети
	float Overlap {0.75};			///Доля перекрытия
	int MaxZoomOut{ 1 },				///Максимальное удаление (h, w) = (h_base, w_baser) * pow(2, zoom_out);		//-1, 0 , 1, 2...
		MinZoomOut{1};
};


///
class PyramidEmbedding {
protected:
	///{hor_pos_idx, vert_pos_idx, zoom_out_idx, data_img_id, x, y}
	std::list<std::tuple<int, int, int, int, float, float>> GetNearestPatchIndicesSingleScale(
		const float x,
		const float y,
		const int zoom_out_idx,
		const int data_img_id,
		const PyramidEmbedderProperties &properties,
		const cv::Size &img_size
	);

	///!!!Должно быть согласовано с GetNextSample(const CompactData &data)
	std::list<std::tuple<int, int, int, int, float, float>>  GetNearestPatchIndicesMultiScale(
		const float x,
		const float y,
		const float scale,
		const int data_img_id,
		const PyramidEmbedderProperties &properties,
		const cv::Size &img_size
	);

	torch::Tensor Interpolate(
		const int hor_pos_idx1, const int hor_pos_idx2,
		const int vert_pos_idx1, const int vert_pos_idx2,
		const int zoom_out_idx1, const int zoom_out_idx2,
		const int data_img_id,
		const float x1, const float x2, const float y1, const float y2,
		const float x,
		const float y,
		const cv::Size &img_size
	);	
public:
	///{hor_pos_idx, vert_pos_idx, zoom_out_idx, data_img_id}, {features}
	std::map<std::tuple<int, int, int, int>, torch::Tensor> Embeddings;

	///Процедуру вычисления эмбединга для каждого пикселя в зависимости от масштаба 
	///трилинейная интерполяция между центрами патчей на ближайших к скейлу масштабах покрывает все случаи
	torch::Tensor GetPixelValue(
		const float x,
		const float y,
		const float scale,
		const int data_img_id,
		const PyramidEmbedderProperties &properties,
		const cv::Size &img_size
	);

	void Save(const std::filesystem::path &data_file);		//itemName + ".pt
	void Load(const std::filesystem::path &data_file);
};


///
class PyramidEmbedder {
protected:
	PyramidEmbedderProperties Properties;
	torch::Device Device{torch::kCUDA};

	CLIP Clip = nullptr;
	std::shared_ptr<RuCLIPProcessor> ClipProcessor = nullptr;

	int ZoomOutIdx{-1},	//-1, 0 , 2...
		HorPosIdx{0},
		VertPosIdx{0},
		DataImageIdx{0};

	cv::Mat DataImage;
public:
	PyramidEmbedder(CLIP clip, std::shared_ptr<RuCLIPProcessor> clip_processor, const PyramidEmbedderProperties &properties)
		: Clip(clip), ClipProcessor(clip_processor), Properties(properties)
	{}

	std::pair <CLIP, std::shared_ptr<RuCLIPProcessor>> Initialize(
		const std::filesystem::path &clip_path,
		const std::filesystem::path &tokenizer_path,
		const int input_img_size,
		torch::Device device
	);

	///Разбить на патчи с перекрытием  +  парочку масштабов (zoomout) и кэшировать эмбеддинги от них
	PyramidEmbedding operator()(const NeRFDatasetParams &data);

	///Получить очередной фрагмент изображения вместе с его индексами
	///!!!Должно быть согласовано с GetNearestPatchCenters/Vertices
	virtual std::tuple<int, int, int, int, cv::Mat> GetNextSample(const NeRFDatasetParams &data);
};