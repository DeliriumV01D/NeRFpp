#pragma once

#include "TorchHeader.h"
#include "NeRFRenderer.h"
#include "NeRFDatasetParams.h"
#include "PyramidEmbeddingDataloader.h"

#include <future>
#include <random>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


///Структура для хранения лучей
struct RayBatch {
	torch::Tensor rays_o;
	torch::Tensor rays_d;
	torch::Tensor cone_angle;
	float near;
	float far;
};

///Структура для хранения таргетов
struct TargetBatch {
	torch::Tensor target_s;
	torch::Tensor target_lang_embedding;
	torch::Tensor target_lerf_scale;	///Per-ray pyramid scale normalized to [0,1] across the pyramid range
	float target_image_height{ 0.f };	///View height in pixels (world-scale conversion happens in the renderer)
	float target_focal_y{ 0.f };		///View focal length fy in pixels
};

///Raybatch + targets
using NeRFDataExample = torch::data::Example<RayBatch, TargetBatch>;


///
class NeRFDataset : public torch::data::datasets::BatchDataset<NeRFDataset, NeRFDataExample, std::vector<size_t>> {
protected:
	NeRFDatasetParams Params;

	LeRFDatasetParams LeRFParams;
	std::unique_ptr<PyramidEmbeddingDataloader> PyramidClipEmbedding{nullptr};

	int BatchSize,
		PrecorpIters,
		CurrentIter{ 0 };
	float PrecorpFrac;
	torch::Device Device;

	///Состояние изображений
	int CurrentImageIdx{ -1 },
		NextImageIdx{ -1 };
	torch::Tensor CurrentImage,
		NextImage;
	std::future<void> LoadingFuture;

	///Генератор случайных чисел (используется только из главного потока)
	std::mt19937 Rng;

	int GetRandomTrainIdx();
	torch::Tensor LoadImage(const int idx) const;
	void PrefetchNextImage();
	std::tuple<int, int, int, int> CalculateBounds() const;
	void InitializePyramidClipEmbedding(const CLIP clip, const std::shared_ptr<RuCLIPProcessor> clip_processor);
public:
	NeRFDataset(
		const NeRFDatasetParams &params,
		const LeRFDatasetParams &lerf_params,
		const int batch_size,
		const int precorp_iters,
		const float precorp_frac,
		torch::Device device,
		const CLIP clip,
		const std::shared_ptr<RuCLIPProcessor> clip_processor
	);

	///
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetRayBatch(
		const torch::Tensor &rand_h,
		const torch::Tensor &rand_w,
		int H,
		int W,
		const torch::Tensor &K,
		const torch::Tensor &c2w
	);

	///
	NeRFDataExample get_batch(std::vector<size_t> request/*Не используется*/) override;
	void SetCurrentIter(int iter) { CurrentIter = iter; }
	PyramidEmbeddingDataloader * GetPyramidClipEmbedding() { return PyramidClipEmbedding.get(); }
	std::optional<size_t> size() const override { return torch::nullopt; /*Датасет бесконечен (он генерирует батчи на лету)*/ }
};