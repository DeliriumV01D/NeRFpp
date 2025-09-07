#pragma once

#include "TorchHeader.h"
#include "json.hpp"

#include <filesystem>
#include <string>
#include <list>
#include <vector>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

///
struct NeRFDatasetParams {
	torch::Device device{torch::kCUDA};
	int H{ 0 }, W{ 0 };
	float Focal{ 0 },
		Near{ 0 },
		Far{ 0 };
	bool WhiteBgr{ false };
	std::vector<int> SplitsIdx = { 0,0,0 };
	std::vector<std::string> Splits = { "train", "val", "test" };
	torch::Tensor K{torch::Tensor()},
		BoundingBox{ torch::Tensor() };
	cv::Mat d { cv::Mat::zeros(5, 1, CV_64F) };		//k1,k2,k3,p1,p2 коэффициенты дисторсии
	std::vector<torch::Tensor> Poses,
		RenderPoses;
	std::vector<std::filesystem::path> ImagePaths;

	nlohmann::json ToJson() const
	{
		nlohmann::json j;
		j["H"] = H;
		j["W"] = W;
		j["Focal"] = Focal;
		j["Near"] = Near;
		j["Far"] = Far;
		j["SplitsIdx"] = SplitsIdx;
		j["Splits"] = Splits;

		//Serialize torch::Tensor K
		j["K"] = std::vector<float>(K.data_ptr<float>(), K.data_ptr<float>() + K.numel());

		//Serialize torch::Tensor BoundingBox
		j["BoundingBox"] = std::vector<float>(BoundingBox.data_ptr<float>(), BoundingBox.data_ptr<float>() + BoundingBox.numel());

		//Serialize cv::Mat d
		j["d"] = std::vector<double>(d.ptr<double>(), d.ptr<double>() + d.total());

		//Serialize tensors in vectors
		auto serialize_tensors = [](const std::vector<torch::Tensor>& tensors)
		{
			std::vector<std::vector<float>> serialized;
			for (/*const */auto& tensor : tensors)
			{
				serialized.push_back(std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel()));
			}
			return serialized;
		};

		j["Poses"] = serialize_tensors(Poses);
		//j["RenderPoses"] = serialize_tensors(RenderPoses);

		std::vector<std::string> image_paths_str;
		image_paths_str.reserve(ImagePaths.size());
		for (const auto& path : ImagePaths)
			image_paths_str.push_back(path.string());
		j["ImagePaths"] = image_paths_str;
		j["WhiteBgr"] = WhiteBgr;

		return j;
	}		//CompactData::ToJson

	void FromJson(const nlohmann::json& j)
	{
		j.at("H").get_to(H);
		j.at("W").get_to(W);
		j.at("Focal").get_to(Focal);
		j.at("Near").get_to(Near);
		j.at("Far").get_to(Far);
		j.at("SplitsIdx").get_to(SplitsIdx);
		j.at("Splits").get_to(Splits);

		//Deserialize torch::Tensor K
		std::vector<float> k_data = j.at("K").get<std::vector<float>>();
		K = torch::from_blob(k_data.data(), { 3, 3 }, torch::kFloat32).clone().detach();

		//Deserialize torch::Tensor BoundingBox
		std::vector<float> bbox_data = j.at("BoundingBox").get<std::vector<float>>();
		BoundingBox = torch::from_blob(bbox_data.data(), { static_cast<long>(bbox_data.size()) }).clone();

		//Deserialize cv::Mat d
		std::vector<double> d_data = j.at("d").get<std::vector<double>>();
		d = cv::Mat(d_data, true).reshape(1, 5); //Reshape to (5, 1)

		//Deserialize tensors in vectors
		auto deserialize_tensors = [](const std::vector<std::vector<float>>& serialized, at::IntArrayRef sz)
		{
			std::vector<torch::Tensor> tensors;
			for (const auto& data : serialized)
			{
				tensors.push_back(torch::from_blob((float*)data.data(), (sz.size() == 0) ? static_cast<long>(data.size()) : sz).clone());
			}
			return tensors;
		};

		Poses = deserialize_tensors(j.at("Poses").get<std::vector<std::vector<float>>>(), { 4, 4 });
		//RenderPoses = deserialize_tensors(j.at("RenderPoses").get<std::vector<std::vector<float>>>());

		if (j.contains("ImagePaths"))
		{
			std::vector<std::string> image_paths_str = j.at("ImagePaths").get<std::vector<std::string>>();
			ImagePaths.clear();
			ImagePaths.reserve(image_paths_str.size());
			for (const auto& str : image_paths_str)
				ImagePaths.push_back(std::filesystem::path(str));
		}
		j.at("WhiteBgr").get_to(WhiteBgr);
	}		//CompactData::FromJson

	void LoadFromFile(const std::filesystem::path& file_path)
	{
		std::ifstream fs(file_path.string());
		nlohmann::json j;
		fs >> j;
		FromJson(j);
	}

	void SaveToFile(const std::filesystem::path& file_path)
	{
		std::ofstream fs(file_path);
		fs << ToJson() << std::endl;
	}
};		//NeRFDatasetParams


struct LeRFDatasetParams {
	bool UseLerf;
	int clip_input_img_size,
		lang_embed_dim;
	float pyr_embedder_overlap;
	std::filesystem::path PyramidClipEmbeddingSaveDir;
};		//LeRFDatasetParams