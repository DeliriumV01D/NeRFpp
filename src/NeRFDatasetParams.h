#pragma once

#include "TorchHeader.h"
#include "json.hpp"

#include <filesystem>
#include <string>
#include <list>
#include <vector>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

struct View {
	unsigned int ID{ 0 };
	int H{ 0 },
		W{ 0 };
	float Focal{ 0 },
		Near{ 0 },
		Far{ 0 };
	torch::Tensor K{torch::Tensor()},
		Pose;           //c2w
	cv::Mat d { cv::Mat::zeros(5, 1, CV_64F) };     //k1,k2,k3,p1,p2 коэффициенты дисторсии
	std::filesystem::path ImagePath;

	nlohmann::json ToJson() const
	{
		nlohmann::json j;
		j["ID"] = ID;
		j["H"] = H;
		j["W"] = W;
		j["Focal"] = Focal;
		j["Near"] = Near;
		j["Far"] = Far;

		if (K.defined())
		{
			j["K"] = {
					{"data", std::vector<float>(K.data_ptr<float>(), K.data_ptr<float>() + K.numel())},
					{"shape", K.sizes().vec()}
			};
		} else {
			j["K"] = nullptr;
		}

		if (Pose.defined())
		{
			j["Pose"] = {
					{"data", std::vector<float>(Pose.data_ptr<float>(), Pose.data_ptr<float>() + Pose.numel())},
					{"shape", Pose.sizes().vec()}
			};
		} else {
			j["Pose"] = nullptr;
		}

		if (!d.empty())
		{
			j["d"] = std::vector<double>(d.ptr<double>(), d.ptr<double>() + d.total());
		} else {
			j["d"] = std::vector<double>();
		}

		j["ImagePath"] = ImagePath.string();

		return j;
	}

	void FromJson(const nlohmann::json &j)
	{
		j.at("ID").get_to(ID);
		j.at("H").get_to(H);
		j.at("W").get_to(W);
		j.at("Focal").get_to(Focal);
		j.at("Near").get_to(Near);
		j.at("Far").get_to(Far);

		if (!j.at("K").is_null())
		{
			auto k_data = j.at("K").at("data").get<std::vector<float>>();
			auto k_shape = j.at("K").at("shape").get<std::vector<int64_t>>();
			K = torch::from_blob(k_data.data(), k_shape, torch::kFloat).clone();
		}

		if (!j.at("Pose").is_null())
		{
			auto pose_data = j.at("Pose").at("data").get<std::vector<float>>();
			auto pose_shape = j.at("Pose").at("shape").get<std::vector<int64_t>>();
			Pose = torch::from_blob(pose_data.data(), pose_shape, torch::kFloat).clone();
		}

		auto d_data = j.at("d").get<std::vector<double>>();
		if (!d_data.empty())
		{
			d = cv::Mat(d_data, true).reshape(1, 5); // Reshape to (5, 1)
		} else {
			d = cv::Mat::zeros(5, 1, CV_64F);
		}

		ImagePath = j.at("ImagePath").get<std::string>();
	}
};

///
struct NeRFDatasetParams {
	torch::Device device{torch::kCUDA};
	bool WhiteBgr{ false };
	std::vector<int> SplitsIdx = { 0,0,0 };
	std::vector<std::string> Splits = { "train", "val", "test" };
	torch::Tensor BoundingBox{ torch::Tensor() };
	std::vector<View> Views;

	nlohmann::json ToJson() const
	{
		nlohmann::json j;
		j["WhiteBgr"] = WhiteBgr;
		j["SplitsIdx"] = SplitsIdx;
		j["Splits"] = Splits;

		if (BoundingBox.defined())
		{
			j["BoundingBox"] = {
					{"data", std::vector<float>(BoundingBox.data_ptr<float>(), BoundingBox.data_ptr<float>() + BoundingBox.numel())},
					{"shape", BoundingBox.sizes().vec()}
			};
		} else {
			j["BoundingBox"] = nullptr;
		}

		nlohmann::json views_json;
		for (const auto& view : Views)
			views_json.push_back(view.ToJson());

		j["Views"] = views_json;

		return j;
	}

	void FromJson(const nlohmann::json &j)
	{
		j.at("WhiteBgr").get_to(WhiteBgr);
		j.at("SplitsIdx").get_to(SplitsIdx);
		j.at("Splits").get_to(Splits);

		if (!j.at("BoundingBox").is_null())
		{
			auto bbox_data = j.at("BoundingBox").at("data").get<std::vector<float>>();
			auto bbox_shape = j.at("BoundingBox").at("shape").get<std::vector<int64_t>>();
			BoundingBox = torch::from_blob(bbox_data.data(), bbox_shape, torch::kFloat).clone();
		}

		auto views_json = j.at("Views").get<std::vector<nlohmann::json>>();
		Views.clear();
		for (const auto &view_json : views_json)
		{
			View view;
			view.FromJson(view_json);
			Views.push_back(view);
		}
	}

	void LoadFromFile(const std::filesystem::path &file_path)
	{
		std::ifstream fs(file_path.string());
		nlohmann::json j;
		fs >> j;
		FromJson(j);
	}

	void SaveToFile(const std::filesystem::path &file_path)
	{
		std::ofstream fs(file_path);
		fs << ToJson().dump(4) << std::endl;
	}
};

///
struct LeRFDatasetParams {
	bool UseLerf;
	int clip_input_img_size,
		lang_embed_dim,
		MinZoomOut;		//0 or -1
	float pyr_embedder_overlap;
	std::filesystem::path PyramidClipEmbeddingSaveDir;
};		//LeRFDatasetParams