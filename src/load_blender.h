#pragma once

#include "json.hpp"
#include "TorchHeader.h"
#include "NeRFRenderer.h"
//#include "RayUtils.h"

#include <filesystem>
#include <string>
#include <list>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

const float PI = acosf(-1.0f);

inline torch::Tensor trans_t (const float t)
{
	float data[] = { 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, t,
		0, 0, 0, 1 };
	torch::Tensor result = torch::from_blob(data, { 4, 4 });
	return result;
}

inline torch::Tensor rot_phi (const float phi)
{
	float data[] = { 1, 0, 0, 0,
		0, cosf(phi), -sinf(phi), 0,
		0, sinf(phi), cosf(phi), 0,
		0, 0, 0, 1 };
	torch::Tensor result = torch::from_blob(data, { 4, 4 });
	return result;
}

inline torch::Tensor rot_theta(const float th)
{
	float data[] = {
		cosf(th), 0, -sinf(th), 0,
		0, 1, 0, 0,
		sinf(th), 0, cosf(th), 0,
		0, 0, 0, 1 };
	torch::Tensor result = torch::from_blob(data, { 4, 4 });
	return result;
}

inline torch::Tensor pose_spherical(const float theta, const float phi, const float radius, const float x = 0, const float y = 0, const float z = 0)
{
	auto c2w = trans_t(radius);
	c2w = torch::matmul(rot_phi(phi / 180. * PI), c2w);
	c2w = torch::matmul(rot_theta(theta / 180. * PI), c2w);
	float data[] = { -1, 0, 0, 0,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 0, 1 };
	c2w = torch::matmul(torch::from_blob(data, { 4, 4 }), c2w);
	c2w[0][3] = c2w[0][3] + x;	//put
	c2w[1][3] = c2w[1][3] + y;
	c2w[2][3] = c2w[2][3] + z;
	return c2w;
}

//Задать калибровки камеры
inline torch::Tensor GetCalibrationMatrix(const float focal, const float w, const float h)
{
	float kdata[] = { focal, 0, 0.5f * w,
		0, focal, 0.5f * h,
		0, 0, 1 };
	return torch::from_blob(kdata, { 3, 3 }/*, torch::kFloat32*/);
}

///
struct CompactData {
	int H{0}, W{0};
	float Focal{0},
		Near{0},
		Far{0};
	std::vector<int> SplitsIdx = { 0,0,0 };
	std::vector<std::string> Splits = { "train", "val", "test" };
	torch::Tensor K{torch::Tensor()},
		BoundingBox{torch::Tensor()};
	cv::Mat d { cv::Mat::zeros(5, 1, CV_64F) };		//k1,k2,k3,p1,p2 коэффициенты дисторсии
	std::vector<torch::Tensor> Imgs,
		Poses,
		RenderPoses;

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
		auto serialize_tensors = [](const std::vector<torch::Tensor> &tensors)
		{
			std::vector<std::vector<float>> serialized;
			for (/*const */auto &tensor : tensors)
			{
				serialized.push_back(std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel()));
			}
			return serialized;
		};

		//j["Imgs"] = serialize_tensors(Imgs);
		j["Poses"] = serialize_tensors(Poses);
		//j["RenderPoses"] = serialize_tensors(RenderPoses);

		return j;
	}		//CompactData::ToJson

	void FromJson(const nlohmann::json &j)
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
		K = torch::from_blob(k_data.data(), { static_cast<long>(k_data.size()) }).clone();

		//Deserialize torch::Tensor BoundingBox
		std::vector<float> bbox_data = j.at("BoundingBox").get<std::vector<float>>();
		BoundingBox = torch::from_blob(bbox_data.data(), { static_cast<long>(bbox_data.size()) }).clone();

		//Deserialize cv::Mat d
		std::vector<double> d_data = j.at("d").get<std::vector<double>>();
		d = cv::Mat(d_data, true).reshape(1, 5); //Reshape to (5, 1)

		//Deserialize tensors in vectors
		auto deserialize_tensors = [](const std::vector<std::vector<float>> &serialized, at::IntArrayRef sz)
		{
			std::vector<torch::Tensor> tensors;
			for (const auto &data : serialized)
			{
				tensors.push_back(torch::from_blob((float*)data.data(), (sz.size() == 0)?static_cast<long>(data.size()):sz).clone());
			}
			return tensors;
		};

		//Imgs = deserialize_tensors(j.at("Imgs").get<std::vector<std::vector<float>>>());
		Poses = deserialize_tensors(j.at("Poses").get<std::vector<std::vector<float>>>(), {4, 4});
		//RenderPoses = deserialize_tensors(j.at("RenderPoses").get<std::vector<std::vector<float>>>());
	}		//CompactData::FromJson

	void LoadFromFile(const std::filesystem::path &file_path)
	{
		std::ifstream fs(file_path);
		nlohmann::json j;
		fs >> j;
		FromJson(j);
	}

	void SaveToFile(const std::filesystem::path &file_path)
	{
		std::ofstream fs(file_path);
		fs << ToJson() << std::endl;
	}
};		//CompactData

///BoundingBox calculation
inline torch::Tensor GetBbox3dForObj(const CompactData &data) 
{
	////Задать калибровки камеры
	//float kdata[] = { data.Focal, 0, 0.5f * data.W,
	//	0, data.Focal, 0.5f * data.H,
	//	0, 0, 1 };
	//torch::Tensor k = torch::from_blob(kdata, { 3, 3 });

	//torch::Tensor directions = GetDirections(result.H, result.W, k);
	torch::Tensor min_bound = torch::tensor({ 1e8f, 1e8f, 1e8f }),
		max_bound = torch::tensor({ -1e8f, -1e8f, -1e8f });

	//std::vector<torch::Tensor> train_poses;
	//std::copy(data.Poses.begin(), std::next(data.Poses.begin(), data.SplitsIdx[0]), std::back_inserter(train_poses));
	//for (auto &c2w : train_poses)
	for (auto c2w = data.Poses.begin(); c2w != std::next(data.Poses.begin(), data.SplitsIdx[0]); c2w++)
	{
		auto [rays_o, rays_d] = GetRays(data.H, data.W, data.K, *c2w);		//[800, 800, 3]
		//цикл по ограничивающим угловым лучам
		for (auto it : std::vector <std::pair<int, int>>({ {0, 0}, {data.W - 1, 0}, {0, data.H - 1}, {data.W - 1, data.H - 1} }))
		{
			auto min_point = rays_o[it.first][it.second] + data.Near * rays_d[it.first][it.second];		//[3]
			auto max_point = rays_o[it.first][it.second] + data.Far * rays_d[it.first][it.second];		//[3]
			min_bound = torch::min(min_bound, min_point);
			min_bound = torch::min(min_bound, max_point);
			max_bound = torch::max(max_bound, min_point);
			max_bound = torch::max(max_bound, max_point);
		}
	}
	std::cout << "min_bound: " << min_bound << "; max_bound: " << max_bound << std::endl;
	return torch::cat({ min_bound, max_bound }, -1);
}

inline CompactData load_blender_data(const std::filesystem::path &basedir, const bool half_res = false, const bool testskip = true)
{
	using json = nlohmann::json;
	CompactData result;
	//std::vector<torch::Tensor> img_vec,
	//	pose_vec;

	for (int i_split = 0; i_split < result.Splits.size(); i_split++)
	{
		if (testskip && result.Splits[i_split] == "test")
			continue;
		std::filesystem::path path = basedir;
		path /= ("transforms_" + result.Splits[i_split] + ".json");
		std::cout << path << std::endl;
		std::ifstream f(path);
		json data = json::parse(f);

		for (auto frame : data["frames"])
		{
			std::filesystem::path img_path = basedir;
			img_path /= (std::string(frame["file_path"]) + ".png");
			cv::Mat img = cv::imread(img_path.string(), cv::ImreadModes::IMREAD_UNCHANGED);			//keep all 4 channels(RGBA)
			result.SplitsIdx[i_split]++;
			result.W = img.cols;
			result.H = img.rows;
			if (half_res)
				cv::resize(img, img, cv::Size(result.W/2, result.H/2));
			
			std::cout << "channels" << img.channels() << std::endl;
			cv::imshow("img", img);
			cv::waitKey(1);
			result.Imgs.emplace_back(CVMatToTorchTensor(img));

			std::cout << img_path << std::endl;
			std::cout << "transform_matrix: " << frame["transform_matrix"] << std::endl;

			torch::Tensor pose = torch::zeros({ 4, 4 });
			for (size_t row = 0; row < frame["transform_matrix"].size(); row++)
			{
				auto val_row = frame["transform_matrix"][row];
				
				for (size_t col = 0; col < val_row.size(); col++)
					pose[row][col] = (float)val_row[col];
			}
			std::cout <<"pose: " << pose << std::endl;
			//pose.index_put_({ torch::indexing::Slice(torch::indexing::None, 4), -1 }, pose.index({ torch::indexing::Slice(torch::indexing::None, 4), -1 })/10);
			//std::cout << "pose: " << pose << std::endl;
			result.Poses.emplace_back(pose);
		}			//for (auto frame : data["frames"])
		float camera_angle_x = float(data["camera_angle_x"]);
		result.Focal = .5f * result.W / tanf(.5 * camera_angle_x);
	}				//for (int i_split = 0; i_split < splits.size(); i_split++)

	float n = 40 + 1;
	float delta = 360 / n;
	for (float angle = -180; angle <= 180; angle += delta)
	{
		result.RenderPoses.emplace_back(pose_spherical(angle, -30.0, 4.0));
		std::cout << angle << " " << result.RenderPoses.back()/*pose_spherical(angle, -30.0, 4.0)*/ << std::endl;
	}

	if (half_res)
	{
		result.H = result.H / 2;
		result.W = result.W / 2;
		result.Focal = result.Focal / 2;
	}
	result.Near = 2.f;
	result.Far = 6.f;
	float kdata[] = { result.Focal, 0, 0.5f * result.W,
		0, result.Focal, 0.5f * result.H,
		0, 0, 1 };
	result.K = torch::from_blob(kdata, { 3, 3 }, torch::kFloat32);
	//result.K = GetCalibrationMatrix(result.Focal, result.W, result.H);
	result.BoundingBox = GetBbox3dForObj(result);		//(train_poses, result.H, result.W, /*near =*/ 2.0f, /*far =*/ 6.0f);
	return result;
}