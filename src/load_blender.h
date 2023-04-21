#pragma once

#include "json.hpp"
#include "TorchHeader.h"
#include "BaseNeRFRenderer.h"

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

inline torch::Tensor pose_spherical(const float theta, const float phi, const float radius)
{
	auto c2w = trans_t(radius);
	c2w = torch::mm(rot_phi(phi / 180. * PI), c2w);
	c2w = torch::mm(rot_theta(theta / 180. * PI), c2w);
	float data[] = { -1, 0, 0, 0,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 0, 1 };
	c2w = torch::mm(torch::from_blob(data, { 4, 4 }), c2w);
	return c2w;
}

///
struct CompactData {
	std::vector<torch::Tensor> Imgs,
		Poses,
		RenderPoses;
	int H{0}, W{0};
	float Focal{0};
	std::vector<std::string> Splits = { "train", "val", "test" };
	std::vector<int> SplitsIdx = { 0,0,0 };
};

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
			result.Poses.emplace_back(pose);
			//pose_vec.emplace_back(pose);
		}			//for (auto frame : data["frames"])
		float camera_angle_x = float(data["camera_angle_x"]);
		result.Focal = .5f * result.W / tanf(.5 * camera_angle_x);
	}				//for (int i_split = 0; i_split < splits.size(); i_split++)

	//result.Imgs = torch::cat(img_vec, 0);			//-1?0?
	//result.Poses = torch::cat(pose_vec, 0);		//-1?0?

	float n = 40 + 1;
	float delta = 360 / n;
	for (float angle = -180; angle <= 180; angle += delta)
	{
		result.RenderPoses.emplace_back(pose_spherical(angle, -30.0, 4.0));
		//result.RenderPoses = torch::cat({ result.RenderPoses, pose_spherical(angle, -30.0, 4.0) }, 0);
		std::cout << angle << " " << result.RenderPoses.back()/*pose_spherical(angle, -30.0, 4.0)*/ << std::endl;
	}

	if (half_res)
	{
		result.H = result.H / 2;
		result.W = result.W / 2;
		result.Focal = result.Focal / 2;
	}
	return result;
}