#pragma once
#include "load_blender.h"

#include "TorchHeader.h"

#include <string>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>


void ColmapReconstruction(	const std::filesystem::path &image_path, const std::filesystem::path &workspace_path);

/////
//std::pair<float, float> ComputeNearFarForImage(
//	const colmap::Image &image,
//	const colmap::Reconstruction &reconstruction,
//	float near_percentile = 0.1f,
//	float far_percentile = 0.9f
//);
//
/////
//std::pair<float, float> ComputeGlobalNearFar(
//	colmap::Reconstruction &reconstruction,
//	float near_percentile = 0.1f,
//	float far_percentile = 0.9f
//);


///Чтение параметров камер из базы данных colmap реконструкции
NeRFDatasetParams LoadFromColmapReconstruction( const std::filesystem::path &workspace_path);
