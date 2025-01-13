#pragma once

#include "TorchHeader.h"

#include "load_blender.h"

#include <string>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>


void ColmapReconstruction(	const std::filesystem::path &image_path, const std::filesystem::path &workspace_path);

///Чтение параметров камер из базы данных colmap реконструкции
CompactData LoadFromColmapReconstruction( const std::filesystem::path &workspace_path);
