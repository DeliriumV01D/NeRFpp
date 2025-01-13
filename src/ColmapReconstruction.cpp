#include "ColmapReconstruction.h"

//#ifdef USE_COLMAP
#include <colmap/scene/database.h>
#include <colmap/scene/camera.h>
#include <colmap/scene/reconstruction.h>
//#include <colmap/scene/image_database.h>
#include <colmap/estimators/two_view_geometry.h>
//#include <colmap/util/logging.h>
//#include <colmap/util/flags.h>
//#include <colmap/util/parallel.h>
#include <colmap/util/timer.h>
#include <colmap/util/misc.h>
//#include <colmap/util/filesystem.h>

#include <colmap/controllers/feature_extraction.h>
#include <colmap/controllers/feature_matching.h>
//#include <colmap/sfm/sfm.h>
#include <colmap/util/logging.h>
#include <colmap/util/file.h>
#include <colmap/controllers/automatic_reconstruction.h>
#include <colmap/scene/reconstruction_manager.h>
#include <colmap/controllers/hierarchical_pipeline.h>
//#endif

#include <opencv2/core/quaternion.hpp>

static torch::Tensor Rigid3dToTransformationMat(const colmap::Rigid3d &rigid3d)
{
	//Создание кватерниона из колмаповского кватерниона	
	cv::Quat<float> q(rigid3d.rotation.w(), rigid3d.rotation.x(), rigid3d.rotation.y(), rigid3d.rotation.z()); // w, x, y, z

	//Вычисление матрицы вращения из кватерниона
	auto R = q.toRotMat3x3();

	//Объединение матрицы вращения и матрицы трансляции в матрицу преобразования
	torch::Tensor pose = torch::zeros({ 4, 4 });
	for (size_t row = 0; row < R.rows; row++)
		for (size_t col = 0; col < R.cols; col++)
			pose[row][col] = R(row, col);

	for (size_t row = 0; row < rigid3d.translation.size(); row++)
		pose[row][3] = rigid3d.translation[row];
	pose[3][3] = 1;

	//w2c->c2w
	pose = pose.inverse();
	//Convert from COLMAP's camera coordinate system (OpenCV) to NeRF (OpenGL) | righthanded <-> lefthanded
	pose.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(1, 3)}) *= - 1;

	std::cout<<"rigid3d: "<<rigid3d<<std::endl;
	std::cout<<"quat: "<<q<<"Rot: "<<R<<std::endl;
	
	return pose;
}

////Левосторонняя <-> правосторонняя
//static torch::Tensor Rigid3dToTransformationMat(const colmap::Rigid3d &rigid3d)
//{
//	//Создание кватерниона из колмаповского кватерниона	
//	cv::Quat<float> q(rigid3d.rotation.w(), -rigid3d.rotation.x(), -rigid3d.rotation.y(), rigid3d.rotation.z()); // w, x, y, z
//
//	//Вычисление матрицы вращения из кватерниона
//	auto R = q.toRotMat3x3();
//
//	//Объединение матрицы вращения и матрицы трансляции в матрицу преобразования
//	torch::Tensor pose = torch::zeros({ 4, 4 });
//	for (size_t row = 0; row < R.rows; row++)
//		for (size_t col = 0; col < R.cols; col++)
//			pose[row][col] = R(row, col);
//
//	for (size_t row = 0; row < rigid3d.translation.size(); row++)
//		pose[row][3] = rigid3d.translation[row];
//	pose[1][3] = - pose[1][3];
//	pose[3][3] = 1;
//
//	std::cout<<"rigid3d: "<<rigid3d<<std::endl;
//	std::cout<<"quat: "<<q<<"Rot: "<<R<<std::endl;
//	
//	return pose;
//}


void ColmapReconstruction(	const std::filesystem::path &image_path, const std::filesystem::path &workspace_path)
{
	//colmap::InitLogging();

	///Автоматическая реконструкция(!!!можно сэкономить ведь нам нужны только положения камер)
	colmap::AutomaticReconstructionController::Options options;
	std::shared_ptr<colmap::ReconstructionManager> reconstruction_manager = std::make_shared<colmap::ReconstructionManager>();
	options.image_path = image_path.string();					// The path to the image folder which are used as input
	options.workspace_path = workspace_path.string();	// The path to the workspace folder in which all results are stored.
	options.quality = colmap::AutomaticReconstructionController::Quality::EXTREME;			// Whether to perform low- or high-quality reconstruction.
	options.single_camera = true;/*false*/;						// Whether to use shared intrinsics or not.
	options.single_camera_per_folder = true;/*false*/;	// Whether to use shared intrinsics or not for all images in the same sub-folder.
	options.camera_model = "OPENCV";					// Which camera model to use for images.  FULL_OPENCV, OPENCV_FISHEYE
	options.camera_params;										// Initial camera params for all images.
	options.extraction = true;								// Whether to perform feature extraction.
	options.matching = true;									// Whether to perform feature matching.
	options.sparse = true;										// Whether to perform sparse mapping.
	options.dense = false;										// Whether to perform dense mapping.
	std::shared_ptr<colmap::AutomaticReconstructionController> controller = std::make_shared<colmap::AutomaticReconstructionController>(options, reconstruction_manager);

	controller->Start();

	while (!controller->IsFinished())
	{

	}
	std::cout << "Reconstruction completed successfully." << std::endl;

	//controller.Stop();
	//while (!controller.IsStopped()){}

	//ВМЕСТО AutomaticReconstructionController
	// 	colmap::HierarchicalPipeline controller(options, reconstruction_manager);
	//// Hierarchical mapping first hierarchically partitions the scene into multiple overlapping clusters, then reconstructs them separately using incremental
	//// mapping, and finally merges them all into a globally consistent reconstruction. This is especially useful for larger-scale scenes, since
	//// incremental mapping becomes slow with an increasing number of images.
		
	//Или вообще спуститься на уровень пониже для кастомизации
	// void RunFeatureExtraction();
	// void RunFeatureMatching();
	// void RunSparseMapper();
}


///Чтение параметров камер из базы данных colmap реконструкции
CompactData LoadFromColmapReconstruction( const std::filesystem::path &workspace_path)
{
	CompactData result;
	std::string image_path;
	
	const std::filesystem::path database_path = workspace_path/"database.db";
	const std::filesystem::path sparse_path = workspace_path/"sparse";

	//Прочитать путь image_path из строки *.ini файла
	std::filesystem::path ini_file_path;
	for (const auto &entry : std::filesystem::directory_iterator(sparse_path))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".ini")
		{
			ini_file_path = entry.path();
			break;
		}
	}
	if (!ini_file_path.empty())
		std::cout<<"found COLMAP project configuration file "<<ini_file_path.string()<<std::endl;
	else
		std::cout<<"could not found COLMAP project configuration file!"<<std::endl;

	std::ifstream file(ini_file_path);
	if (!file.is_open())
		std::cerr << "cannot open file " <<ini_file_path.string()<< std::endl;

	std::string line;
	while (std::getline(file, line))
	{
		if (line.rfind("image_path=", 0) == 0)
		{
			image_path = line.substr(11);
			break;
		}
	}
	std::cout<<"image path: "<<image_path<<std::endl;

	//// Получение всех реконструкций из базы данных
	//std::vector<colmap::Reconstruction> reconstructions = database.ReadAllReconstructions(&reconstructions);
	colmap::Reconstruction reconstruction;
	reconstruction.Read((sparse_path/"0").string());

	//reconstruction.ComputeBoundingBox();
	//result.SplitsIdx[0] = reconstruction.Images().size();

	// Итерация по всем камерам и вывод их параметров
	for (const auto &camera : reconstruction.Cameras())
	{
		result.H = camera.second.height;
		result.W = camera.second.width;
		result.Focal = camera.second.FocalLength();
		std::cout<<"camera_t: "<<camera.first<< "Camera ID: " << camera.second.camera_id << "Model: " << camera.second.ModelName() << std::endl;
		//std::cout<<camera.second.PrincipalPointX()<<" "<<camera.second.PrincipalPointY()<<" "<<camera.second.FocalLengthX()<<" "<<camera.second.FocalLengthY()<<std::endl;
	}

	int i_split = 0;
	for (const auto &im : reconstruction.Images())
	{
		cv::Mat img = cv::imread(image_path + "/" + im.second.Name(), cv::ImreadModes::IMREAD_UNCHANGED);			//keep all 4 channels(RGBA)
		result.SplitsIdx[i_split]++;

		std::cout << "channels" << img.channels() << std::endl;
		cv::imshow("img", img);
		cv::waitKey(1);
		result.Imgs.emplace_back(CVMatToTorchTensor(img));
		std::cout << image_path + "/" + im.second.Name() << std::endl;

		if (im.second.HasPose())
		{
			auto pose = Rigid3dToTransformationMat(im.second.CamFromWorld());
			std::cout <<"pose: " << pose << std::endl;
			result.Poses.emplace_back(pose);
		} else {
			std::cout<<"have not pose"<<std::endl;
		}
		std::cout << std::endl;
	}

	//!!!
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