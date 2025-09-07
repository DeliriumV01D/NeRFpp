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
#include <colmap/feature/sift.h>

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

	std::cout << "reconstruction image_path = " << image_path << std::endl;

	// Инициализация путей
	const std::filesystem::path database_path = workspace_path / "database.db";
	const std::filesystem::path sparse_path = workspace_path / "sparse";

	////1. Feature Extraction
	//colmap::ImageReaderOptions reader_options;
	//reader_options.database_path = database_path.string();			//Path to database in which to store the extracted data.
	//reader_options.image_path = image_path.string();						//Root path to folder which contains the images.
	//reader_options.camera_model = "OPENCV";											//Which camera model to use for images.  FULL_OPENCV, OPENCV_FISHEYE
	//reader_options.single_camera = true;												//Whether to use the same camera for all images.
	//reader_options.single_camera_per_folder = false;						//Whether to use the same camera for all images in the same sub-folder.
	//reader_options.single_camera_per_image = false;							//Whether to use a different camera for each image.
	////Whether to explicitly use an existing camera for all images. Note that in
	////this case the specified camera model and parameters are ignored.
	//reader_options.existing_camera_id = colmap::kInvalidCameraId;
	////Manual specification of camera parameters. If empty, camera parameters
	////will be extracted from EXIF, i.e. principal point and focal length.
	//reader_options.camera_params = "";
	////If camera parameters are not specified manually and the image does not have focal length EXIF information,
	////the focal length is set to the value `default_focal_length_factor * max(width, height)`.
	//reader_options.default_focal_length_factor = 1.2;
	////Optional path to an image file specifying a mask for all images. No features will be extracted in regions 
	////where the mask is black (pixel intensity value 0 in grayscale).
	//reader_options.camera_mask_path = "";

	//colmap::SiftExtractionOptions sift_fe_options;
	//sift_fe_options.num_threads = 4;		//Number of threads for feature extraction.
	//sift_fe_options.use_gpu = true;		// Whether to use the GPU for feature extraction.
	////Index of the GPU used for feature extraction. For multi-GPU extraction, you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
	//sift_fe_options.gpu_index = "-1";
	//sift_fe_options.max_image_size = 3200;																				//Maximum image size, otherwise image will be down-scaled.
	//sift_fe_options.max_num_features = 8192;																			//Maximum number of features to detect, keeping larger-scale features.
	//sift_fe_options.first_octave = -1;																						//First octave in the pyramid, i.e. -1 upsamples the image by one level.
	//sift_fe_options.num_octaves = 4;																							//Number of octaves.
	//sift_fe_options.octave_resolution = 3;																				//Number of levels per octave.
	//sift_fe_options.peak_threshold = 0.02 / sift_fe_options.octave_resolution;		//Peak threshold for detection.
	//sift_fe_options.edge_threshold = 10.0;																				//Edge threshold for detection.
	////Estimate affine shape of SIFT features in the form of oriented ellipses as opposed to original SIFT which estimates oriented disks.
	//sift_fe_options.estimate_affine_shape = false;
	//sift_fe_options.max_num_orientations = 2;																			//Maximum number of orientations per keypoint if not estimate_affine_shape.
	//sift_fe_options.upright = false;																							//Fix the orientation to 0 for upright features.
	////Whether to adapt the feature detection depending on the image darkness. Note that this feature is only available in the OpenGL SiftGPU version.
	//sift_fe_options.darkness_adaptivity = false;
	////Domain-size pooling parameters. Domain-size pooling computes an average
	////SIFT descriptor across multiple scales around the detected scale. This was proposed in "Domain-Size Pooling in Local Descriptors and Network
	////Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to outperform other SIFT variants and learned descriptors in "Comparative
	////Evaluation of Hand-Crafted and Learned Local Features", Schönberger, Hardmeier, Sattler, Pollefeys, CVPR 2016.
	//sift_fe_options.domain_size_pooling = false;
	//sift_fe_options.dsp_min_scale = 1.0 / 6.0;
	//sift_fe_options.dsp_max_scale = 3.0;
	//sift_fe_options.dsp_num_scales = 10;
	////Whether to force usage of the covariant VLFeat implementation. Otherwise, the covariant implementation is only used when
	////estimate_affine_shape or domain_size_pooling are enabled, since the normal Sift implementation is faster.
	//sift_fe_options.force_covariant_extractor = false;
	//std::unique_ptr<colmap::Thread> feature_extractor = colmap::CreateFeatureExtractorController(reader_options, sift_fe_options);
	////auto active_thread = feature_extractor.get();
	//feature_extractor->Start();
	//feature_extractor->Wait();
	////feature_extractor.reset();
	////active_thread = nullptr;

	////// 2. Feature Matching
	////colmap::ExhaustiveFeatureMatcher::Options matcher_options;

	////// Настройка качества
	////switch (options.quality) {
	////case colmap::AutomaticReconstructionController::Quality::EXTREME:
	////	matcher_options.num_threads = -1;
	////	matcher_options.gpu_index = "0";
	////	matcher_options.sift_matching_options.max_num_matches = 32768;
	////	break;
	////}

	////colmap::ExhaustiveFeatureMatcher feature_matcher(
	////	matcher_options, database_path, "");
	////feature_matcher.Start();
	////feature_matcher.Wait();


	////// 3. Sparse Reconstruction
	////if (options.sparse)
	////{
	////	auto reconstruction_manager =
	////		std::make_shared<colmap::ReconstructionManager>();

	////	colmap::IncrementalMapperController::Options mapper_options;
	////	colmap::OptionManager option_manager;

	////	// Настройка параметров маппинга
	////	mapper_options.min_num_matches = 15;
	////	mapper_options.init_image_id1 = colmap::kInvalidImageId;
	////	mapper_options.init_image_id2 = colmap::kInvalidImageId;
	////	mapper_options.max_num_models = 1;
	////	mapper_options.num_threads = -1;

	////	// Настройка качества
	////	switch (options.quality) {
	////	case colmap::AutomaticReconstructionController::Quality::EXTREME:
	////		mapper_options.ba_global_images_ratio = 1.1;
	////		mapper_options.ba_global_points_ratio = 1.1;
	////		mapper_options.ba_global_max_num_iterations = 100;
	////		break;
	////	}

	////	colmap::IncrementalMapperController mapper(
	////		&option_manager, mapper_options, image_path, database_path,
	////		*reconstruction_manager);
	////	mapper.Start();
	////	mapper.Wait();

	////	// Сохранение результатов
	////	if (reconstruction_manager->Size() > 0) {
	////		reconstruction_manager->Get(0).Write(sparse_path);
	////	}
	////}

	///Автоматическая реконструкция(!!!можно сэкономить ведь нам нужны только положения камер)
	colmap::AutomaticReconstructionController::Options options;
	std::shared_ptr<colmap::ReconstructionManager> reconstruction_manager = std::make_shared<colmap::ReconstructionManager>();
	options.image_path = image_path.string();					// The path to the image folder which are used as input
	options.workspace_path = workspace_path.string();	// The path to the workspace folder in which all results are stored.
	options.quality = colmap::AutomaticReconstructionController::Quality::EXTREME;			// Whether to perform low- or high-quality reconstruction.
	options.single_camera = true;/*false*/;						// Whether to use shared intrinsics or not.
	options.single_camera_per_folder = true;/*false*/;	// Whether to use shared intrinsics or not for all images in the same sub-folder.
	options.camera_model = "OPENCV";					// Which camera model to use for images.  FULL_OPENCV, OPENCV_FISHEYE
	//options.camera_params = "1000,1000,400,400,0,0,0,0";//"fx,fy,cx,cy,k1,k2,p1,p2" for OPENCV // Initial camera params for all images.
	options.extraction = true;								// Whether to perform feature extraction.
	options.matching = true;									// Whether to perform feature matching.
	options.sparse = true;										// Whether to perform sparse mapping.
	options.dense = false;										// Whether to perform dense mapping.
	std::shared_ptr<colmap::AutomaticReconstructionController> controller = std::make_shared<colmap::AutomaticReconstructionController>(options, reconstruction_manager);

	std::cout << "begin reconstruction" << std::endl;


	controller->Start();
	controller->Wait();
	//while (!controller->IsFinished()) {
	//	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	//}

	std::cout << "Reconstruction completed successfully." << std::endl;

	//controller->Stop();
	//while (!controller->IsStopped()){}

	//ВМЕСТО AutomaticReconstructionController
	// 	colmap::HierarchicalPipeline controller(options, reconstruction_manager);
	//// Hierarchical mapping first hierarchically partitions the scene into multiple overlapping clusters, then reconstructs them separately using incremental
	//// mapping, and finally merges them all into a globally consistent reconstruction. This is especially useful for larger-scale scenes, since
	//// incremental mapping becomes slow with an increasing number of images.
}


///Чтение параметров камер из базы данных colmap реконструкции
NeRFDatasetParams LoadFromColmapReconstruction( const std::filesystem::path &workspace_path)
{
	NeRFDatasetParams result;
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
		result.ImagePaths.push_back(image_path + "/" + im.second.Name());
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

	//// Get intrinsic calibration matrix composed from focal length and principal
	//// point parameters, excluding distortion parameters.
	//Eigen::Matrix3d CalibrationMatrix() const; in reconstruction.Cameras()
	//!!!
	float kdata[] = { result.Focal, 0, 0.5f * result.W,
		0, result.Focal, 0.5f * result.H,
		0, 0, 1 };
	result.K = torch::from_blob(kdata, { 3, 3 }, torch::kFloat32).clone().detach();;
	//result.K = GetCalibrationMatrix(result.Focal, result.W, result.H);
	auto bounds = GetBoundsForObj(result);			///!!!Можно придумать что-то поизящнее чем просто найти максимальную дистанцию между камерами, например, привязаться к параметрам камеры, оценить из имеющейся разреженной реконструкции
	result.Near = bounds.first;
	result.Far = bounds.second;
	result.BoundingBox = GetBbox3dForObj(result);		//(train_poses, result.H, result.W, /*near =*/ 2.0f, /*far =*/ 6.0f);
	return result;
}