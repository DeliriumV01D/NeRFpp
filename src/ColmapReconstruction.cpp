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

#include <colmap/image/undistortion.h>
//#endif


#include <opencv2/core/quaternion.hpp>

static torch::Tensor ColmapW2CToNeRFC2W(const colmap::Rigid3d &rigid3d)
{
	//Создание кватерниона из колмаповского кватерниона	
	cv::Quat<float> q(static_cast<float>(rigid3d.rotation.w()), static_cast<float>(rigid3d.rotation.x()), static_cast<float>(rigid3d.rotation.y()), static_cast<float>(rigid3d.rotation.z())); // w, x, y, z

	//Вычисление матрицы вращения из кватерниона
	auto R = q.toRotMat3x3();

	torch::Tensor R_tens = torch::zeros({ 3, 3 });
	for (int row = 0; row < R.rows; row++)
		for (int col = 0; col < R.cols; col++)
			R_tens[row][col] = R(row, col);

	//Вектор трансляции
	torch::Tensor t_tens = torch::zeros({ 3 });
	for (size_t row = 0; row < rigid3d.translation.size(); row++)
		t_tens[row] = rigid3d.translation[row];

	//w2c->c2w
	torch::Tensor R_inv = torch::linalg_inv(R_tens);	//R.transpose(0, 1); // Для ортогональной матрицы вращения обратная = транспонированная
	torch::Tensor t_inv = -torch::matmul(R_inv, t_tens);

	//Собираем обратную матрицу camera-to-world
	torch::Tensor pose = torch::eye(4);
	pose.index_put_({ torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3) }, R_inv);
	pose.index_put_({ torch::indexing::Slice(0, 3), 3 }, t_inv);
	//pose[3][3] = 1;

	//Convert from COLMAP's camera coordinate system (OpenCV) to NeRF (OpenGL) | righthanded <-> lefthanded
	pose.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(1, 3)}) *= - 1;

	std::cout<<"rigid3d: "<<rigid3d<<std::endl;
	std::cout<<"quat: "<<q<<"Rot: "<<R<<std::endl;
	
	return pose;
}

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
	options.quality = colmap::AutomaticReconstructionController::Quality::HIGH;			// Whether to perform low- or high-quality reconstruction.
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

	std::cout << "Reconstruction completed successfully." << std::endl;

	//ВМЕСТО AutomaticReconstructionController
	// 	colmap::HierarchicalPipeline controller(options, reconstruction_manager);
	//// Hierarchical mapping first hierarchically partitions the scene into multiple overlapping clusters, then reconstructs them separately using incremental
	//// mapping, and finally merges them all into a globally consistent reconstruction. This is especially useful for larger-scale scenes, since
	//// incremental mapping becomes slow with an increasing number of images.
}


///
std::pair<float, float> ComputeNearFarForImage(
	const colmap::Image &image,
	const colmap::Reconstruction &reconstruction,
	float near_percentile /*= 0.1f*/,
	float far_percentile /*= 0.9f*/
) {
	std::vector<float> distances;
	const colmap::Camera &camera = reconstruction.Camera(image.CameraId());

	//Позиция камеры в мировых координатах
	Eigen::Vector3d camera_pos = image.CamFromWorld().translation;

	for (const auto &point2D : image.Points2D())
	{
		if (point2D.HasPoint3D())
		{
			const colmap::Point3D &point3D = reconstruction.Point3D(point2D.point3D_id);
			double distance = (point3D.xyz - camera_pos).norm();
			distances.push_back(static_cast<float>(distance));
		}
	}

	if (distances.empty())
		return { 0.f, 0.f }; //Значения по умолчанию

	std::sort(distances.begin(), distances.end());
	size_t near_idx = std::min(static_cast<size_t>(near_percentile * distances.size()), distances.size() - 1);
	size_t far_idx = std::min(static_cast<size_t>(far_percentile * distances.size()), distances.size() - 1);

	return { distances[near_idx], distances[far_idx] };
}


void UndistortImages(colmap::Reconstruction &reconstruction, const std::filesystem::path &image_path, const std::filesystem::path &output_path, bool use_gpu /*= false*/)
{
	std::cout << "Starting image undistortion..." << std::endl;
	colmap::UndistortCameraOptions undistortion_options;
	//undistortion_options.max_image_size = max_image_size;
	//COLMAPUndistorter, PMVSUndistorter, CMPMVSUndistorter
	colmap::COLMAPUndistorter undistorter(undistortion_options, reconstruction, image_path.string(), output_path.string());
	//undistorter.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
	undistorter.Run();
	std::cout << "Image undistortion completed successfully." << std::endl;
}


///Чтение параметров камер из базы данных colmap реконструкции
NeRFDatasetParams LoadFromColmapReconstruction( const std::filesystem::path &workspace_path)
{
	NeRFDatasetParams result;
	std::string image_path;
	
	const std::filesystem::path database_path = workspace_path/"database.db";
	const std::filesystem::path sparse_path = workspace_path/"sparse";
	const std::filesystem::path undistorted_path = workspace_path/"undistorted";



	////Получение всех реконструкций из базы данных
	//std::vector<colmap::Reconstruction> reconstructions = database.ReadAllReconstructions(&reconstructions);
	colmap::Reconstruction reconstruction;
	reconstruction.Read((sparse_path/"0").string());

	//Андисторт
	if (!std::filesystem::exists(undistorted_path))
	{
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
			std::cout << "found COLMAP project configuration file " << ini_file_path.string() << std::endl;
		else
			std::cout << "could not found COLMAP project configuration file!" << std::endl;

		std::ifstream file(ini_file_path);
		if (!file.is_open())
			std::cerr << "cannot open file " << ini_file_path.string() << std::endl;

		std::string line;
		while (std::getline(file, line))
		{
			if (line.rfind("image_path=", 0) == 0)
			{
				image_path = line.substr(11);
				break;
			}
		}
		std::cout << "image path: " << image_path << std::endl;

		std::filesystem::create_directories(undistorted_path);
		UndistortImages(reconstruction, image_path, undistorted_path, /*use_gpu*/false);
	}

	image_path = (undistorted_path / "images").string();
	std::cout << "image path: " << image_path << std::endl;
	reconstruction.Read((undistorted_path / "sparse").string());

	//Итерация по всем камерам и вывод их параметров
	for (const auto &camera : reconstruction.Cameras())
	{
		//result.H = camera.second.height;
		//result.W = camera.second.width;
		//result.Focal = camera.second.FocalLength();
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

		View view;
		view.ID = im.second.ImageId();
		view.ImagePath = image_path + "/" + im.second.Name();
		std::cout << view.ImagePath << std::endl;

		if (im.second.HasCameraPtr())
		{
			view.H = im.second.CameraPtr()->height;
			view.W = im.second.CameraPtr()->width;
			view.Focal = im.second.CameraPtr()->FocalLength();
			//Get intrinsic calibration matrix composed from focal length and principal point parameters, excluding distortion parameters.
			//Eigen::Matrix3d CalibrationMatrix() const; in reconstruction.Cameras()
			float kdata[] = { (float)im.second.CameraPtr()->FocalLengthX(), 0, (float)im.second.CameraPtr()->PrincipalPointX(),
				0, (float)im.second.CameraPtr()->FocalLengthY(), (float)im.second.CameraPtr()->PrincipalPointY(),
				0, 0, 1 };
			view.K = torch::from_blob(kdata, { 3, 3 }, torch::kFloat32).clone().detach();
			std::cout << "K: " << view.K << std::endl;
			//result.K = GetCalibrationMatrix(result.Focal, result.W, result.H);
		} else {
			std::cout << "has not camera ptr" << std::endl;
			continue;
		}

		if (im.second.HasPose())
		{
			view.Pose = ColmapW2CToNeRFC2W(im.second.CamFromWorld());
			std::tie(view.Near, view.Far) = ComputeNearFarForImage(im.second, reconstruction, 0.01f, 0.99f);
			std::cout <<"pose: " << view.Pose <<"; near:  "<<view.Near<<"; far:  "<< view.Far << std::endl;
		} else {
			std::cout<<"has not pose"<<std::endl;
			continue;
		}
		std::cout << std::endl;
		result.Views.emplace_back(view);
	}

	//result.BoundingBox = GetBbox3dForObj(result);		//(train_poses, result.H, result.W, /*near =*/ 2.0f, /*far =*/ 6.0f);
	auto colmap_bb = reconstruction.ComputeBoundingBox(0.005, 0.995);
	torch::Tensor min_bound = torch::tensor({ (float)colmap_bb.first.x(), (float)colmap_bb.first.y(), (float)colmap_bb.first.z() }),
		max_bound = torch::tensor({ (float)colmap_bb.second.x(), (float)colmap_bb.second.y(), (float)colmap_bb.second.z() });
	auto d = torch::norm(max_bound - min_bound);
	result.BoundingBox = torch::cat({ min_bound - d * 0.01, max_bound + d * 0.01 }, -1);
	std::cout << "result.BoundingBox: " << result.BoundingBox << std::endl;
	return result;
}