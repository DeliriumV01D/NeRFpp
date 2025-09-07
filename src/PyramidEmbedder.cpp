#include "PyramidEmbedder.h"

///{hor_pos_idx, vert_pos_idx, zoom_out_idx, data_img_id, x, y}
std::list<std::tuple<int, int, int, int, float, float>> PyramidEmbedding :: GetNearestPatchIndicesSingleScale(
	const float x,
	const float y,
	const int zoom_out_idx,
	const int data_img_id,
	const PyramidEmbedderProperties &properties,
	const cv::Size &img_size
)	{
	std::list<std::tuple<int, int, int, int, float, float>> result;
	cv::Rect window_rect;

	window_rect.width = properties.ImgSize.width * pow(2, zoom_out_idx);
	window_rect.height = properties.ImgSize.height * pow(2, zoom_out_idx);

	int nw = static_cast<int>((img_size.width - window_rect.width * properties.Overlap)/(window_rect.width * (1. - properties.Overlap)));
	int nh = static_cast<int>((img_size.height - window_rect.height * properties.Overlap)/(window_rect.height * (1. - properties.Overlap)));

	float hor_pos = x / window_rect.width / (1.f - properties.Overlap),
		vert_pos = y / window_rect.height / (1.f - properties.Overlap);

	int hor_pos_idx1,
		hor_pos_idx2,
		vert_pos_idx1,
		vert_pos_idx2;

	float temp;

	hor_pos_idx1 = static_cast<int>(hor_pos - 2);
	hor_pos_idx2 = static_cast<int>(hor_pos - 1);

	vert_pos_idx1 = static_cast<int>(vert_pos - 2);
	vert_pos_idx2 = static_cast<int>(vert_pos - 1);


	if (hor_pos_idx1 < 0) hor_pos_idx1 = 0;
	if (hor_pos_idx1 >= nw) hor_pos_idx1 = nw - 1;
	if (hor_pos_idx2 < 0) hor_pos_idx2 = 0;
	if (hor_pos_idx2 >= nw) hor_pos_idx2 = nw - 1;
	if (vert_pos_idx1 < 0) vert_pos_idx1 = 0;
	if (vert_pos_idx1 >= nh) vert_pos_idx1 = nh - 1;
	if (vert_pos_idx2 < 0) vert_pos_idx2 = 0;
	if (vert_pos_idx2 >= nh) vert_pos_idx2 = nh - 1;

	cv::Point2f p1((hor_pos_idx1 != nw ? static_cast<int>(hor_pos_idx1 * window_rect.width * (1. - properties.Overlap)) : static_cast<int>(img_size.width - window_rect.width)),
			(vert_pos_idx1 != nh ? static_cast<int>(vert_pos_idx1 * window_rect.height * (1. - properties.Overlap)) : static_cast<int>(img_size.height - window_rect.height))),
		p2((hor_pos_idx2 != nw ? static_cast<int>(hor_pos_idx2 * window_rect.width * (1. - properties.Overlap)) : static_cast<int>(img_size.width - window_rect.width)),
			(vert_pos_idx1 != nh ? static_cast<int>(vert_pos_idx1 * window_rect.height * (1. - properties.Overlap)) : static_cast<int>(img_size.height - window_rect.height))), 
		p3((hor_pos_idx1 != nw ? static_cast<int>(hor_pos_idx1 * window_rect.width * (1. - properties.Overlap)) : static_cast<int>(img_size.width - window_rect.width)),
			(vert_pos_idx2 != nh ? static_cast<int>(vert_pos_idx2 * window_rect.height * (1. - properties.Overlap)) : static_cast<int>(img_size.height - window_rect.height))), 
		p4((hor_pos_idx2 != nw ? static_cast<int>(hor_pos_idx2 * window_rect.width * (1. - properties.Overlap)) : static_cast<int>(img_size.width - window_rect.width)),
			(vert_pos_idx2 != nh ? static_cast<int>(vert_pos_idx2 * window_rect.height * (1. - properties.Overlap)) : static_cast<int>(img_size.height - window_rect.height)));

	//В результат идет точка центра прямоугольника
	result.push_back({hor_pos_idx1, vert_pos_idx1, zoom_out_idx, data_img_id, p1.x + window_rect.width/2, p1.y + window_rect.height/2});
	result.push_back({hor_pos_idx2, vert_pos_idx1, zoom_out_idx, data_img_id, p2.x + window_rect.width/2, p2.y + window_rect.height/2});
	result.push_back({hor_pos_idx1, vert_pos_idx2, zoom_out_idx, data_img_id, p3.x + window_rect.width/2, p3.y + window_rect.height/2});
	result.push_back({hor_pos_idx2, vert_pos_idx2, zoom_out_idx, data_img_id, p4.x + window_rect.width/2, p4.y + window_rect.height/2});

	////test
	//cv::Mat test_img(800, 800, CV_8UC3, cv::Scalar(0,0,0));
	//std::cout<<x<<" "<<y<<" "<<hor_pos<<" "<<hor_pos_idx1<<" "<<hor_pos_idx2<<" "<<vert_pos<<" "<<vert_pos_idx1<<" "<<vert_pos_idx2<<" "<<zoom_out_idx<<std::endl;

	//cv::rectangle(test_img, cv::Rect(p1.x, p1.y,window_rect.width, window_rect.height), cv::Scalar(255,100,100));
	//cv::rectangle(test_img, cv::Rect(p1.x + window_rect.width/2 - 1, p1.y + window_rect.height/2 - 1, 3, 3), cv::Scalar(255,100,100));

	//cv::rectangle(test_img, cv::Rect(p2.x, p2.y, window_rect.width, window_rect.height), cv::Scalar(100,255,100));
	//cv::rectangle(test_img, cv::Rect(p2.x + window_rect.width/2 - 1, p2.y + window_rect.height/2 - 1, 3, 3), cv::Scalar(100,255,100));

	//cv::rectangle(test_img, cv::Rect(p3.x, p3.y, window_rect.width, window_rect.height), cv::Scalar(100,100,255));
	//cv::rectangle(test_img, cv::Rect(p3.x + window_rect.width/2 - 1, p3.y + window_rect.height/2 - 1, 3, 3), cv::Scalar(100,100,255));

	//cv::rectangle(test_img, cv::Rect(p4.x, p4.y, window_rect.width, window_rect.height), cv::Scalar(100,255,255));
	//cv::rectangle(test_img, cv::Rect(p4.x + window_rect.width/2 - 1, p4.y + window_rect.height/2, 3, 3), cv::Scalar(100,255,255));

	//cv::rectangle(test_img, cv::Point(x-1,y-1), cv::Point(x+1, y+1), cv::Scalar(255,255,255));
	//cv::imshow("test_img", test_img);
	//cv::waitKey(0);

	return result;
}		//PyramidEmbedding :: GetNearestPatchIndicesSingleScale


///!!!Должно быть согласовано с GetNextSample(const CompactData &data)
std::list<std::tuple<int, int, int, int, float, float>> PyramidEmbedding :: GetNearestPatchIndicesMultiScale(
	const float x,
	const float y,
	const float scale,
	const int data_img_id,
	const PyramidEmbedderProperties &properties,
	const cv::Size &img_size
)	{
	//scale = img_scale * f / t; scale = pow(2, zoom_out_idx);
	//Get zoom_out_idx from scale //-1, 0, 1, 2 ... <- 1/2, 1, 2, 4 ...
	int zoom_out_idx1 = int(std::log2(scale));
		
	//Максимальное приближение
	if (zoom_out_idx1 < -1)
		zoom_out_idx1 = -1;
	//Максимальное удаление
	if (zoom_out_idx1 > properties.MaxZoomOut)
		zoom_out_idx1 = properties.MaxZoomOut;

	int zoom_out_idx2 = zoom_out_idx1 + 1;

	//Максимальное приближение
	if (zoom_out_idx2 < -1)
		zoom_out_idx2 = -1;
	//Максимальное удаление
	if (zoom_out_idx2 > properties.MaxZoomOut)
		zoom_out_idx2 = properties.MaxZoomOut;

	auto result = GetNearestPatchIndicesSingleScale(x, y, zoom_out_idx1, data_img_id, properties, img_size);
	result.splice(result.end(), GetNearestPatchIndicesSingleScale(x, y, zoom_out_idx2, data_img_id, properties, img_size));

	return result;
}		//PyramidEmbedding :: GetNearestPatchIndicesMultiScale


torch::Tensor PyramidEmbedding :: Interpolate(
	const int hor_pos_idx1, const int hor_pos_idx2,
	const int vert_pos_idx1, const int vert_pos_idx2,
	const int zoom_out_idx1, const int zoom_out_idx2,
	const int data_img_id,
	const float x1, const float x2, const float y1, const float y2,
	const float x,
	const float y,
	const cv::Size &img_size
)	{
	//auto r = (x-x1) * (x-x1) + (y-y1) * (y-y1);
	//auto result = Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}];

	//auto r1 = (x-x2) * (x-x2) + (y-y1) * (y-y1);
	//if (r1 < r)
	//{
	//	r = r1;
	//	result = Embeddings[{hor_pos_idx2, vert_pos_idx1, zoom_out_idx1, data_img_id}];
	//}

	//r1 = (x-x1) * (x-x1) + (y-y2) * (y-y2);
	//if (r1 < r)
	//{
	//	r = r1;
	//	result = Embeddings[{hor_pos_idx1, vert_pos_idx2, zoom_out_idx1, data_img_id}];
	//}

	//r1 = (x-x2) * (x-x2) + (y-y2) * (y-y2);
	//if (r1 < r)
	//{
	//	r = r1;
	//	result = Embeddings[{hor_pos_idx2, vert_pos_idx2, zoom_out_idx1, data_img_id}];
	//}

	////r1 = (x - 0) * (x - 0);
	////if (r1 < r)
	////	result = zeros_like(result);

	////r1 = (y - 0) * (y - 0);
	////if (r1 < r)
	////	result = zeros_like(result);

	////r1 = (x - img_size.width) * (x - img_size.width);
	////if (r1 < r)
	////	result = zeros_like(result);

	////r1 = (y - img_size.height) * (y - img_size.height);
	////if (r1 < r)
	////	result = zeros_like(result);

	//return result;
		
	if (x2 == x1 && y2 == y1)
		return Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}];
		
	if (x2 == x1)
	{
		auto d1 = (y2 - y1);
		return Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}] + 
			(Embeddings[{hor_pos_idx1, vert_pos_idx2, zoom_out_idx1, data_img_id}] - Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}]) / d1 * (y - y1);
	}

	if (y2 == y1)
	{
		auto d1 = (x2 - x1);
		return Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}] + 
			(Embeddings[{hor_pos_idx2, vert_pos_idx1, zoom_out_idx1, data_img_id}] - Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}]) / d1 * (x - x1);
	}

	auto d1 = (x2 - x1) * (y2 - y1);
	return Embeddings[{hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id}] / d1 * (x2 - x) * (y2- y) +
		Embeddings[{hor_pos_idx2, vert_pos_idx1, zoom_out_idx1, data_img_id}] / d1 * (x - x1) * (y2 - y) +
		Embeddings[{hor_pos_idx1, vert_pos_idx2, zoom_out_idx1, data_img_id}] / d1 * (x2 - x) * (y - y1) +
		Embeddings[{hor_pos_idx2, vert_pos_idx2, zoom_out_idx1, data_img_id}] / d1 * (x - x1) * (y - y1);
}			//PyramidEmbedding :: Interpolate


void PyramidEmbedding :: Save(const std::filesystem::path &data_file)		//itemName + ".pt
{
	std::vector<torch::Tensor> saving_data;
	for (auto &it : Embeddings)
	{
		std::array<int,4> idx;
		std::tie(idx[0],idx[1],idx[2],idx[3]) = it.first;
		saving_data.push_back(torch::tensor({ idx }));
		saving_data.push_back(it.second);
	}
	torch::save(saving_data, data_file.string());
}

void PyramidEmbedding :: Load(const std::filesystem::path &data_file)
{
	std::vector<torch::Tensor> loading_data;
	torch::load(loading_data, data_file.string());
	for (auto it = loading_data.begin(); it != loading_data.end(); it++)
	{
		torch::Tensor idx = *it;
		it++;
		torch::Tensor embed = *it;
		Embeddings[{idx[0].item<int>(), idx[1].item<int>(), idx[2].item<int>(), idx[3].item<int>()}] = embed;
	}
}




///Процедуру вычисления эмбединга для каждого пикселя в зависимости от масштаба 
///трилинейная интерполяция между центрами патчей на ближайших к скейлу масштабах покрывает все случаи
torch::Tensor PyramidEmbedding :: GetPixelValue(
	const float x,
	const float y,
	const float scale,
	const int data_img_id,
	const PyramidEmbedderProperties &properties,
	const cv::Size &img_size		//Размер обрабатываемого изображения
){
	//1. получить индексы эмбеддингов 8 ближайших точек
	auto patch_idx = GetNearestPatchIndicesMultiScale(x, y, scale, data_img_id, properties, img_size);
	//2. Трилинейно интерполировать между ними
	//std::unordered_map<std::tuple<int, int, int, int>, torch::Tensor> Embeddings;
	//в GetNearestPatchIndicesMultiScale упаковывается так:
	//result.push_back({hor_pos_idx1, vert_pos_idx1, zoom_out_idx, data_img_id, x, y
	//result.push_back({hor_pos_idx2, vert_pos_idx1, zoom_out_idx, data_img_id, x, y
	//result.push_back({hor_pos_idx1, vert_pos_idx2, zoom_out_idx, data_img_id, x, y
	//result.push_back({hor_pos_idx2, vert_pos_idx2, zoom_out_idx, data_img_id, x, y
	//затем в таком же порядке второй слой с другим scale
	//auto d = (x2-x1)*(y2-y1)*(z2-z1);
	//f(x,y,z) = 
	//	f(x1,y1,z1)/d*(x2-x)*(y2-y)*(z2-z) +
	//	f(x1,y1,z2)/d*(x2-x)*(y2-y)*(z-z1) +
	//	f(x1,y2,z1)/d*(x2-x)*(y-y1)*(z2-z) +
	//	f(x1,y2,z2)/d*(x2-x)*(y-y1)*(z-z1) +
	//	f(x2,y1,z1)/d*(x-x1)*(y2-y)*(z2-z) +
	//	f(x2,y1,z2)/d*(x-x1)*(y2-y)*(z-z1) +
	//	f(x2,y2,z1)/d*(x-x1)*(y-y1)*(z2-z) +
	//	f(x2,y2,z2)/d*(x-x1)*(y-y1)*(z-z1);
	//Или, поскольку координаты уголков/центров патчей не совпадают между слоями - 
	// две билинейных интерполяции по одной на каждый слой + линейная интерполяция между слоями
	//auto d = (x2 - x1)*(y2 - y1);
	//f(x,y) = f(x1, y1)/d*(x2-x)*(y2-y) +
	//	f(x2, y1)/d*(x-x1)*(y2-y) +
	//	f(x1, y2)/d*(x2-x)*(y-y1) +
	//	f(x2, y2)/d*(x-x1)*(y-y1);
	torch::Tensor e1, e2, result;
	float zoom_out1, zoom_out2;
	auto it = patch_idx.begin();
	{
		auto [hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id, x1, y1] = *it;
		it++;it++;it++;
		auto [hor_pos_idx2, vert_pos_idx2, zoom_out_idx2, data_img_id2, x2, y2] = *it;
		it++;

		e1 = Interpolate(hor_pos_idx1, hor_pos_idx2, vert_pos_idx1, vert_pos_idx2,
			zoom_out_idx1, zoom_out_idx2, data_img_id,
			x1, x2, y1, y2,
			x,
			y, 
			img_size);
		zoom_out1 = zoom_out_idx1;
	}
	{
		auto [hor_pos_idx1, vert_pos_idx1, zoom_out_idx1, data_img_id, x1, y1] = *it;
		it++;it++;it++;
		auto [hor_pos_idx2, vert_pos_idx2, zoom_out_idx2, data_img_id2, x2, y2] = *it;
		it++;

		e2 = Interpolate(hor_pos_idx1, hor_pos_idx2, vert_pos_idx1, vert_pos_idx2,
			zoom_out_idx1, zoom_out_idx2, data_img_id,
			x1, x2, y1, y2,
			x,
			y,
			img_size);
		zoom_out2 = zoom_out_idx2;
	}

	//См GetNearestPatchIndicesMultiScale
	//scale = img_scale * f / t; scale = pow(2, zoom_out_idx);
	//Get zoom_out_idx from scale //-1, 0, 1, 2 ... <- 1/2, 0, 2, 4 ...
	float zoom_out = std::log2(scale);
	if (zoom_out == zoom_out1)
		result = e1;
	if (zoom_out == zoom_out2)
		result = e2;

	if (zoom_out != zoom_out1 && zoom_out != zoom_out2)		//хоть это и флоты но присвоены от интов поэтому можно так сравнить
		result = e1 + (e2 - e1) / (zoom_out2 - zoom_out1) * (zoom_out - zoom_out1);

	return result;
}		//PyramidEmbedding :: GetPixelValue





std::pair <CLIP, std::shared_ptr<RuCLIPProcessor>> PyramidEmbedder :: Initialize(
	const std::filesystem::path &clip_path,
	const std::filesystem::path &tokenizer_path,
	const int input_img_size,
	torch::Device device
){
	Device = device;

	std::cout << "Loading CLIP from: " << clip_path << std::endl;
	CLIP clip = FromPretrained(clip_path);
	clip->to(device);

	std::cout << "Loading tokenizer model from: " << tokenizer_path << std::endl;
	std::shared_ptr<RuCLIPProcessor> clip_processor = std::make_shared<RuCLIPProcessor>(
		tokenizer_path,
		input_img_size
		//77,
		//{ 0.48145466, 0.4578275, 0.40821073 },
		//{ 0.26862954, 0.26130258, 0.27577711 }
	);
	return {clip, clip_processor};
}


///Разбить на патчи с перекрытием  +  парочку масштабов (zoomout) и кэшировать эмбеддинги от них
PyramidEmbedding PyramidEmbedder :: operator()(const NeRFDatasetParams &data)
{
	PyramidEmbedding result;

	ZoomOutIdx = -1;  //-1, 0 , 2...
	HorPosIdx = 0;
	VertPosIdx = 0;
	DataImageIdx = 0;
	DataImage = cv::imread(data.ImagePaths[DataImageIdx].string(), cv::IMREAD_COLOR/*cv::IMREAD_UNCHANGED*/);

	while (true)
	{
		torch::NoGradGuard no_grad;
		///Получить очередной фрагмент изображения вместе с его индексами
		auto [hor_pos_idx, vert_pos_idx, zoom_out_idx, data_img_idx, sample_mat] = GetNextSample(data);
		if (sample_mat.empty())
			break;

		auto input = ClipProcessor->operator()(std::vector <std::string>(), {sample_mat});
		auto image_features = Clip->EncodeImage(input.second.to(Device));
		//normalize features
		image_features = image_features / image_features.norm(2/*L2*/, -1, true);

		result.Embeddings[{hor_pos_idx, vert_pos_idx, zoom_out_idx, data_img_idx}] = image_features.to(torch::kCPU);
	}
	return result;
}


///Получить очередной фрагмент изображения вместе с его индексами
///!!!Должно быть согласовано с GetNearestPatchCenters/Vertices
std::tuple<int, int, int, int, cv::Mat> PyramidEmbedder :: GetNextSample(const NeRFDatasetParams &data)
{
	cv::Mat sample;
	cv::Rect window_rect;
	bool found = true;
	std::tuple<int, int, int, int, cv::Mat> result;


	if (!DataImage.empty())
	{
		window_rect.width = Properties.ImgSize.width * pow(2, ZoomOutIdx);
		window_rect.height = Properties.ImgSize.height * pow(2, ZoomOutIdx);

		int h = data.H,
			w = data.W;

		int nw = static_cast<int>((w - window_rect.width * Properties.Overlap)/(window_rect.width * (1. - Properties.Overlap)));
		int nh = static_cast<int>((h - window_rect.height * Properties.Overlap)/(window_rect.height * (1. - Properties.Overlap)));


		//if (HorPosIdx != nw)
			window_rect.x = static_cast<int>(HorPosIdx * window_rect.width * (1. - Properties.Overlap));
		//else
		//	window_rect.x = static_cast<int>(w - window_rect.width);

		//if (VertPosIdx != nh)
			window_rect.y = static_cast<int>(VertPosIdx * window_rect.height * (1. - Properties.Overlap));
		//else
		//	window_rect.y = static_cast<int>(h - window_rect.height);

		if (found)
		{
			DataImage(window_rect).copyTo(sample);

			if (ZoomOutIdx != 0)
			{
				cv::resize(sample, sample, Properties.ImgSize);
			}
		}
		result = {HorPosIdx, VertPosIdx, ZoomOutIdx, DataImageIdx, sample};

		////test
		//cv::Mat test_img(800, 800, CV_8UC3, cv::Scalar(0,0,0));
		//std::cout<<window_rect.x<<" "<<window_rect.y<<" "<<HorPosIdx<<" "<<VertPosIdx<<" "<<ZoomOutIdx<<" "<<DataImageIdx<<std::endl;
		//cv::rectangle(test_img, cv::Rect(window_rect), cv::Scalar(255,100,100));
		//cv::rectangle(test_img, cv::Rect(window_rect.x + window_rect.width/2 - 1, window_rect.y + window_rect.height/2 - 1, 3, 3), cv::Scalar(255,100,100));
		//cv::imshow("sample", sample);
		//cv::imshow("test_img", test_img);
		//cv::waitKey(0);

		//Циклы по вертикальным и горизонтальным позициям
		VertPosIdx++;
		if (VertPosIdx == nh /*+ 1*/)
		{
			VertPosIdx = 0;

			HorPosIdx++;
			if (HorPosIdx == nw /*+ 1*/)
			{
				HorPosIdx = 0;

				//Цикл по масштабам окна
				int n = std::min(log2f(w / Properties.ImgSize.width), log2f(h / Properties.ImgSize.height));
				ZoomOutIdx++;
				if (ZoomOutIdx > std::min(n, Properties.MaxZoomOut))
				{
					ZoomOutIdx = -1;

					//Цикл по сэмплам датасета
					DataImageIdx++; //RandomInt() % data.Imgs.size();
					if (DataImageIdx < data.ImagePaths.size())
						DataImage = cv::imread(data.ImagePaths[DataImageIdx].string(), cv::IMREAD_COLOR/*cv::IMREAD_UNCHANGED*/);
					else
						DataImage = cv::Mat();
				}
			}
		}
	} //if ()

	//hor_pos_idx, vert_pos_idx, zoom_out_idx, data_img_idx, sample_mat
	return result;
}							//PyramidEmbedder :: GetSample