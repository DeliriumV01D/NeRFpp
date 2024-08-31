#include "CuSHEncoder.h"

///
std::pair<torch::Tensor, torch::Tensor> CuSHEncoderImpl :: forward(torch::Tensor input)
{
	auto device = input.device();
	return std::make_pair(CuSHEncode(input), torch::Tensor());
}