#pragma once

#include "BaseEmbedder.h"

///
class CuSHEncoderImpl : public BaseEmbedderImpl {
protected:
	int InputDim,
		Degree,
		OutputDims;
	torch::Tensor CuSHEncode(const torch::Tensor &input);
public:
	CuSHEncoderImpl(
		const std::string &module_name,
		const int input_dim = 3,
		const int degree = 4
	) : BaseEmbedderImpl(module_name), InputDim(input_dim), Degree(degree), OutputDims(pow(degree, 2))
	{
		//assert input_dim == 3
		//assert degree >= 1 && self.degree <= 5
	}
	virtual ~CuSHEncoderImpl() {}

	int GetOutputDims() override { return OutputDims; }

	///
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) override;
};
TORCH_MODULE(CuSHEncoder);