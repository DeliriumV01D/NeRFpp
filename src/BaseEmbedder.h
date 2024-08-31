#pragma once

#include "TorchHeader.h"

///
class BaseEmbedderImpl : public torch::nn::Module {
protected:
public:
	BaseEmbedderImpl(const std::string &module_name) : torch::nn::Module(module_name) {}
	virtual ~BaseEmbedderImpl() {}
	virtual int GetOutputDims() { return 0; }//abstract;
	///embedding + mask(can be empty)
	virtual std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) { return std::make_pair(torch::Tensor(), torch::Tensor()); }//abstract;
};
TORCH_MODULE(BaseEmbedder);