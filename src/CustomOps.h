#pragma once

#include <torch/torch.h>

namespace torch::autograd {

class TruncExp : public Function<TruncExp> {
public:
	static variable_list forward(AutogradContext *ctx, Tensor input);
	static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}
