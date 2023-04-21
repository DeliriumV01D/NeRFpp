#pragma once

#include "TorchHeader.h"


struct Trainable : public torch::nn::Module
{
	template <typename T>				//torch::nn::ModuleHolder<ImplType>, NamedModuleList
	static int ParamsCount(T &module);
	template <typename T>				//torch::nn::ModuleHolder<ImplType>, NamedModuleList
	static void Initialize(T &module);

	Trainable (const std::string module_name) : torch::nn::Module(module_name){}
	virtual ~Trainable(){}
	virtual torch::Tensor forward(torch::Tensor x) = 0;
};

template <typename T>				//torch::nn::ModuleHolder<ImplType>, NamedModuleList
int Trainable :: ParamsCount(T &module)
{
	int result = 0;
	for (auto p : module->parameters())
	{
		int ss = 1;
		for (auto s : p.sizes())
			ss *= s;
		result += ss;
	}
	return result;
}

template <typename T>				//torch::nn::ModuleHolder<ImplType>, NamedModuleList
void Trainable :: Initialize(T &module)
{
	for (auto &p : module->named_parameters())
	{
		if (p.key().find("norm") != p.key().npos && p.key().find(".weight") != p.key().npos)
		{
			module->named_parameters()[p.key()] = torch::nn::init::constant_(p.value(), 1.);
			std::cout << p.key() << std::endl;
		} else if (p.key().find(".weight") != p.key().npos)
		{
			module->named_parameters()[p.key()] = torch::nn::init::xavier_normal_(p.value(), 0.1);
			std::cout << p.key() << std::endl;
		}

		if (p.key().find(".bias") != p.key().npos)
		{
			module->named_parameters()[p.key()] = torch::nn::init::constant_(p.value(), 0.);
			std::cout << p.key() << std::endl;
		}
	}
}
