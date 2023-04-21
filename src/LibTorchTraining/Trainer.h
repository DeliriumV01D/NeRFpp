#pragma once

#include "TorchHeader.h"
#include "TrainHistoryLog.h"
#include "TensorBoard.h"

struct TrainerParams {
	int64_t NumberOfEpochs,
					LogInterval,
					CheckpointEvery;
	std::string NetCheckpointPath,
							OptimizerCheckpointPath;
};

class Trainer {
protected:
	TrainerParams Params;
public:
	Trainer(const TrainerParams &params) : Params(params){}
	virtual ~Trainer(){}

	template <typename Net, typename DataLoader, typename Loss, typename Accuracy>
	std::pair<float, float> Test (
		Net net,
		torch::Device &device,
		DataLoader &data_loader,
		Loss &loss_function,
		Accuracy &accuracy_function
	){
		net->eval();
		torch::NoGradGuard no_grad;
		float mean_loss = 0,
					mean_acc = 0;
		int64_t batch_index = 0;
		for (auto &batch : *data_loader)
		{
			//net->zero_grad();
			torch::Tensor output = net->forward(batch.data.to(device));
			mean_loss += loss_function(output, batch.target.to(device)).template item<float>();
			mean_acc += accuracy_function(output, batch.target.to(device)).template item<float>();
			batch_index++;
		}
		mean_loss /= batch_index;
		mean_acc /= batch_index;
		return {mean_loss, mean_acc};
	}		//Test

	template <typename Net, typename DataLoader, typename Loss, typename Accuracy>
	TrainHistoryLog Train (
		Net net,
		torch::Device &device,
		DataLoader &train_data_loader,
		DataLoader &val_data_loader,
		torch::optim::Optimizer &optimizer,
		torch::optim::LRScheduler &sheduler,
		Loss &loss_function,
		Accuracy &accuracy_function,
		TensorBoard * tensor_board = nullptr
	){
		TrainHistoryLog train_history_log;
		net->train();

		int64_t checkpoint_counter = 1;
		int64_t batch_index = 0;
		//train loop из семи залуп
		for (int64_t epoch = 1; epoch <= static_cast<int64_t>(Params.NumberOfEpochs); epoch++)
		{
			for (auto &p : optimizer.param_groups())
				std::cout << "nnap optimizer lr = "<< p.options().get_lr() << std::endl;

			for (auto &batch : *train_data_loader)
			{
				net->zero_grad();
				torch::Tensor output = net->forward(batch.data.to(device));
				torch::Tensor loss = loss_function(output, batch.target.to(device));
				loss.backward();
				optimizer.step();

				batch_index++;

				if (batch_index % Params.LogInterval == 0)
				{
					torch::Tensor acc = accuracy_function(output, batch.target.to(device));
					TrainHistoryLogEntry entry;
					entry.Epoch = static_cast<float>(epoch);
					entry.TrainLoss = loss.item<float>();
					entry.TrainAcc = acc.item<float>();
					std::tie(entry.ValLoss,	entry.ValAcc) = Test(net, device, val_data_loader, loss_function, accuracy_function);
					train_history_log.push_back(entry);
					if (tensor_board != nullptr)
						tensor_board->Visualize(train_history_log);
					std::cout << "[" << epoch << "|" << Params.NumberOfEpochs << "][" << batch_index << "] train loss: " <<entry.TrainLoss << " train acc: " <<entry.TrainAcc<< " val loss: " <<  entry.ValLoss << " val acc: " << entry.ValAcc << std::endl;
				}

				if (batch_index % Params.CheckpointEvery == 0)
				{
					// Checkpoint the model and optimizer state.
					if (!Params.NetCheckpointPath.empty())
						torch::save(net, Params.NetCheckpointPath);
//					if (!Params.OptimizerCheckpointPath.empty())
//						torch::save(optimizer, Params.OptimizerCheckpointPath);
					std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
				}
			}
			sheduler.step();
		}
		return train_history_log;
	}		//Train

};			//Trainer
