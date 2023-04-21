#pragma once

#include <vector>

struct TrainHistoryLogEntry {
	float Epoch,
				TrainLoss,
				TrainAcc,
				ValLoss,
				ValAcc;
};

using TrainHistoryLog = std::vector<TrainHistoryLogEntry>;
