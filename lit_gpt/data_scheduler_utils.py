import math
import torch


class ConstWeight(object):
    def __init__(self, initial_weight):
        self.initial_weight = initial_weight

    def get_weight(self, current_step):
        return self.initial_weight


class StepWeight(object):
    def __init__(self, initial_weight, step_size, gamma):
        self.initial_weight = initial_weight
        self.step_size = step_size
        self.gamma = gamma

    def get_weight(self, current_step):
        num_decreases = current_step // self.step_size
        current_weight = self.initial_weight * (self.gamma**num_decreases)

        return current_weight


class LinearWeight(object):
    def __init__(self, initial_weight, end_weight, total_steps):
        self.initial_weight = initial_weight
        self.end_weight = end_weight
        self.total_steps = total_steps

    def get_weight(self, current_step):
        current_step = min(current_step, self.total_steps)
        current_weight = self.initial_weight + (self.end_weight - self.initial_weight) * (
            current_step / self.total_steps
        )

        return current_weight


class CosineAnnealingWeight(object):
    def __init__(self, initial_weight, eta_min, T_max):
        self.eta_max = initial_weight
        self.eta_min = eta_min
        self.T_max = T_max

    def get_weight(self, current_step):
        current_step = current_step % self.T_max
        current_weight = (
            self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * current_step / self.T_max)) / 2
        )

        return current_weight


class PiecewiseWeight(object):
    def __init__(self, scheduler_configs, default_weight, args):
        self.scheduler_configs = scheduler_configs
        self.args = args
        self.default_weight = default_weight
        self.intervals = [scheduler_curr_config[0] for scheduler_curr_config in scheduler_configs]
        self._schedulers = [
            GetScheduler(scheduler_curr_config[1], default_weight, args) for scheduler_curr_config in scheduler_configs
        ]

    def get_weight(self, current_step):
        for i, start in enumerate(self.intervals):
            end = float("inf") if i == len(self.intervals) - 1 else self.intervals[i + 1]

            if start <= current_step < end:
                return self._schedulers[i].get_weight(current_step - start)

        return self.default_weight


class BaseWeight(object):
    def __init__(self):
        pass

    def get_weight(self, current_step):
        return 0


def GetScheduler(scheduler_curr_config, default_weight, args):
    if type(scheduler_curr_config[0]) == list:
        return PiecewiseWeight(scheduler_curr_config, default_weight, args)
    elif scheduler_curr_config[0] == "const":
        # ["const", initial_weight]
        return ConstWeight(scheduler_curr_config[1])
    elif scheduler_curr_config[0] == "step":
        # ["step", initial_weight, step_size, gamma]
        return StepWeight(scheduler_curr_config[1], scheduler_curr_config[2], scheduler_curr_config[3])
    elif scheduler_curr_config[0] == "linear":
        # ["linear", initial_weight, end_weight, total_steps]
        if len(scheduler_curr_config) == 3:
            return LinearWeight(scheduler_curr_config[1], scheduler_curr_config[2], args.max_iters)
        else:
            return LinearWeight(scheduler_curr_config[1], scheduler_curr_config[2], scheduler_curr_config[3])
    elif scheduler_curr_config[0] == "cosine":
        # ["cosine", initial_weight, eta_min, T_max]
        return CosineAnnealingWeight(scheduler_curr_config[1], scheduler_curr_config[2], scheduler_curr_config[3])
    elif scheduler_curr_config[0] == "base":
        return BaseWeight()
    else:
        raise NotImplementedError


class DataScheduler(object):
    def __init__(self, data_scheduler_tracker, data_config, args):
        self.data_scheduler_tracker = data_scheduler_tracker
        self.data_config = data_config
        self.args = args
        self.base_id = None

        self.max_epochs = []
        self._schedulers = []
        for i, curr_config in enumerate(self.data_config):
            if "max_epoch" in curr_config:
                self.max_epochs.append(curr_config["max_epoch"])
            else:
                self.max_epochs.append(float("inf"))

            if "scheduler" not in curr_config:
                scheduler_curr_config = ["const", curr_config["weight"]]
            else:
                scheduler_curr_config = curr_config["scheduler"]

                if curr_config["scheduler"][0] == "base":
                    self.base_id = i
                    self.data_scheduler_tracker.base_id = i

            self._schedulers.append(GetScheduler(scheduler_curr_config, curr_config["weight"], args))

        self.data_scheduler_tracker.max_epochs = self.max_epochs

    def step(self, curr_step):
        for i in range(len(self._schedulers)):
            self.data_scheduler_tracker.weights[i] = self._schedulers[i].get_weight(curr_step)

            if self.max_epochs[i] <= self.data_scheduler_tracker.epoch_count[i]:
                self.data_scheduler_tracker.weights[i] = 0

        if self.base_id is not None:
            self.data_scheduler_tracker.weights[self.base_id] = 100.0 - sum(self.data_scheduler_tracker.weights)

    def get_data_weights(self):
        return self.data_scheduler_tracker.weights

    def get_sample_count(self):
        return self.data_scheduler_tracker.sample_count

    def get_epoch_count(self):
        return self.data_scheduler_tracker.epoch_count

    def set_one_hot_schedule(self, idx):
        self._schedulers = [ConstWeight(0) for _ in range(len(self._schedulers))]
        self._schedulers[idx] = ConstWeight(1)


class DataSchedulerTracker(object):
    def __init__(self, weights):
        self.weights = weights
        self.sample_count = torch.zeros(len(weights))
        self.epoch_count = torch.zeros(len(weights))
        self.max_epochs = None
        self.base_id = None

    def get_data_weights(self):
        return self.weights

    def get_sample_count(self):
        return self.sample_count

    def get_epoch_count(self):
        return self.epoch_count
