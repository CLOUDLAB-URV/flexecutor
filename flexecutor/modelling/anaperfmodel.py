from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, ResourceConfig, ConfigBounds


def io_func(x, a, b):
    return a / x + b


def comp_func(x, a, b):
    return a / x + b


class AnaPerfModel(PerfModel):
    """
    AnaPerfModel records the mean parameter value.
    Advantage: it is fast and accurate enough to optimize the average performance.
    Shortcoming: it does not guarantee the bounded performance.

    Ditto, Caerus model.
    Adapted from https://github.com/pkusys/Jolteon/blob/main/workflow/perf_model_analytic.py
    """

    def __init__(self, model_name, model_dst, stage_id, stage_name) -> None:
        assert isinstance(stage_name, str)
        assert isinstance(stage_id, int) and stage_id >= 0
        super().__init__("analytic", model_name, model_dst)

        self._stage_name = stage_name
        self._stage_id = stage_id

        self._allow_parallel = True

        # Init in train, list with size three
        self._write_params = None
        self._read_params = None
        self._comp_params = None
        self._cold_params = None

        self._profiling_results = None

    @classmethod
    def _config_to_xparam(cls, num_vcpu, memory, num_func):
        return round(num_vcpu * memory * num_func, 1)

    # TODO: review that and rethink
    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self._allow_parallel = allow_parallel

    @overrides
    def save_model(self):
        pass

    @overrides
    def load_model(self):
        pass

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        self._profiling_results = stage_profile_data

        for config_data in stage_profile_data.values():
            assert (
                all(key in config_data for key in FunctionTimes.profile_keys())
            ), f"Each configuration's data must contain {FunctionTimes.profile_keys()} keys."

        print("Training Analytical performance model for %s" % self._stage_name)

        cold_arr = np.array(
            [data["cold_start"] for _, data in stage_profile_data.items()]
        )
        self._cold_params = np.mean(cold_arr)

        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}

        for config, data in stage_profile_data.items():
            num_vcpu, memory, num_func = config

            # adapt to parallel mode
            # if the stage does not allow more than one function, ignore num_func
            if self._allow_parallel:
                config_key = self._config_to_xparam(num_vcpu, memory, num_func)
            else:
                config_key = self._config_to_xparam(num_vcpu, memory, 1)

            # collect data for read step
            if config_key not in size2points_read:
                size2points_read[config_key] = []
            size2points_read[config_key].extend(data["read"])

            # collect data for comp step
            if config_key not in size2points_comp:
                size2points_comp[config_key] = []
            size2points_comp[config_key].extend(data["compute"])

            # collect data for write step
            if config_key not in size2points_write:
                size2points_write[config_key] = []
            size2points_write[config_key].extend(data["write"])

        # average the data
        for config in size2points_read:
            size2points_read[config] = np.mean(size2points_read[config])
        for config in size2points_comp:
            size2points_comp[config] = np.mean(size2points_comp[config])
        for config in size2points_write:
            size2points_write[config] = np.mean(size2points_write[config])

        print(size2points_read)
        print(size2points_comp)
        print(size2points_write)

        def fit_params(data, func):
            assert isinstance(data, dict)
            arr_x = list(data.keys())
            arr_y = [data[x] for x in arr_x]

            arr_x = np.array(arr_x)
            arr_y = np.array(arr_y)

            initial_guess = [1, 1]

            def residuals(para, x, y):
                return func(x, *para) - y

            params, _ = scipy_opt.leastsq(residuals, initial_guess, args=(arr_x, arr_y))

            return params.tolist()

        # Fit the parameters
        self._read_params = fit_params(size2points_read, io_func)
        self._comp_params = fit_params(size2points_comp, comp_func)
        self._write_params = fit_params(size2points_write, io_func)

        print(self._read_params)
        print(self._comp_params)
        print(self._write_params)

    @property
    @overrides
    def parameters(self):
        a = sum([self._read_params[0], self._comp_params[0], self._write_params[0]])
        b = sum([self._read_params[1], self._comp_params[1], self._write_params[1]])
        return a, b

    def predict(self, config: ResourceConfig) -> FunctionTimes:
        assert config.workers > 0
        # key = num_vcpu + runtime_memory + num_workers
        key = self._config_to_xparam(config.cpu, config.memory, config.workers)
        predicted_read_time = io_func(key, *self._read_params)
        predicted_comp_time = comp_func(key, *self._comp_params)
        predicted_write_time = io_func(key, *self._write_params)
        total_predicted_time = (
            predicted_read_time
            + predicted_comp_time
            + predicted_write_time
            + self._cold_params
        )
        return FunctionTimes(
            total=total_predicted_time,
            read=predicted_read_time,
            compute=predicted_comp_time,
            write=predicted_write_time,
            cold_start=self._cold_params,
        )

    def optimize(self, config: ConfigBounds) -> ResourceConfig:
        raise NotImplementedError

    # def fit_polynomial(self, x, y, degree):
    #     coeffs = np.polyfit(x, y, degree)
    #     return np.poly1d(coeffs)

    # def visualize(self, step="compute", degree=2):
    #     assert step in ["read", "compute", "write"]
    #     data = self.profiling_results
    #     config_labels = ["num_vcpu", "runtime_memory", "num_workers", "chunk_size"]

    #     for label in config_labels:
    #         x = np.array([res["config"][config_labels.index(label)] for res in data])
    #         y = np.array([res[f"avg_{step}_time"] for res in data])

    #         if len(x) == 0 or len(y) == 0:
    #             print(f"Warning: No data to plot for {step} step with {label}")
    #             continue

    #         # Polynomial fitting
    #         poly = self.fit_polynomial(x, y, degree)
    #         x_fit = np.linspace(min(x), max(x), 100)
    #         y_fit = poly(x_fit)

    #         plt.figure()
    #         plt.scatter(x, y, label="Observed")
    #         plt.plot(x_fit, y_fit, "r-", label="Fitted")
    #         plt.xlabel(f"Config ({label})")
    #         plt.ylabel(f"{step.capitalize()} Time")
    #         plt.title(f"{step.capitalize()} Time vs {label.capitalize()}")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig(f"{step}_time_vs_{label}.png")
    #         plt.show()
