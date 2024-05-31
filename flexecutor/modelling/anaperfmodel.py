from itertools import chain
from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.modelling.prediction import Prediction


def io_func(x, a, b, c):
    return a / (x + c) + b


def comp_func(x, a, b, c):
    return a / (x + c) + b


class AnaPerfModel(PerfModel):
    def __init__(self, stage_id, stage_name) -> None:
        assert isinstance(stage_name, str)
        assert isinstance(stage_id, int) and stage_id >= 0
        super().__init__("analytic")

        self._stage_name = stage_name
        self._stage_id = stage_id
        self._objective_func = "dummy_func(x, a, b, c)"

        self._allow_parallel = True

        # Init in train, list with size three
        self._write_params = None
        self._read_params = None
        self._comp_params = None
        self._cold_params = None

        self._profiling_results = None

    # TODO: review that and rethink
    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self._allow_parallel = allow_parallel

    @overrides
    def train(self, profiling_results: Dict) -> None:
        print("Training Analytical performance model for %s" % self._stage_name)

        average_dict = {}
        for i in ['read', 'compute', 'write', 'cold_start_time']:
            average_dict[i] = np.array(
                [np.mean(i) for i in list(chain(*([value[i] for key, value in profiling_results.items()])))])

        self._cold_params = np.mean(average_dict['cold_start_time'])

        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}

        for idx, config in enumerate(profiling_results.keys()):
            # config = eval(config)
            num_vcpu = config[0]
            runtime_memory = config[1]
            num_workers = config[2]
            chunk_size = config[3] if len(config) > 3 else 256

            key = (num_vcpu, runtime_memory, num_workers, chunk_size)

            if key not in size2points_read:
                size2points_read[key] = []
            size2points_read[key].append(average_dict['read'][idx])

            if key not in size2points_comp:
                size2points_comp[key] = []
            size2points_comp[key].append(average_dict['compute'][idx])

            if key not in size2points_write:
                size2points_write[key] = []
            size2points_write[key].append(average_dict['write'][idx])

        for config in size2points_read:
            size2points_read[config] = np.mean(size2points_read[config])
        for config in size2points_comp:
            size2points_comp[config] = np.mean(size2points_comp[config])
        for config in size2points_write:
            size2points_write[config] = np.mean(size2points_write[config])

        # Flatten the keys and corresponding values for fitting
        flattened_read = {
            sum([vcpu, mem, workers, chunk]): size2points_read[
                (vcpu, mem, workers, chunk)
            ]
            for vcpu, mem, workers, chunk in size2points_read
        }
        flattened_comp = {
            sum([vcpu, mem, workers, chunk]): size2points_comp[
                (vcpu, mem, workers, chunk)
            ]
            for vcpu, mem, workers, chunk in size2points_comp
        }
        flattened_write = {
            sum([vcpu, mem, workers, chunk]): size2points_write[
                (vcpu, mem, workers, chunk)
            ]
            for vcpu, mem, workers, chunk in size2points_write
        }

        def fit_params(data, func):
            assert isinstance(data, dict)
            arr_x = list(data.keys())
            arr_y = [data[x] for x in arr_x]

            arr_x = np.array(arr_x)
            arr_y = np.array(arr_y).flatten()

            initial_guess = [1, 1, 1]

            def residuals(para, x, y):
                return func(x, *para) - y

            params, _ = scipy_opt.leastsq(residuals, initial_guess, args=(arr_x, arr_y))

            return params.tolist()

        # Fit the parameters
        self._read_params = fit_params(flattened_read, io_func)
        self._comp_params = fit_params(flattened_comp, comp_func)
        self._write_params = fit_params(flattened_write, io_func)

        self._profiling_results = profiling_results

    @property
    @overrides
    def parameters(self):
        a = sum([self._read_params[0], self._comp_params[0], self._write_params[0]])
        b = sum([self._read_params[1], self._comp_params[1], self._write_params[1]])
        c = sum([self._read_params[2], self._comp_params[2], self._write_params[2]])
        return a, b, c

    def predict(
            self,
            num_vcpu,
            runtime_memory,
            num_workers,
            chunk_size=None,
    ) -> Prediction:
        assert num_workers > 0
        key = num_vcpu + runtime_memory + num_workers + chunk_size
        predicted_read_time = io_func(key, *self._read_params) / num_workers
        predicted_comp_time = comp_func(key, *self._comp_params) / num_workers
        predicted_write_time = io_func(key, *self._write_params) / num_workers
        total_predicted_time = (
                predicted_read_time
                + predicted_comp_time
                + predicted_write_time
                + self._cold_params
        )
        return Prediction(
            total_time=total_predicted_time,
            read_time=predicted_read_time,
            compute_time=predicted_comp_time,
            write_time=predicted_write_time,
            cold_start_time=self._cold_params,
        )

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
