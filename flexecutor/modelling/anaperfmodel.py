from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
from workflow.stage import Stage


def phase_func(x, a, b):
    return a / x + b


coldstart_func = io_func = comp_func = phase_func


class AnaPerfModel(PerfModel):
    """
    AnaPerfModel records the mean parameter value.
    Advantage: it is fast and accurate enough to optimize the average performance.
    Shortcoming: it does not guarantee the bounded performance.

    Ditto, Caerus model.
    Adapted from https://github.com/pkusys/Jolteon/blob/main/workflow/perf_model_analytic.py
    """

    def __init__(self, stage: Stage) -> None:
        super().__init__("analytic", stage)

        # Init in train, list with size three
        self._write_params = None
        self._read_params = None
        self._comp_params = None
        self._cold_params = None

        self._profiling_results = None

    @classmethod
    def _config_to_xparam(cls, num_vcpu, memory, num_func):
        return round(num_vcpu * memory * num_func, 1)

    @overrides
    def save_model(self):
        pass

    @overrides
    def load_model(self):
        pass

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        if len(stage_profile_data) < 2:
            raise ValueError(
                "At least two profiled configurations for each stage are required to train the step model."
            )
        self._profiling_results = stage_profile_data

        for config_data in stage_profile_data.values():
            assert all(
                key in config_data for key in FunctionTimes.profile_keys()
            ), f"Each configuration's data must contain {FunctionTimes.profile_keys()} keys."

        # print(f"Training Analytical performance model for {self._stage_name}")

        size2points_coldstart = {}
        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}

        for config_tuple, data in stage_profile_data.items():
            num_vcpu, memory, num_func = config_tuple
            # adapt to parallel mode
            # if the stage does not allow more than one function, ignore num_func and set to 1
            num_vcpu = num_vcpu if self.allow_parallel else 1
            config_key = self._config_to_xparam(num_vcpu, memory, num_func)

            for size2points, phase in zip(
                [
                    size2points_coldstart,
                    size2points_read,
                    size2points_comp,
                    size2points_write,
                ],
                ["cold_start", "read", "compute", "write"],
            ):
                if config_key not in size2points:
                    size2points[config_key] = []
                size2points[config_key].extend(data[phase])

        for size2points in [
            size2points_coldstart,
            size2points_read,
            size2points_comp,
            size2points_write,
        ]:
            for config in size2points:
                size2points[config] = np.mean(size2points[config])

        # print(size2points_coldstart)
        # print(size2points_read)
        # print(size2points_comp)
        # print(size2points_write)

        def fit_params(data, func):
            assert isinstance(data, dict)
            arr_x = list(data.keys())
            arr_y = [data[x] for x in arr_x]

            arr_x = np.array(arr_x)
            arr_y = np.array(arr_y)

            initial_guess = [1, 1]

            def residuals(para, x, y):
                predicted = func(x, *para)
                residuals = predicted - y
                return residuals

            params, _ = scipy_opt.leastsq(residuals, initial_guess, args=(arr_x, arr_y))

            return params.tolist()

        # Fit the parameters
        # print("Fitting parameters...")
        self._cold_params = fit_params(size2points_coldstart, coldstart_func)
        self._read_params = fit_params(size2points_read, io_func)
        self._comp_params = fit_params(size2points_comp, comp_func)
        self._write_params = fit_params(size2points_write, io_func)

        # print(
        #     f"COLD START: alpha parameter = {self._cold_params[0]}, beta parameter = {self._cold_params[1]}"
        # )
        # print(
        #     f"READ STEP: alpha parameter = {self._read_params[0]}, beta parameter = {self._read_params[1]}"
        # )
        # print(
        #     f"COMPUTE STEP: alpha parameter = {self._comp_params[0]}, beta parameter = {self._comp_params[1]}"
        # )
        # print(
        #     f"WRITE_STEP: alpha parameter = {self._write_params[0]}, beta parameter = {self._write_params[1]}"
        # )

    @property
    @overrides
    def parameters(self):
        # parameter a (alpha), represents the paralelizable part, while beta is some non-paralellizable constant
        a = sum(
            [
                self._cold_params[0],
                self._read_params[0],
                self._comp_params[0],
                self._write_params[0],
            ]
        )
        b = sum(
            [
                self._cold_params[1],
                self._read_params[1],
                self._comp_params[1],
                self._write_params[1],
            ]
        )
        return a, b

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        assert config.workers > 0

        # FIXME: Use the parameter function to predict the time
        # key = num_vcpu + runtime_memory + num_workers
        key = self._config_to_xparam(config.cpu, config.memory, config.workers)
        predicted_read_time = io_func(key, *self._read_params)
        predicted_comp_time = comp_func(key, *self._comp_params)
        predicted_write_time = io_func(key, *self._write_params)
        predicted_cold_time = coldstart_func(key, *self._cold_params)
        total_predicted_time = (
            predicted_read_time
            + predicted_comp_time
            + predicted_write_time
            + predicted_cold_time
        )

        a, b = self.parameters
        print(
            f"Predicted time: {a} / {key} + {b} = {total_predicted_time} = {(a / key) + b}"
        )

        return FunctionTimes(
            total=total_predicted_time,
            read=predicted_read_time,
            compute=predicted_comp_time,
            write=predicted_write_time,
            cold_start=predicted_cold_time,
        )
