import numpy as np
import scipy.optimize as scipy_opt
from typing import List, Dict


def io_func(x, a, b, c):
    return a / (x + c) + b


def comp_func(x, a, b, c):
    return a / (x + c) + b


class AnaPerfModel:
    def __init__(self, _stage_id, _stage_name) -> None:
        assert isinstance(_stage_name, str)
        assert isinstance(_stage_id, int) and _stage_id >= 0
        self.stage_name = _stage_name
        self.stage_id = _stage_id

        self.allow_parallel = True

        # Init in train, list with size three
        self.write_params = None
        self.read_params = None
        self.comp_params = None
        self.cold_params = None

        self.profiling_results = None

    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self.allow_parallel = allow_parallel

    def fit_params(self, data, func):
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

    def fit_polynomial(self, x, y, degree):
        coeffs = np.polyfit(x, y, degree)
        return np.poly1d(coeffs)

    def train(self, profiling_results: List[Dict]) -> None:
        print("Training Analytical performance model for %s" % self.stage_name)

        read_arr = np.array([res["avg_read_time"] for res in profiling_results])
        comp_arr = np.array([res["avg_compute_time"] for res in profiling_results])
        write_arr = np.array([res["avg_write_time"] for res in profiling_results])
        cold_arr = np.array([res["avg_cold_start_time"] for res in profiling_results])

        self.cold_params = np.mean(cold_arr)

        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}

        for idx, res in enumerate(profiling_results):
            config = res["config"]
            num_vcpu = config[0]
            runtime_memory = config[1]
            num_workers = config[2]
            chunk_size = config[3]  # Include chunk size in the config

            key = (num_vcpu, runtime_memory, num_workers, chunk_size)

            if key not in size2points_read:
                size2points_read[key] = []
            size2points_read[key].append(read_arr[idx])

            if key not in size2points_comp:
                size2points_comp[key] = []
            size2points_comp[key].append(comp_arr[idx])

            if key not in size2points_write:
                size2points_write[key] = []
            size2points_write[key].append(write_arr[idx])

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

        # Fit the parameters
        self.read_params = self.fit_params(flattened_read, io_func)
        self.comp_params = self.fit_params(flattened_comp, comp_func)
        self.write_params = self.fit_params(flattened_write, io_func)

        self.profiling_results = profiling_results

    def get_params(self):
        a = sum([self.read_params[0], self.comp_params[0], self.write_params[0]])
        b = sum([self.read_params[1], self.comp_params[1], self.write_params[1]])
        c = sum([self.read_params[2], self.comp_params[2], self.write_params[2]])
        return a, b, c

    def predict(
        self,
        num_vcpu,
        runtime_memory,
        num_workers,
        chunk_size,
    ):
        assert num_workers > 0
        key = num_vcpu + runtime_memory + num_workers + chunk_size
        predicted_read_time = io_func(key, *self.read_params) / num_workers
        predicted_comp_time = comp_func(key, *self.comp_params) / num_workers
        predicted_write_time = io_func(key, *self.write_params) / num_workers
        total_predicted_time = (
            predicted_read_time
            + predicted_comp_time
            + predicted_write_time
            + self.cold_params
        )
        return {
            "predicted_read_time": predicted_read_time,
            "predicted_comp_time": predicted_comp_time,
            "predicted_write_time": predicted_write_time,
            "predicted_cold_start_time": self.cold_params,
            "total_predicted_time": total_predicted_time,
        }

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

    def generate_obj_func_code(self):
        return "dummy_func(x, a, b, c)"


if __name__ == "__main__":
    profiling_results = [
        {
            "config": (1, 1024, 4, 64),
            "avg_read_time": 0.5,
            "avg_compute_time": 1.0,
            "avg_write_time": 0.3,
            "avg_cold_start_time": 0.2,
        },
        {
            "config": (2, 2048, 8, 128),
            "avg_read_time": 0.4,
            "avg_compute_time": 0.9,
            "avg_write_time": 0.25,
            "avg_cold_start_time": 0.15,
        },
        {
            "config": (3, 3072, 16, 256),
            "avg_read_time": 0.3,
            "avg_compute_time": 0.8,
            "avg_write_time": 0.2,
            "avg_cold_start_time": 0.1,
        },
    ]
    perfmodel = AnaPerfModel(1, "stage1")
    perfmodel.update_allow_parallel(True)
    perfmodel.train(profiling_results)

    print(perfmodel.get_params())

    perfmodel.visualize(step="compute", degree=2)
    perfmodel.visualize(step="read", degree=2)
    perfmodel.visualize(step="write", degree=2)

    # Generate function code for latency
    print(perfmodel.generate_func_code())
