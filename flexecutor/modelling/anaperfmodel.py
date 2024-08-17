import os
import math
from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides
from itertools import cycle
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
import matplotlib.pyplot as plt

LAMBDA_PRICE_PER_GB_SECOND = 0.0000166667
MEMORY_PER_CPU = 1769


def get_vcpus(memory_mb):
    return math.ceil(memory_mb / MEMORY_PER_CPU)


def coldstart_func(x, a, b):
    return np.maximum(0, a / x + b)


def io_func(workers, a, b):
    return np.maximum(0, a / workers + b)


def comp_func(x, a, b):
    return np.maximum(0, a / x + b)


class AnaPerfModel(PerfModel):
    """
    AnaPerfModel records the mean parameter value.
    Advantage: it is fast and accurate enough to optimize the average performance.
    Shortcoming: it does not guarantee the bounded performance.

    Ditto, Caerus model.
    Adapted from https://github.com/pkusys/Jolteon/blob/main/workflow/perf_model_analytic.py
    """

    def __init__(
        self, model_name, model_dst, stage_id, stage_name, compute_scaling
    ) -> None:
        assert isinstance(stage_name, str)
        print("Stage ID: ", stage_id)
        assert isinstance(stage_id, str) and stage_id is not None
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
        self._compute_scaling = compute_scaling
        assert compute_scaling in [
            "cpu",
            "worker",
        ], "compute_scaling must be either 'cpu' or 'worker'"

    @classmethod
    def _config_to_xparam(cls, num_cpu, num_func):
        return round(num_cpu * num_func, 3)

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
        if len(stage_profile_data) < 2:
            raise ValueError(
                "At least two profiled configurations for each stage are required to train the step model."
            )
        self._profiling_results = stage_profile_data

        for config_data in stage_profile_data.values():
            assert all(
                key in config_data for key in FunctionTimes.profile_keys()
            ), f"Each configuration's data must contain {FunctionTimes.profile_keys()} keys."

        print(f"Training Analytical performance model for {self._stage_name}")

        size2points_coldstart = {}
        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}

        print(stage_profile_data)
        for config_tuple, data in stage_profile_data.items():
            print(f"Training for config: {config_tuple}")
            num_cpu, _, num_workers = config_tuple
            config_key = self._config_to_xparam(num_cpu, num_workers)

            if self._compute_scaling == "cpu":
                comp_key = self._config_to_xparam(num_cpu, num_workers)
            else:  # worker scaling
                comp_key = num_workers

            worker_key = num_workers

            def average_worker_times(data):
                return [np.mean(worker_data) for worker_data in zip(*data)]

            # collect and flatten data for cold_start
            if config_key not in size2points_coldstart:
                size2points_coldstart[config_key] = []
            size2points_coldstart[config_key].extend(
                average_worker_times(data["cold_start"])
            )

            # collect and flatten data for read step
            if worker_key not in size2points_read:
                size2points_read[worker_key] = []
            size2points_read[worker_key].extend(average_worker_times(data["read"]))

            # collect and flatten data for comp step
            if comp_key not in size2points_comp:
                size2points_comp[comp_key] = []
            size2points_comp[comp_key].extend(average_worker_times(data["compute"]))

            # collect and flatten data for write step
            if worker_key not in size2points_write:
                size2points_write[worker_key] = []
            size2points_write[worker_key].extend(average_worker_times(data["write"]))

        # average the data
        size2points_coldstart = {
            k: np.mean(v) for k, v in size2points_coldstart.items()
        }
        size2points_read = {k: np.mean(v) for k, v in size2points_read.items()}
        size2points_comp = {k: np.mean(v) for k, v in size2points_comp.items()}
        size2points_write = {k: np.mean(v) for k, v in size2points_write.items()}

        print(size2points_coldstart)
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
                predicted = func(x, *para)
                residuals = predicted - y
                return residuals

            params, _ = scipy_opt.leastsq(residuals, initial_guess, args=(arr_x, arr_y))

            return params.tolist()

        # Fit the parameters
        print("Fitting parameters...")
        self._cold_params = fit_params(size2points_coldstart, coldstart_func)
        self._read_params = fit_params(size2points_read, io_func)
        self._comp_params = fit_params(size2points_comp, comp_func)
        self._write_params = fit_params(size2points_write, io_func)

        print(
            f"COLD START: alpha parameter = {self._cold_params[0]}, beta parameter = {self._cold_params[1]}"
        )
        print(
            f"READ STEP: alpha parameter = {self._read_params[0]}, beta parameter = {self._read_params[1]}"
        )
        print(
            f"COMPUTE STEP: alpha parameter = {self._comp_params[0]}, beta parameter = {self._comp_params[1]}"
        )
        print(
            f"WRITE STEP: alpha parameter = {self._write_params[0]}, beta parameter = {self._write_params[1]}"
        )

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

        cpu_key = self._config_to_xparam(config.cpu, config.workers)
        worker_key = config.workers

        predicted_read_time = io_func(worker_key, *self._read_params)
        if self._compute_scaling == "cpu":
            predicted_comp_time = comp_func(cpu_key, *self._comp_params)
        else:  # worker scaling
            predicted_comp_time = comp_func(worker_key, *self._comp_params)
        predicted_write_time = io_func(worker_key, *self._write_params)
        predicted_cold_time = coldstart_func(cpu_key, *self._cold_params)
        total_predicted_time = (
            predicted_read_time
            + predicted_comp_time
            + predicted_write_time
            + predicted_cold_time
        )

        return FunctionTimes(
            total=total_predicted_time,
            read=predicted_read_time,
            compute=predicted_comp_time,
            write=predicted_write_time,
            cold_start=predicted_cold_time,
        )

    def calculate_cost(self, config: StageConfig) -> float:
        predicted_times = self.predict_time(config)
        execution_time_seconds = predicted_times.total

        total_memory_mb = config.cpu * MEMORY_PER_CPU * config.workers
        gb_seconds = execution_time_seconds * (total_memory_mb / 1024.0)
        execution_cost = gb_seconds * LAMBDA_PRICE_PER_GB_SECOND

        return execution_cost

    def plot(self, config_bounds: ConfigBounds):
        fig = plt.figure(figsize=(20, 30))
        fig.suptitle(f"Performance Model for {self._stage_name}", fontsize=16)

        # Create 6 subplots: 4 2D plots for components, 2 2D plots for total time and cost
        ax_read = fig.add_subplot(321)
        ax_compute = fig.add_subplot(322)
        ax_write = fig.add_subplot(323)
        ax_cold = fig.add_subplot(324)
        ax_total = fig.add_subplot(325)
        ax_cost = fig.add_subplot(326)

        min_total_cpus = config_bounds.cpu[0] * config_bounds.workers[0]
        max_total_cpus = config_bounds.cpu[1] * config_bounds.workers[1]
        total_cpu_range = np.linspace(min_total_cpus, max_total_cpus, 100)

        def get_workers(total_cpu):
            return np.clip(
                total_cpu / config_bounds.cpu[1],
                config_bounds.workers[0],
                config_bounds.workers[1],
            )

        worker_range = get_workers(total_cpu_range)

        # Calculate times for each component
        read_times = io_func(worker_range, *self._read_params)
        write_times = io_func(worker_range, *self._write_params)
        if self._compute_scaling == "cpu":
            compute_times = comp_func(total_cpu_range, *self._comp_params)
            compute_x_range = total_cpu_range
            compute_x_label = "Total Number of CPUs"
        else:  # worker scaling
            compute_times = comp_func(worker_range, *self._comp_params)
            compute_x_range = worker_range
            compute_x_label = "Number of Workers"
        cold_start_times = coldstart_func(total_cpu_range, *self._cold_params)

        # Calculate total time
        total_times = read_times + write_times + compute_times + cold_start_times

        # Calculate costs
        costs = []
        for total_cpu in total_cpu_range:
            workers = get_workers(total_cpu)
            cpus_per_worker = total_cpu / workers
            memory_per_worker = cpus_per_worker * MEMORY_PER_CPU
            temp_config = StageConfig(
                cpu=cpus_per_worker, memory=memory_per_worker, workers=workers
            )
            cost = self.calculate_cost(temp_config)
            costs.append(cost)

        # Plotting 2D components
        ax_read.plot(worker_range, read_times, label="Predicted Read", color="black")
        ax_compute.plot(
            compute_x_range, compute_times, label="Predicted Compute", color="black"
        )
        ax_write.plot(worker_range, write_times, label="Predicted Write", color="black")
        ax_cold.plot(
            total_cpu_range,
            cold_start_times,
            label="Predicted Cold Start",
            color="black",
        )

        # 2D contour plots for total execution time and cost
        workers, cpus = np.meshgrid(
            np.linspace(config_bounds.workers[0], config_bounds.workers[1], 20),
            np.linspace(config_bounds.cpu[0], config_bounds.cpu[1], 20),
        )

        total_times_2d = np.zeros_like(workers)
        costs_2d = np.zeros_like(workers)
        for i in range(workers.shape[0]):
            for j in range(workers.shape[1]):
                config = StageConfig(
                    cpu=cpus[i, j],
                    memory=cpus[i, j] * MEMORY_PER_CPU,
                    workers=workers[i, j],
                )
                total_times_2d[i, j] = self.predict_time(config).total
                costs_2d[i, j] = self.calculate_cost(config)

        # Total Execution Time plot
        time_contour = ax_total.contourf(
            workers, cpus, total_times_2d, levels=20, cmap="viridis"
        )
        ax_total.contour(
            workers, cpus, total_times_2d, levels=10, colors="k", alpha=0.3
        )
        ax_total.set_xlabel("Number of Workers")
        ax_total.set_ylabel("CPUs per Worker")
        cbar_time = fig.colorbar(time_contour, ax=ax_total)
        cbar_time.set_label("Execution Time (s)")

        # Cost plot
        cost_contour = ax_cost.contourf(
            workers, cpus, costs_2d, levels=20, cmap="viridis"
        )
        ax_cost.contour(workers, cpus, costs_2d, levels=10, colors="k", alpha=0.3)
        ax_cost.set_xlabel("Number of Workers")
        ax_cost.set_ylabel("CPUs per Worker")
        cbar_cost = fig.colorbar(cost_contour, ax=ax_cost)
        cbar_cost.set_label("Cost ($)")

        # Plotting profiled data points and adding annotations
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self._profiling_results)))
        for (num_cpu, memory, num_workers), color in zip(
            self._profiling_results.keys(), colors
        ):
            config = StageConfig(cpu=num_cpu, memory=memory, workers=num_workers)
            predicted_time = self.predict_time(config).total
            cost = self.calculate_cost(config)

            # Calculate actual total time from components
            actual_time = np.mean(
                [
                    np.mean(
                        self._profiling_results[(num_cpu, memory, num_workers)]["read"]
                    )
                    + np.mean(
                        self._profiling_results[(num_cpu, memory, num_workers)][
                            "compute"
                        ]
                    )
                    + np.mean(
                        self._profiling_results[(num_cpu, memory, num_workers)]["write"]
                    )
                    + np.mean(
                        self._profiling_results[(num_cpu, memory, num_workers)][
                            "cold_start"
                        ]
                    )
                ]
            )

            # Calculate the color based on the difference between actual and predicted time
            time_diff = actual_time - predicted_time
            max_time_diff = max(
                abs(np.min(total_times_2d) - np.max(total_times_2d)), abs(time_diff)
            )
            time_color_val = (time_diff + max_time_diff) / (2 * max_time_diff)
            time_color = time_contour.cmap(time_color_val)

            ax_total.scatter(
                num_workers,
                num_cpu,
                color=time_color,
                s=150,
                edgecolor="black",
                linewidth=2,
                zorder=5,
            )
            ax_total.annotate(
                f"A: {actual_time:.2f}s\nP: {predicted_time:.2f}s",
                (num_workers, num_cpu),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=1),
                zorder=6,
            )
            ax_total.annotate(
                f"({num_cpu},{memory},{num_workers})",
                (num_workers, num_cpu),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=1),
                zorder=6,
            )

            # plot on cost plot
            cost_color = cost_contour.cmap(cost_contour.norm(cost))
            ax_cost.scatter(
                num_workers,
                num_cpu,
                color=cost_color,
                s=150,
                edgecolor="black",
                linewidth=2,
                zorder=5,
            )
            ax_cost.annotate(
                f"${cost:.6f}",
                (num_workers, num_cpu),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=1),
                zorder=6,
            )
            ax_cost.annotate(
                f"({num_cpu},{memory},{num_workers})",
                (num_workers, num_cpu),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=1),
                zorder=6,
            )

            # plots component-wise data points with original colors
            config_label = f"({num_cpu},{memory},{num_workers})"
            ax_read.scatter(
                num_workers,
                np.mean(
                    self._profiling_results[(num_cpu, memory, num_workers)]["read"]
                ),
                color=color,
                marker="o",
                label=config_label,
            )
            ax_compute.scatter(
                (
                    num_workers
                    if self._compute_scaling == "worker"
                    else num_cpu * num_workers
                ),
                np.mean(
                    self._profiling_results[(num_cpu, memory, num_workers)]["compute"]
                ),
                color=color,
                marker="o",
                label=config_label,
            )
            ax_write.scatter(
                num_workers,
                np.mean(
                    self._profiling_results[(num_cpu, memory, num_workers)]["write"]
                ),
                color=color,
                marker="o",
                label=config_label,
            )
            ax_cold.scatter(
                num_cpu * num_workers,
                np.mean(
                    self._profiling_results[(num_cpu, memory, num_workers)][
                        "cold_start"
                    ]
                ),
                color=color,
                marker="o",
                label=config_label,
            )

        # add legends to component plots
        for ax in [ax_read, ax_compute, ax_write, ax_cold]:
            ax.legend(
                title="Configurations",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
            )

        # setting titles and labels
        ax_read.set_title("Read Time")
        ax_compute.set_title("Compute Time")
        ax_write.set_title("Write Time")
        ax_cold.set_title("Cold Start Time")
        ax_total.set_title("Total Execution Time")
        ax_cost.set_title("Cost")

        # Set x-axis labels
        ax_read.set_xlabel("Number of Workers")
        ax_write.set_xlabel("Number of Workers")
        ax_compute.set_xlabel(compute_x_label)
        ax_cold.set_xlabel("Total Number of CPUs")

        for ax in [ax_read, ax_compute, ax_write, ax_cold, ax_total, ax_cost]:
            ax.grid(True)
            ax.set_ylabel(
                "Time (s)" if ax not in [ax_cost, ax_total] else "CPUs per Worker"
            )

        plt.tight_layout()
        plot_dir = "./plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{self._stage_id}_performance_model.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Performance model plot saved at {plot_path}")
        plt.close(fig)
