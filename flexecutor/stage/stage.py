import collections
import functools
import json
import os
import time
from contextlib import contextmanager
from typing import Optional, List, Tuple, Dict, Callable, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from lithops import FunctionExecutor, Storage

from flexecutor.modelling import AnaPerfModel
from flexecutor.utils import setup_logging


@contextmanager
def operation(op_type: str, timings: dict):
    start_time = time.time()
    yield
    end_time = time.time()
    timings[op_type] += end_time - start_time


def initialize_timings():
    return {"read": 0, "compute": 0, "write": 0}


def get_timings(timings: dict):
    return timings


def reset_timings(timings: dict):
    for key in timings:
        timings[key] = 0


class WorkflowStage:
    def __init__(
        self,
        name: str,
        function: Callable,
        input_data,
        output_data,
        model,
        config=None,
        storage_config=None,
    ):
        self.config = config if config else {}
        self.storage_config = storage_config if storage_config else {}
        self.storage = Storage(config=self.storage_config)
        self.log_level = config.get("log_level", "INFO")
        self.logger = setup_logging(self.log_level)
        self.function = function
        self.input_data = input_data
        self.profiling_file_name = f"{name}_profiling_results.json"
        self.output_data = output_data
        self.perf_model = model
        self.logger.info("WorkflowStage initialized")

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def get_perf_model(self):
        return self.perf_model

    def train(self):
        self.logger.info("Training model with profiling results")
        profiling_results = self.load_profiling_results()
        self.logger.info(profiling_results)
        self.perf_model.train(profiling_results)

    def predict(self):
        self.logger.info("Predicting best configuration using the trained model")
        best_config = self.perf_model.predict()
        self.logger.info(f"Predicted best configuration: {best_config}")
        return best_config

    def save_profiling_results(self, results):
        serializable_results = {str(k): v for k, v in results.items()}
        with open(self.profiling_file_name, "w") as f:
            json.dump(serializable_results, f, indent=4)

    def load_profiling_results(self):
        if not os.path.exists(self.profiling_file_name):
            self.logger.info(
                f"No existing profiling results found at {self.profiling_file_name}. Initializing empty results."
            )
            return {}
        try:
            with open(self.profiling_file_name, "r") as f:
                data = json.load(f)
                results = {eval(k): v for k, v in data.items()}
                return results
        except json.JSONDecodeError:
            self.logger.error(
                f"Error decoding JSON from {self.profiling_file_name}. Initializing empty results."
            )
            return {}

    def profile(self, config_space: List[Tuple[int, int, int]], num_iter: int):
        for num_cpu, runtime_memory, num_workers in config_space:
            self.config["runtime_memory"] = runtime_memory
            self.config["runtime_cpu"] = num_cpu
            self.config["workers"] = num_workers

            self.logger.info(
                f"Profiling with config: CPUs={num_cpu}, Memory={runtime_memory}MB, Workers={num_workers}"
            )

            for _ in range(num_iter):
                self.logger.info(f"Partitioning data into {num_workers} chunks.")
                self.run(obj_chunk_number=num_workers, activate_profiling=True)
            self.logger.debug(
                f"Config: {num_cpu} CPUs, {runtime_memory}MB Memory, {num_workers} Workers"
            )

    def wrap_user_function(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(parameters):
            result = fn(parameters)
            return result

        return wrapper

    def run(
        self,
        chunksize: Optional[int] = None,
        extra_args: Optional[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]] = None,
        extra_env: Optional[Dict[str, str]] = None,
        runtime_memory: Optional[int] = None,
        obj_chunk_size: Optional[int] = None,
        obj_chunk_number: Optional[int] = None,
        obj_newline: Optional[str] = "\n",
        timeout: Optional[int] = None,
        include_modules: Optional[List[str]] = [],
        exclude_modules: Optional[List[str]] = [],
        activate_profiling: Optional[bool] = None,
    ):
        include_modules = include_modules or []
        exclude_modules = exclude_modules or []

        self.__fexec = FunctionExecutor(**self.config)

        futures = self.__fexec.map(
            map_function=self.function,
            map_iterdata=self.input_data,
            chunksize=chunksize,
            extra_args=extra_args,
            extra_env=extra_env,
            runtime_memory=runtime_memory,
            obj_chunk_size=obj_chunk_size,
            obj_chunk_number=obj_chunk_number,
            obj_newline=obj_newline,
            timeout=timeout,
            include_modules=include_modules,
            exclude_modules=exclude_modules,
        )

        result = self.__fexec.get_result(fs=futures)

        worker_results = {"read": [], "compute": [], "write": [], "cold_start_time": []}
        for future, res in zip(futures, result):
            stats = future.stats
            host_submit_tstamp = stats["host_submit_tstamp"]
            worker_start_tstamp = stats["worker_start_tstamp"]
            res["cold_start_time"] = worker_start_tstamp - host_submit_tstamp
            for key, value in res.items():
                worker_results[key].append(value)

        if activate_profiling:
            results = self.load_profiling_results()
            config_key = (
                self.config["runtime_cpu"],
                self.config["runtime_memory"],
                self.config["workers"],
            )
            if config_key not in results:
                results[config_key] = {"read": [], "compute": [], "write": [], "cold_start_time": []}
            for i, value in worker_results.items():
                results[config_key][i].append(value)
            self.save_profiling_results(results)

        return worker_results

    def generate_objective_function(self):
        return self.perf_model.generate_objective_function()

    def plot_model_performance(self, config_space):
        actual_latencies = []
        predicted_latencies = []
        configurations = []
        resources = []

        profiling_data = self.load_profiling_results()

        for config in config_space:
            cpus, memory, workers = config
            total_resources = cpus * workers + memory * workers
            resources.append((total_resources, config))

        resources.sort()

        for total_resources, config in resources:
            cpus, memory, workers = config
            total_latencies = []

            if config in profiling_data:
                executions = profiling_data[config]
                total_latencies = [
                    sum(lats)
                    for breaks in zip(
                        executions["read"],
                        executions["compute"],
                        executions["write"],
                        executions["cold_start_time"],
                    )
                    for lats in zip(*breaks)
                ]
                avg_actual_latency = np.mean(total_latencies)
                actual_latencies.append(avg_actual_latency)
            else:
                actual_latencies.append(None)

            predicted_latency = self.perf_model.predict(cpus, memory, workers)
            predicted_latencies.append(predicted_latency)
            configurations.append(f"({cpus}, {memory}, {workers})")

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(configurations))

        ax.plot(x, predicted_latencies, label="Predicted Latencies", marker="x")

        if any(actual_latencies):
            ax.plot(x, actual_latencies, label="Actual Latencies", marker="o")

        ax.set_xlabel("Configurations")
        ax.set_ylabel("Latency")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(configurations, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig("model_performance.png")


if __name__ == "__main__":
    config = {"log_level": "INFO"}
    logger = setup_logging(config["log_level"])

    def word_occurrence_count(obj):
        timings = initialize_timings()
        storage = Storage()

        with operation("read", timings):
            data = obj.data_stream.read().decode("utf-8")

        with operation("compute", timings):
            words = data.split()
            word_count = collections.Counter(words)

        with operation("write", timings):
            result_key = (
                f"results_{obj.data_byte_range[0]}-{obj.data_byte_range[1]}.txt"
            )
            result_data = (
                f"Word Count: {len(word_count)}\nWord Frequencies: {dict(word_count)}\n"
            )

            storage.put_object(obj.bucket, result_key, result_data.encode("utf-8"))

        return timings

    ws = WorkflowStage(
        name="word_count",
        model=AnaPerfModel(1, "word_count"),
        function=word_occurrence_count,
        input_data="test-bucket/tiny_shakespeare.txt",
        output_data="test-bucket/tiny_shakespeare.txt",
        config=config,
    )

    ws(obj_chunk_number=5)

    ws.profile(
        config_space=[(2, 400, 5)],
        num_iter=2,
        # data_location="test-bucket/tiny_shakespeare.txt",
    )
