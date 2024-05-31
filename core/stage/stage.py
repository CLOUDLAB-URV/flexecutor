from lithops import FunctionExecutor, Storage
from typing import Optional, List, Tuple, Dict, Callable, Any, Union
from contextlib import contextmanager
from flexexecutor.core.modelling import AnaPerfModel
from flexexecutor.core.utils import setup_logging

import time
import functools
import json
import collections
import os


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
        self.model_file_name = f"{name}_model.pkl"
        self.output_data = output_data
        self.perf_model = model
        self.logger.info("WorkflowStage initialized")

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __save_model(self, file_name):
        self.perf_model.save_model(file_name)

    def __load_model(self, file_name):
        self.perf_model.load_model(file_name)

    # TODO: implement the api so there's no need to interact with the performance model directly, but through wrapping methods
    def get_perf_model(self):
        return self.perf_model

    def update_config(self, cpu, memory, workers):
        self.config["runtime_cpu"] = cpu
        self.config["runtime_memory"] = memory
        self.config["workers"] = workers

    def profile(self, config_space: List[Tuple[int, int, int]], num_iter: int):
        for num_cpu, runtime_memory, num_workers in config_space:
            self.config["runtime_memory"] = runtime_memory
            self.config["runtime_cpu"] = num_cpu
            self.config["workers"] = num_workers
            for _ in range(num_iter):
                self.logger.info(f"Partitioning data into {num_workers} chunks.")
                self.run(activate_profiling=True)

    def train(self):
        self.logger.info("Training model with profiling results")
        profiling_results = self.load_profiling_results()
        self.logger.info(profiling_results)
        self.perf_model.train(profiling_results)
        self.__save_model(self.model_file_name)

    def predict_latency(self, cpu, memory, workers):
        return self.perf_model.predict_latency(cpu, memory, workers)

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
        # obj_chunk_number: Optional[int] = None,
        obj_newline: Optional[str] = "\n",
        timeout: Optional[int] = None,
        include_modules: Optional[List[str]] = [],
        exclude_modules: Optional[List[str]] = [],
        activate_profiling: Optional[bool] = None,
    ):

        self.logger.info(
            f"Running with configuration #CPUs: {self.config['runtime_cpu']}, Memory: {self.config['runtime_memory']}MB, Workers: {self.config['workers']}"
        )
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
            obj_chunk_number=self.config["workers"],
            obj_newline=obj_newline,
            timeout=timeout,
            include_modules=include_modules,
            exclude_modules=exclude_modules,
        )

        result = self.__fexec.get_result(fs=futures)

        worker_results = []
        for future, res in zip(futures, result):
            stats = future.stats
            host_submit_tstamp = stats["host_submit_tstamp"]
            worker_start_tstamp = stats["worker_start_tstamp"]
            cold_start_time = worker_start_tstamp - host_submit_tstamp
            res["cold_start_time"] = cold_start_time
            worker_results.append(res)

        if activate_profiling:
            results = self.load_profiling_results()
            config_key = (
                self.config["runtime_cpu"],
                self.config["runtime_memory"],
                self.config["workers"],
            )
            if config_key not in results:
                results[config_key] = []
            results[config_key].append(worker_results)
            self.save_profiling_results(results)

        return worker_results

    def get_objective_function(self):
        self.__load_model(self.model_file_name)
        return self.perf_model.get_objective_function()


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
        data_location="test-bucket/tiny_shakespeare.txt",
    )
