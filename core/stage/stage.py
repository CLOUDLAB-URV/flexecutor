import time
import logging
import functools
import json
from lithops import FunctionExecutor, Storage
from typing import Optional, List, Tuple, Dict, Callable, Any, Union
from contextlib import contextmanager
from flexexecutor.core.modelling import AnaPerfModel
import collections
import os


def setup_logging(level):
    # Loggers
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    debug_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s"
    info_format = "%(asctime)s [%(levelname)s] %(message)s"

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(debug_format, datefmt="%Y-%m-%d %H:%M:%S")
    debug_handler.setFormatter(debug_formatter)

    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_formatter = logging.Formatter(info_format, datefmt="%Y-%m-%d %H:%M:%S")
    info_handler.setFormatter(info_formatter)

    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.propagate = False

    return logger


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
        self.output_data = output_data
        self.perf_model = model
        self.logger.info("WorkflowStage initialized")

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def get_perf_model(self):
        return self.perf_model

    def train(self, profiling_results: List[Dict]):
        self.logger.info("Training model with profiling results")
        self.perf_model.train(profiling_results)

    def predict(self):
        self.logger.info("Predicting best configuration using the trained model")
        best_config = self.perf_model.predict()
        self.logger.info(f"Predicted best configuration: {best_config}")
        return best_config

    def save_profiling_results(self, results, filename):
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

    def load_profiling_results(self, filename):
        if not os.path.exists(filename):
            self.logger.info(
                f"No existing profiling results found at {filename}. Initializing empty results."
            )
            return []
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.logger.error(
                f"Error decoding JSON from {filename}. Initializing empty results."
            )
            return []

    def profile(
        self,
        config_space: List[Tuple[int, int, int]],
        num_iter: int,
        data_location: str,
        profiling_results_file: str = "profiling_results.json",
    ):
        existing_results = self.load_profiling_results(profiling_results_file)
        results = existing_results.copy()

        for (
            num_cpu,
            runtime_memory,
            num_workers,
        ) in config_space:
            try:
                self.config["runtime_memory"] = runtime_memory
                self.config["runtime_cpu"] = num_cpu
                self.config["workers"] = num_workers

                self.logger.debug(
                    f"Profiling with config: CPUs={num_cpu}, Memory={runtime_memory}MB, Workers={num_workers}"
                )

                execution_results = []

                for _ in range(num_iter):
                    self.logger.info(f"Partitioning data into {num_workers} chunks.")
                    result = self.run(
                        obj_chunk_number=num_workers, data_location=data_location
                    )

                    worker_stats = [
                        {
                            "read": r["read"],
                            "compute": r["compute"],
                            "write": r["write"],
                            "cold_start_time": r["cold_start_time"],
                        }
                        for r in result
                    ]
                    execution_results.append(worker_stats)

                results.append(
                    {
                        "config": (
                            num_cpu,
                            runtime_memory,
                            num_workers,
                        ),
                        "execution_results": execution_results,
                    }
                )

                self.logger.debug(
                    f"Config: {num_cpu} CPUs, {runtime_memory}MB Memory, {num_workers} Workers"
                )
                self.logger.debug(f"Execution results: {execution_results}")

                self.save_profiling_results(results, profiling_results_file)
            except Exception as e:
                self.logger.error(f"Error occurred: {e}")
                self.save_profiling_results(results, profiling_results_file)
                raise

        return results

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
        data_location: Optional[str] = None,
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

        for future, res in zip(futures, result):
            stats = future.stats
            host_submit_tstamp = stats["host_submit_tstamp"]
            worker_start_tstamp = stats["worker_start_tstamp"]
            cold_start_time = worker_start_tstamp - host_submit_tstamp
            res["cold_start_time"] = cold_start_time

        return result

    # In jolteon is implemented under the scheduler, and the scheduler recieves the workflow and optimizes the stages.
    # This is a much much simpler version of the optimal config. The find optimal config that jolteon first models the stage and then uses optimization methods (PCPSolver class)
    # Jolteon can also consider cost as well as latency, for now we just consider latency.
    def find_optimal_config_latency(self, bucket, key, function, dummy_configs):
        # Profile stage with the following configurations
        profiling_results_file = "profiling_results.json"

        if not os.path.exists(profiling_results_file):
            dataset_size = (
                int(Storage().head_object(bucket=bucket, key=key)["content-length"])
                / 1024**2
            )
            config_space = [
                (2, 400, 5),  # 2 CPUs, 200MB memory, 5 workers
                (8, 1600, 4),  # 8 CPUs, 1600MB memory, 4 workers
                (16, 3200, 2),  # 16 CPUs, 3200MB memory, 2 workers
                (32, 6400, 1),  # 32 CPUs, 6400MB memory, 1 worker
                (3, 600, 4),  # 3 CPUs, 600MB memory, 4 workers
                (6, 1200, 3),  # 6 CPUs, 1200MB memory, 3 workers
                (12, 2400, 2),  # 12 CPUs, 2400MB memory, 2 workers
                (24, 4800, 1),  # 24 CPUs, 4800MB memory, 1 worker
                (6, 600, 6),  # 6 CPUs, 600MB memory, 6 workers
                (9, 900, 3),  # 9 CPUs, 900MB memory, 3 workers
                (18, 1800, 2),  # 18 CPUs, 1800MB memory, 2 workers
            ]

            self.profile(
                config_space,
                num_iter=2,
                data_location=f"{bucket}/{key}",
            )

        self.perf_model.train(self.load_profiling_results("profiling_results.json"))

        # Predict the optimal configuration in new configs out of the training stage, using the trained model
        best_config = None
        best_metric = float("inf")

        for config in dummy_configs:
            num_vcpu, memory, workers = config
            prediction = self.perf_model.predict(num_vcpu, memory, workers)
            total_time = prediction["total_predicted_time"]

            if total_time < best_metric:
                best_metric = total_time
                best_config = config

        self.logger.info(
            f"Optimal configuration found: CPUs={best_config[0]}, Memory={best_config[1]}MB, Workers={best_config[2]}, Total Latency={best_metric}"
        )
        return best_config

    def generate_objective_function(ftype: str):
        if ftype == "latency":
            return "Dummy objective function for latency"

        if ftype == "cost":
            return "Dummy objective function for cost"


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
