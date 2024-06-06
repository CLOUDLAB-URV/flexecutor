import logging
import os
from typing import Dict, Set, List, Iterable, Optional

import numpy as np
import pandas as pd
from lithops import FunctionExecutor
from matplotlib import pyplot as plt
from pandas import DataFrame

from flexecutor.utils.dataclass import FunctionTimes, ResourceConfig, ConfigBounds
from flexecutor.utils.utils import load_profiling_results, save_profiling_results
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.processors import ThreadPoolProcessor
from flexecutor.workflow.stage import Stage
from flexecutor.workflow.stagefuture import StageFuture

logger = logging.getLogger(__name__)


class DAGExecutor:
    """
    Executor class that is responsible for executing the DAG

    :param dag: DAG to execute
    :param executor: Executor to use for the stages
    """

    def __init__(
            self,
            dag: DAG,
            executor: FunctionExecutor | None = None,
    ):
        self._dag = dag
        self._processor = ThreadPoolProcessor(executor)

        self._futures: Dict[str, StageFuture] = dict()
        self._num_final_stages = 0
        self._dependence_free_stages: List[Stage] = list()
        self._running_stages: List[Stage] = list()
        self._finished_stages: Set[Stage] = set()
        self._executor = executor

    def _get_timings(self, futures: dict[str, StageFuture]) -> List[FunctionTimes]:
        """Get the timings of the futures."""
        timings_list = []
        for stage_id, future in futures.items():
            results = future.result()
            stats = future.stats
            try:
                for r, s in zip(results, stats):
                    host_submit_tstamp = s["host_submit_tstamp"]
                    worker_start_tstamp = s["worker_start_tstamp"]
                    r["cold_start"] = worker_start_tstamp - host_submit_tstamp
                    timings_list.append(
                        FunctionTimes(read=r["read"],
                                      compute=r["compute"],
                                      write=r["write"],
                                      cold_start=r["cold_start"],
                                      total=r["read"] + r["compute"] + r["write"] + r["cold_start"])
                    )
            except KeyError:
                logger.error(
                    f'Error getting timings for stage {stage_id}. Please review the return values of map function')
        return timings_list

    def _store_profiling(self, file: str, new_profile_data: List[FunctionTimes],
                         config_space: ResourceConfig) -> None:
        profile_data = load_profiling_results(file)
        config_key = config_space.key
        if config_key not in profile_data:
            profile_data[config_key] = {}
        for key in FunctionTimes.profile_keys():
            if key not in profile_data[config_key]:
                profile_data[config_key][key] = []
            profile_data[config_key][key].append([])
        for profiling in new_profile_data:
            for key in FunctionTimes.profile_keys():
                profile_data[config_key][key][-1].append(getattr(profiling, key))
        save_profiling_results(file, profile_data)

    def profile(self,
                config_spaces: Iterable[ResourceConfig],
                stage: Optional[Stage] = None,
                num_iterations: int = 1) -> None:
        """Profile the DAG."""
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage in stages_list:
            os.makedirs(f"profiling/{self._dag.dag_id}", exist_ok=True)
            profiling_file = f"profiling/{self._dag.dag_id}/{stage.stage_id}.json"
            for config_space in config_spaces:
                for iteration in range(num_iterations):
                    timings = self.run_stage(stage, config_space)
                    self._store_profiling(profiling_file, timings, config_space)

    def predict(self,
                config_space: ResourceConfig,
                stage: Optional[Stage] = None
                ) -> List[FunctionTimes]:
        result = []
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage in stages_list:
            result.append(stage.perf_model.predict(config_space))
        return result

    def run_stage(self, stage: Stage, config_space: ResourceConfig) -> List[FunctionTimes]:
        """Run a stage with a given configuration space."""
        # Set the parameters in Lithops config
        self._executor.config['runtime_cpu'] = config_space.cpu
        self._executor.config['runtime_memory'] = config_space.memory
        self._executor.config['workers'] = config_space.workers
        logger.info(f'Running stage {stage.stage_id} with config {config_space}')
        # Execute the stage
        futures = self._processor.process([stage])
        # Store the profiling data
        timings = self._get_timings(futures)
        return timings

    def train(self,
              stage: Optional[Stage] = None) -> None:
        """Train the DAG."""
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage in stages_list:
            profile_data = load_profiling_results(f"profiling/{self._dag.dag_id}/{stage.stage_id}.json")
            stage.perf_model.train(profile_data)
            stage.perf_model.save_model()

    def execute(self) -> Dict[str, StageFuture]:
        """
        Execute the DAG

        :return: A dictionary with the output data of the DAG stages with the stage ID as key
        """
        logger.info(f'Executing DAG {self._dag.dag_id}')

        self._num_final_stages = len(self._dag.leaf_stages)
        logger.info(f'DAG {self._dag.dag_id} has {self._num_final_stages} final stages')

        self._futures = dict()

        # Start by executing the root stages
        self._dependence_free_stages = set(self._dag.root_stages)
        self._running_stages = set()
        self._finished_stages = set()

        # Execute stages until all stages have been executed
        while self._dependence_free_stages or self._running_stages:
            # Select the stages to execute
            batch = list(self._dependence_free_stages)

            # Add the batch to the running stages
            set_batch = set(batch)
            self._running_stages |= set_batch

            # Call the processor to execute the batch
            futures = self._processor.process(batch)

            self._running_stages -= set_batch
            self._dependence_free_stages -= set_batch
            self._finished_stages |= set_batch

            for stage in batch:
                self._futures[stage.stage_id] = futures[stage.stage_id]
                for child in stage.children:
                    if child.parents.issubset(self._finished_stages):
                        self._dependence_free_stages.add(child)

        return self._futures

    def model_perf_metrics(self, stage: Stage, config_spaces: List[ResourceConfig]) -> DataFrame:
        actual_latencies, predicted_latencies = self._prediction_vs_actual(stage, config_spaces)

        actual_latencies = np.array(actual_latencies)
        predicted_latencies = np.array(predicted_latencies)

        data = np.array([
            [config.workers, config.cpu, config.memory, actual, predicted, abs(actual - predicted),
             (actual - predicted) ** 2]
            for config, actual, predicted in zip(config_spaces, actual_latencies, predicted_latencies)
        ])

        df = pd.DataFrame(data, columns=["Workers",
                                         "CPU",
                                         "Memory",
                                         "Actual latency",
                                         "Predicted latency",
                                         "MAE",
                                         "MSE"])

        return df

    def plot_model_performance(self, stage: Stage, config_spaces: List[ResourceConfig]):
        actual_latencies, predicted_latencies = self._prediction_vs_actual(stage, config_spaces)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(config_spaces))

        ax.plot(x, predicted_latencies, label="Predicted Latencies", marker="x")

        if any(actual_latencies):
            ax.plot(x, actual_latencies, label="Actual Latencies", marker="o")

        ax.set_xlabel("Configurations")
        ax.set_ylabel("Latency")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i.key) for i in config_spaces], rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()

        folder = f"images/{self._dag.dag_id}"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"images/{self._dag.dag_id}/{stage.stage_id}.png")

    def _prediction_vs_actual(self, stage: Stage, config_spaces: List[ResourceConfig]):
        actual_latencies = []
        predicted_latencies = []
        profiling_data = load_profiling_results(f"profiling/{self._dag.dag_id}/{stage.stage_id}.json")
        stage.perf_model.train(profiling_data)
        for config in config_spaces:
            if config.key in profiling_data:
                executions = profiling_data[config.key]
                total_latencies = [
                    sum(lats)
                    for breaks in zip(
                        executions["read"],
                        executions["compute"],
                        executions["write"],
                        executions["cold_start"],
                    )
                    for lats in zip(*breaks)
                ]
                avg_actual_latency = np.mean(total_latencies)
                actual_latencies.append(avg_actual_latency)
            else:
                actual_latencies.append(None)

            predicted_latency = stage.perf_model.predict(config).total
            predicted_latencies.append(predicted_latency)
        return actual_latencies, predicted_latencies

    def optimize(self,
                 config_bounds: ConfigBounds,
                 stage: Optional[Stage] = None) -> List[ResourceConfig]:
        result = []
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage in stages_list:
            result.append(stage.perf_model.optimize(config_bounds))
        return result

    def shutdown(self):
        """
        Shutdown the executor
        """
        self._processor.shutdown()
