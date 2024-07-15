import logging
import os
from enum import Enum
from typing import Dict, Set, List, Iterable, Optional

import numpy as np
import pandas as pd
from lithops import FunctionExecutor
from matplotlib import pyplot as plt
from pandas import DataFrame
from copy import deepcopy

from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
from flexecutor.utils.utils import (
    load_profiling_results,
    save_profiling_results,
    get_my_exec_path,
)
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.processors import ThreadPoolProcessor
from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.optimization import BruteforceSolver

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """
    Enum class for asset types
    """

    MODEL = ("model", ".pkl")
    PROFILE = ("profile", ".json")
    IMAGE = ("image", ".png")


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
        self._base_path = get_my_exec_path()
        self._dependence_free_stages: List[Stage] = list()
        self._running_stages: List[Stage] = list()
        self._finished_stages: Set[Stage] = set()
        self._executor = executor

    def _get_asset_path(self, stage: Stage, asset_type: AssetType):
        # previous folder creation
        # TODO: remove ifelse and use AssetType tuples (dir, extension)
        if asset_type == AssetType.MODEL:
            os.makedirs(f"{self._base_path}/models/{self._dag.dag_id}", exist_ok=True)
            return f"{self._base_path}/models/{self._dag.dag_id}/{stage.stage_id}.pkl"
        elif asset_type == AssetType.PROFILE:
            os.makedirs(
                f"{self._base_path}/profiling/{self._dag.dag_id}", exist_ok=True
            )
            return (
                f"{self._base_path}/profiling/{self._dag.dag_id}/{stage.stage_id}.json"
            )
        elif asset_type == AssetType.IMAGE:
            os.makedirs(f"{self._base_path}/images/{self._dag.dag_id}", exist_ok=True)
            return f"{self._base_path}/images/{self._dag.dag_id}/{stage.stage_id}.png"

    def _store_profiling(
        self,
        file: str,
        new_profile_data: List[List[FunctionTimes]],
        resource_config: StageConfig,
    ):
        profile_data = load_profiling_results(file)
        config_key = str(resource_config.key)

        if config_key not in profile_data:
            profile_data[config_key] = {key: [] for key in FunctionTimes.profile_keys()}

        for batch_timings in new_profile_data:
            batch_data = {
                key: [getattr(timing, key) for timing in batch_timings]
                for key in FunctionTimes.profile_keys()
            }
            for key in batch_data:
                profile_data[config_key][key].append(batch_data[key])

        save_profiling_results(file, profile_data)

    def profile(
        self,
        # TODO: add a profile id (also on training) to allow having different
        # trained models, mostly for different backends (k8s, lambda, etc.)
        config_space: Iterable[StageConfig],
        num_reps: int = 1,
    ) -> None:
        # TODO: configuration space is a list of lists of StageConfig. The first
        # list are the configurations for the dag. Each list (config) within
        # contains a StageConfig for each stage in the dag. Check that each
        # config has the same length as the number of stages in the dag,
        # otherwise skip the config with a warning.

        logger.info(f"Profiling DAG {self._dag.dag_id}")

        for config in config_space:
            logger.info(f"Testing configuration: {config}")

            for iteration in range(num_reps):
                logger.info(f"Starting iteration {iteration + 1} of {num_reps}")

                # Create a fresh copy of the DAG for each iteration
                iteration_stages = {
                    stage.stage_id: deepcopy(stage) for stage in self._dag.stages
                }

                for stage in iteration_stages.values():
                    stage.resource_config = config
                    stage.state = StageState.NONE

                self._futures = {}
                self._dependence_free_stages = set(
                    iteration_stages[stage.stage_id] for stage in self._dag.root_stages
                )
                self._running_stages = set()
                self._finished_stages = set()

                while self._dependence_free_stages or self._running_stages:
                    batch = list(self._dependence_free_stages)
                    self._running_stages.update(batch)
                    futures = self._processor.process(batch)

                    for stage in batch:
                        future = futures[stage.stage_id]
                        self._futures[stage.stage_id] = future
                        if future.error():
                            logger.error(f"Error processing stage {stage.stage_id}")
                        else:
                            timings = future.get_timings()
                            profiling_file = self._get_asset_path(
                                stage, AssetType.PROFILE
                            )
                            self._store_profiling(profiling_file, [timings], config)

                    self._running_stages.difference_update(batch)
                    self._dependence_free_stages.difference_update(batch)
                    self._finished_stages.update(stage.stage_id for stage in batch)

                    for stage in batch:
                        for child in stage.children:
                            if all(
                                parent.stage_id in self._finished_stages
                                for parent in child.parents
                            ):
                                self._dependence_free_stages.add(
                                    iteration_stages[child.stage_id]
                                )

                logger.info(f"Iteration {iteration + 1} for config {config} completed")
            logger.info(f"Profiling completed for config {config}")

        logger.info("Profiling completed for all configurations")

    def predict(
        self, resource_config: List[StageConfig], stage: Optional[Stage] = None
    ) -> List[FunctionTimes]:
        # TODO: predict latency/cost of the full dag. Return an object with the
        # breakdown of latencies per stage.

        # FIXME: (?) predict makes sense to move as method of DAG/Stage since models
        # are stored there. Train too?
        # Keep this method as a convenient wrapper for self._dag.predict()

        # FIXME: (?) resource_config as a list or as a dict by stage_id?
        # assert it contains config for all stages

        if stage is not None and len(resource_config) > 1:
            raise ValueError(
                "predict() requires single Stage when only one StageConfig is provided and vice versa."
            )
        elif stage is None and len(resource_config) != len(self._dag.stages):
            raise ValueError(
                "predict() requires a list of StageConfig equal to the number of Stage in the DAG."
            )
        result = []
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage, resource_config in zip(stages_list, resource_config):
            result.append(stage.perf_model.predict(resource_config))
        return result

    def train(self, stage: Optional[Stage] = None) -> None:
        """Train the DAG."""
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage in stages_list:
            profile_data = load_profiling_results(
                self._get_asset_path(stage, AssetType.PROFILE)
            )
            stage.perf_model.train(profile_data)
            stage.perf_model.save_model()

    def execute(self) -> Dict[str, StageFuture]:
        """
        Execute the DAG in an Lazy manner

        :return: A dictionary with the output data of the DAG stages with the stage ID as key
        """
        logger.info(f"Executing DAG {self._dag.dag_id}")

        self._num_final_stages = len(self._dag.leaf_stages)
        logger.info(f"DAG {self._dag.dag_id} has {self._num_final_stages} final stages")

        # Before the execution, get the optimal configurations for all stages in the DAG
        # FIXME: actually optimize, hardcoded for now
        # self.optimize(ConfigBounds(*[(1, 6), (512, 4096), (1, 3)]))
        for stage in self._dag.stages:
            stage.resource_config = StageConfig(cpu=5, memory=722, workers=2)

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

    def model_perf_metrics(
        self, stage: Stage, config_space: List[StageConfig]
    ) -> DataFrame:
        actual_latencies, predicted_latencies = self._prediction_vs_actual(
            stage, config_space
        )

        actual_latencies = np.array(actual_latencies)
        predicted_latencies = np.array(predicted_latencies)

        data = np.array(
            [
                [
                    config.workers,
                    config.cpu,
                    config.memory,
                    actual,
                    predicted,
                    abs(actual - predicted),
                    (actual - predicted) ** 2,
                ]
                for config, actual, predicted in zip(
                    config_space, actual_latencies, predicted_latencies
                )
            ]
        )

        df = pd.DataFrame(
            data,
            columns=[
                "Workers",
                "CPU",
                "Memory",
                "Actual latency",
                "Predicted latency",
                "MAE",
                "MSE",
            ],
        )

        return df

    def plot_model_performance(self, stage: Stage, config_space: List[StageConfig]):
        actual_latencies, predicted_latencies = self._prediction_vs_actual(
            stage, config_space
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(config_space))

        ax.plot(x, predicted_latencies, label="Predicted Latencies", marker="x")

        if any(actual_latencies):
            ax.plot(x, actual_latencies, label="Actual Latencies", marker="o")

        ax.set_xlabel("Configurations")
        ax.set_ylabel("Latency")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i.key) for i in config_space], rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()

        plt.savefig(self._get_asset_path(stage, AssetType.IMAGE))

    def _prediction_vs_actual(self, stage: Stage, config_space: List[StageConfig]):
        actual_latencies = []
        predicted_latencies = []
        profiling_data = load_profiling_results(
            f"{self._base_path}/profiling/{self._dag.dag_id}/{stage.stage_id}.json"
        )
        stage.perf_model.train(profiling_data)
        for config in config_space:
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

    def optimize(
        self,
        dag_critical_path=List[str],
        config_bounds: ConfigBounds = None,
    ):
        """
        Sets the optimal configuration for each stage.
        """

        print(f"Optimizing DAG {self._dag.dag_id}")

        solver = BruteforceSolver("bruteforce")
        solver.solve(self._dag, dag_critical_path, config_bounds)

    def shutdown(self):
        """
        Shutdown the executor
        """
        self._processor.shutdown()
