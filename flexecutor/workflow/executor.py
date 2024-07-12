import logging
import os
from enum import Enum
from typing import Dict, Set, List, Iterable, Optional, Callable
from itertools import product

import numpy as np
from lithops import FunctionExecutor
from lithops.utils import get_executor_id

from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
from flexecutor.utils.utils import (
    load_profiling_results,
    store_profiling,
    get_my_exec_path,
)
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.processors import ThreadPoolProcessor
from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture

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
        executor: FunctionExecutor,
    ):
        self._dag = dag
        self._processor = ThreadPoolProcessor()
        self._futures: Dict[str, StageFuture] = dict()
        self._num_final_stages = 0
        self._base_path = get_my_exec_path()
        self._dependence_free_stages: List[Stage] = list()
        self._running_stages: List[Stage] = list()
        self._finished_stages: Set[Stage] = set()
        self._executor = executor
        self._executor_id = get_executor_id()
        # Initialize executor settings for all stages
        self._assign_executor_to_stages()

    def _assign_executor_to_stages(self):
        for stage in self._dag:
            stage.executor = self._executor

    def _get_asset_path(self, stage: Stage, asset_type: AssetType):
        dir_name, file_extension = asset_type.value
        os.makedirs(f"{self._base_path}/{dir_name}/{self._dag.dag_id}", exist_ok=True)
        return f"{self._base_path}/{dir_name}/{self._dag.dag_id}/{stage.stage_id}{file_extension}"

    def _schedule(
        self,
        stage_execute_function: Callable[
            [Stage, Callable[[Stage, StageFuture], None]], None
        ],
        on_future_done: Callable[[Stage, StageFuture], None] = None,
    ) -> Dict[str, StageFuture]:
        """
        Schedule the DAG

        :return: A dictionary with the output data of the DAG stages with the stage ID as key
        """
        logger.info(f"Executing DAG {self._dag.dag_id}")
        self._num_final_stages = len(self._dag.leaf_stages)
        logger.info(f"DAG {self._dag.dag_id} has {self._num_final_stages} final stages")

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
            print(f"Processing batch: {batch}")
            futures = self._processor.process(
                stages=batch,
                stage_execute_function=stage_execute_function,
                on_future_done=on_future_done,
            )

            self._running_stages -= set_batch
            self._dependence_free_stages -= set_batch
            self._finished_stages |= set_batch

            for stage in batch:
                self._futures[stage.stage_id] = futures[stage.stage_id]
                for child in stage.children:
                    if child.parents.issubset(self._finished_stages):
                        self._dependence_free_stages.add(child)

        return self._futures

    def execute(self):
        self._schedule(Stage.execute)

    def profile(
        self,
        # TODO: add a profile id (also on training) to allow having different
        # trained models, mostly for different backends (k8s, lambda, etc.)
        config_space: Iterable[StageConfig],
        num_reps: int = 1,
    ) -> None:

        logger.info(f"Profiling DAG {self._dag.dag_id}")

        # Check that the config_space has the same length as the number of stages in the DAG and they have the same name
        if len(config_space) != len(self._dag.stages):
            raise ValueError(
                "The configuration space must have the same length as the number of stages in the DAG"
            )

        for stage_id, config in zip(self._dag.stages, config_space):
            if stage_id != config.stage_id:
                raise ValueError(
                    "The stage IDs in the configuration space must match the stage ID in the DAG"
                )

        all_config_combinations = list(
            product(*(config_space[stage_id] for stage_id in config_space))
        )

        for iteration in range(num_reps):
            logger.info(f"Starting iteration {iteration + 1} of {num_reps}")

        for config_combination in all_config_combinations:
            config_description = ", ".join(
                f"{stage_id} config: {config}"
                for stage_id, config in zip(config_space, config_combination)
            )
            logger.info(f"Applying configuration combination: {config_description}")

            for stage, config in zip(self._dag.stages, config_combination):
                stage.resource_config = config
                stage.state = StageState.NONE
                logger.info(f"Configured {stage.stage_id} with {config}")

            futures = self.execute()

            for stage, config in zip(self._dag.stages, config_combination):
                future = futures.get(stage.stage_id)
                if future and not future.error():
                    timings = future.get_timings()
                    profiling_file = self._get_asset_path(stage, AssetType.PROFILE)
                    store_profiling(profiling_file, timings, config)
                    logger.info(
                        f"Profiling data for {stage.stage_id} saved in {profiling_file}"
                    )
                elif future and future.error():
                    logger.error(
                        f"Error processing stage {stage.stage_id}: {future.error()}"
                    )

        logger.info("Profiling completed for all configurations")

    def predict(
        self, resource_config: List[StageConfig], stage: Optional[Stage] = None
    ) -> List[FunctionTimes]:
        """Predicts the latency of the entire dag for a given dictionary of resource configurations.
        If a stage is provided, it will predict the latency of that stage only.


        Args:
            resource_config (List[StageConfig]): Dictionary with the resource configuration for each stage.
            stage (Optional[Stage], optional): Stage to profile. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            List[FunctionTimes]: _description_
        """
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

    def optimize(
        self,
        dag_critical_path=List[str],
        config_bounds: ConfigBounds = None,
    ):
        """
        Sets the optimal configuration for each stage.
        """

        print(f"Optimizing DAG {self._dag.dag_id}")

        # range num_workers = 1...10, range_cpu 0.5 to 4, whatever amazon gives you
        # step 1: profile stages with tuples of stages, each stage could be trained on a different set of stageconfigs
        # step 2: train the analytical model with those profiled configurations.
        # step 3: once trained, the optimize function should predict the latency for each stage in the critical path a bruteforce manner for the given config_bounds

        def calculate_memory_for_cpus(cpus: int) -> int:
            memory = cpus * 1769
            return memory

        cpu_combinations = np.arange(config_bounds.cpu[0], config_bounds.cpu[1], 0.5)
        worker_combinations = np.arange(
            config_bounds.workers[0], config_bounds.workers[1], 1
        )
        for stage in self._dag:
            if stage.stage_id in dag_critical_path:
                for cpu in cpu_combinations:
                    for worker in worker_combinations:
                        if stage.max_concurrency == 1:
                            worker = 1
                        memory = calculate_memory_for_cpus(cpu)
                        if memory <= config_bounds.memory[1]:
                            predicted_time = stage.perf_model.predict_time(
                                StageConfig(cpu=cpu, memory=memory, workers=worker)
                            )
                            if predicted_time < stage.perf_model.predict_time(
                                stage.resource_config
                            ):
                                stage.resource_config = StageConfig(
                                    cpu=cpu, memory=memory, workers=worker
                                )

        for stage in self._dag:
            logger.info(
                f"Optimal configuration for stage {stage.stage_id}: {stage.resource_config}"
            )

    def shutdown(self):
        """
        Shutdown the executor
        """
        self._processor.shutdown()
