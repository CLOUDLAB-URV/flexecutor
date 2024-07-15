import logging
from typing import Dict, Set, List, Iterable, Optional, Callable
from itertools import product

import numpy as np
from lithops import FunctionExecutor
from lithops.utils import get_executor_id

from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
from flexecutor.utils.file_paths import (
    load_profiling_results,
    AssetType,
)
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.processors import ThreadPoolProcessor
from flexecutor.workflow.stage import Stage
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.optimization import BruteforceSolver

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
        executor: FunctionExecutor,
    ):
        self._dag = dag
        self._processor = ThreadPoolProcessor()
        self._futures: Dict[str, StageFuture] = dict()
        self._num_final_stages = 0
        self._dependence_free_stages: List[Stage] = list()
        self._running_stages: List[Stage] = list()
        self._finished_stages: Set[Stage] = set()
        self._executor = executor
        self._executor_id = get_executor_id()
        # Initialize executor settings for all stages
        for stage in self._dag:
            stage.executor = self._executor

    def _execute_dag_stages(
        self,
        stage_execute_function: Callable[
            [Stage, Callable[[Stage, StageFuture], None]], None
        ],
        on_future_done: Callable[[Stage, StageFuture], None] = None,
        stage_configs: Optional[Dict[str, StageConfig]] = None,
        *args,
        **kwargs,
    ) -> Dict[str, StageFuture]:
        """
        Schedule the DAG stages for execution

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
                stage_configs=stage_configs,
                *args,
                **kwargs,
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
        futures = self._execute_dag_stages(Stage.execute, stage_configs=None)
        return futures

    # TODO: add a profile id (also on training) to allow having different
    # trained models, mostly for different backends (k8s, lambda, etc.)
    def profile(
        self, config_space: Dict[str, Iterable[StageConfig]], num_reps: int = 1
    ):
        stage_ids = list(config_space.keys())
        all_config_combinations = list(product(*config_space.values()))

        for iteration in range(num_reps):
            logger.info(f"Starting iteration {iteration + 1} of {num_reps}")

            for config_combination in all_config_combinations:
                config_dict = dict(zip(stage_ids, config_combination))
                config_description = ", ".join(
                    f"{stage_id} config: {config}"
                    for stage_id, config in config_dict.items()
                )
                logger.info(f"Applying configuration combination: {config_description}")

                def on_future_done(stage: Stage, future: StageFuture):
                    logger.info(f"Profiling completed for stage {stage.stage_id}")
                    logger.info(f"Execution time: {future.get_timings()}")

                futures = self._execute_dag_stages(
                    stage_execute_function=Stage.profile,
                    on_future_done=on_future_done,
                    stage_configs=config_dict,
                )

                for stage_id, future in futures.items():
                    if stage_id in config_dict:
                        stage_config = config_dict[stage_id]
                        logger.info(
                            f"Profiling results for stage {stage_id} with config: {stage_config}"
                        )
                    else:
                        logger.info(
                            f"No profiling data for stage {stage_id} (no config provided)"
                        )

        logger.info("Profiling completed for all configurations and repetitions")

    def train(self, stage: Optional[Stage] = None) -> None:
        """Train the DAG."""
        from flexecutor.utils.file_paths import get_asset_path, load_profiling_results

        stages_list = [stage] if stage is not None else self._dag.stages

        for stage in stages_list:
            profiling_file = get_asset_path(stage, AssetType.PROFILE)
            profiling_data = load_profiling_results(profiling_file)
            stage.train(profiling_data)

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
            result.append(stage.predict(resource_config))
        return result

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
