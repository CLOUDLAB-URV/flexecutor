import logging
import os
from typing import Dict, Set, List, Iterable

from lithops import FunctionExecutor

from flexecutor.utils.dataclass import FunctionTimes, ResourceConfig
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
    :param stage_executor: Executor to use for the stages
    """

    def __init__(
            self,
            dag: DAG,
            stage_executor: FunctionExecutor | None = None,
    ):
        self._dag = dag
        self._processor = ThreadPoolProcessor()

        self._futures: Dict[str, StageFuture] = dict()
        self._num_final_stages = 0
        self._dependence_free_stages: List[Stage] = list()
        self._running_stages: List[Stage] = list()
        self._finished_stages: Set[Stage] = set()
        self._stage_executor = stage_executor

        if stage_executor is not None:
            for stage in self._dag.stages:
                stage.executor = stage_executor

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

    def profile(self, config_spaces: Iterable[ResourceConfig], num_iterations: int = 1):
        """Profile the DAG."""
        for stage in self._dag.stages:
            os.makedirs(f"profiling/{self._dag.dag_id}", exist_ok=True)
            profiling_file = f"profiling/{self._dag.dag_id}/{stage.stage_id}.json"
            for config_space in config_spaces:
                for iteration in range(num_iterations):
                    timings = self.run_stage(stage, config_space)
                    self._store_profiling(profiling_file, timings, config_space)

    def run_stage(self, stage: Stage, config_space: ResourceConfig) -> List[FunctionTimes]:
        """Run a stage with a given configuration space."""
        # Set the parameters in Lithops config
        self._stage_executor.config['runtime_cpu'] = config_space.cpu
        self._stage_executor.config['runtime_memory'] = config_space.memory
        self._stage_executor.config['workers'] = config_space.workers
        logger.info(f'Running stage {stage.stage_id} with config {config_space}')
        # Execute the stage
        futures = self._processor.process([stage])
        # Store the profiling data
        timings = self._get_timings(futures)
        return timings

    def train(self) -> None:
        """Train the DAG."""
        for stage in self._dag.stages:
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

    def shutdown(self):
        """
        Shutdown the executor
        """
        self._processor.shutdown()
