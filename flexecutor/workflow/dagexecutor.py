import logging
import math
import os
from typing import Dict, Set, List, Iterable

from lithops import FunctionExecutor

from flexecutor.utils.dataclass import FunctionTimes, ResourceConfig
from flexecutor.utils.utils import load_profiling_results, save_profiling_results
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executors import Executor, CallableExecutor
from flexecutor.workflow.processors import Processor, ThreadPoolProcessor
from flexecutor.workflow.task import Task
from flexecutor.workflow.taskfuture import TaskFuture

logger = logging.getLogger(__name__)


class DAGExecutor:
    """
    Executor class that is responsible for executing the DAG

    :param dag: DAG to execute
    :param processor: Processor to use for executing tasks, defaults to DefaultProcessor
    """

    def __init__(
            self,
            dag: DAG,
            processor: Processor = None,
            executor: Executor = CallableExecutor(),
            task_executor: FunctionExecutor | None = None,
    ):
        self._dag = dag
        self._processor = processor or ThreadPoolProcessor(math.inf)
        self._executor = executor

        self._futures: Dict[str, TaskFuture] = dict()
        self._num_final_tasks = 0
        self._dependence_free_tasks: List[Task] = list()
        self._running_tasks: List[Task] = list()
        self._finished_tasks: Set[Task] = set()
        self._task_executor = task_executor

        if task_executor is not None:
            for task in self._dag.tasks:
                task.executor = task_executor

    def _get_timings(self, futures: dict[str, TaskFuture]) -> List[FunctionTimes]:
        """Get the timings of the futures."""
        timings_list = []
        for task_id, future in futures.items():
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
                    f'Error getting timings for task {task_id}. Please review the return values of map function')
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
        for task in self._dag.tasks:
            os.makedirs(f"profiling/{self._dag.dag_id}", exist_ok=True)
            profiling_file = f"profiling/{self._dag.dag_id}/{task.task_id}.json"
            for config_space in config_spaces:
                for iteration in range(num_iterations):
                    timings = self.run_task(task, config_space)
                    self._store_profiling(profiling_file, timings, config_space)

    def run_task(self, task: Task, config_space: ResourceConfig) -> List[FunctionTimes]:
        """Run a task with a given configuration space."""
        # Set the parameters in Lithops config
        self._task_executor.config['runtime_cpu'] = config_space.cpu
        self._task_executor.config['runtime_memory'] = config_space.memory
        self._task_executor.config['workers'] = config_space.workers
        logger.info(f'Running task {task.task_id} with config {config_space}')
        # Execute the task
        futures = self._processor.process([task], self._executor)
        # Store the profiling data
        timings = self._get_timings(futures)
        return timings

    def train(self) -> None:
        """Train the DAG."""
        for task in self._dag.tasks:
            profile_data = load_profiling_results(f"profiling/{self._dag.dag_id}/{task.task_id}.json")
            task.perf_model.train(profile_data)
            task.perf_model.save_model()

    def execute(self) -> Dict[str, TaskFuture]:
        """
        Execute the DAG

        :return: A dictionary with the output data of the DAG tasks with the task ID as key
        """
        logger.info(f'Executing DAG {self._dag.dag_id}')

        self._num_final_tasks = len(self._dag.leaf_tasks)
        logger.info(f'DAG {self._dag.dag_id} has {self._num_final_tasks} final tasks')

        self._futures = dict()

        # Start by executing the root tasks
        self._dependence_free_tasks = set(self._dag.root_tasks)
        self._running_tasks = set()
        self._finished_tasks = set()

        # Execute tasks until all tasks have been executed
        while self._dependence_free_tasks or self._running_tasks:
            # Select the tasks to execute
            batch = list(self._dependence_free_tasks)

            # Add the batch to the running tasks
            set_batch = set(batch)
            self._running_tasks |= set_batch

            # Call the processor to execute the batch
            futures = self._processor.process(batch, self._executor)

            self._running_tasks -= set_batch
            self._dependence_free_tasks -= set_batch
            self._finished_tasks |= set_batch

            for task in batch:
                self._futures[task.task_id] = futures[task.task_id]
                for child in task.children:
                    if child.parents.issubset(self._finished_tasks):
                        self._dependence_free_tasks.add(child)

        return self._futures

    def shutdown(self):
        """
        Shutdown the executor
        """
        self._processor.shutdown()
