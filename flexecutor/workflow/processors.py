from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Sequence

from lithops import FunctionExecutor

from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture

logger = logging.getLogger(__name__)


class ThreadPoolProcessor:
    """
    Processor that uses a thread pool to execute stages
    """

    def __init__(self, executor: FunctionExecutor, max_concurrency=256):
        super().__init__()
        self._executor = executor
        self._max_concurrency = max_concurrency
        self._pool = ThreadPoolExecutor(max_workers=max_concurrency)

    def process(
        self,
        stages: Sequence[Stage],
        on_future_done: Callable[[Stage, StageFuture], None] = None,
    ) -> dict[str, StageFuture]:
        """
        Process a list of stages
        :param stages: List of stages to process
        :param on_future_done: Callback to execute every time a future is done
        :return: Futures of the stages
        :raises ValueError: If there are no stages to process or if there are more stages than the maximum parallelism
        """
        if len(stages) == 0:
            raise ValueError("No stages to process")

        if len(stages) > self._max_concurrency:
            raise ValueError(
                f"Too many stages to process. Max concurrency is {self._max_concurrency}"
            )

        ex_futures = {}

        for stage in stages:
            logger.info(f"Submitting stage {stage.stage_id}")

            stage.state = StageState.RUNNING
            ex_futures[stage.stage_id] = self._pool.submit(
                lambda: self._process_stage(stage, on_future_done)
            )

        wait(ex_futures.values())

        return {
            stage_id: ex_future.result() for stage_id, ex_future in ex_futures.items()
        }

    def shutdown(self):
        self._pool.shutdown()

    def _process_stage(
        self, stage: Stage, on_future_done: Callable[[Stage, StageFuture], None] = None
    ) -> StageFuture:
        """
        Process a stage

        :param stage: stage to process
        :param on_future_done: Callback to execute every time a future is done
        """
        # Update configuration of resources
        # FIXME: review that, this way not work IMO
        # self._executor.config["workers"] = stage.resource_config.workers
        # self._executor.config["runtime_memory"] = stage.resource_config.memory
        # self._executor.config["runtime_cpu"] = stage.resource_config.cpu

        # TODO:
        # 1. Do a predict call to the model to get the optimal number of workers, memory and vcpus (DONE, HARDCODED)
        # 2. Update the configuration of the executor (FIXME ISSUE WITH RUNTIME_NUMCPUS, FOR NOW WE CAN ONLY PASS THE TOTAL MEMORY)
        # 3. Call the partitioner and pass the resulting iterdata of the partitioner to the function executor

        # kwargs = {"obj_chunk_number": stage.resource_config.workers}

        print(f"Found datasets: {stage.input_dataset.paths}")

        # Partition after the dataset is found?

        future = self._executor.map(
            map_function=stage.map_func,
            map_iterdata=stage.input_dataset.paths,
            runtime_memory=stage.optimal_config.memory,  # **kwargs
        )

        self._executor.wait(future)
        future = StageFuture(stage.stage_id, future)

        stage.state = StageState.FAILED if future.error() else StageState.SUCCESS

        if on_future_done:
            on_future_done(stage, future)

        return future
