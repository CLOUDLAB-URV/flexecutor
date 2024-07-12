from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Callable, Sequence
import logging

from lithops import FunctionExecutor

from flexecutor.storage.wrapper import worker_wrapper
from flexecutor.workflow.stagecontext import InternalStageContext
from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.utils import setup_logging

logger = setup_logging(level=logging.INFO)


class ThreadPoolProcessor:
    """
    Processor that uses a thread pool to execute stages
    """

    def __init__(self, max_threadpool_concurrency=256):
        self._max_concurrency = max_threadpool_concurrency
        self._pool = ThreadPoolExecutor(max_workers=max_threadpool_concurrency)

    def process(
        self,
        stages: Sequence[Stage],
        execute_function: Callable[[Stage, Callable[[Stage, StageFuture], None]], None],
        on_future_done: Callable[[Stage, StageFuture], None] = None,
    ) -> dict[str, StageFuture]:
        """
        Process a list of stages using a specific function for execution.
        :param stages: List of stages to process
        :param execute_function: The function to be executed for each stage
        :param on_future_done: Callback to execute every time a future is done
        :return: Futures of the stages
        :raises ValueError: If there are no stages to process or if there are
        more stages than the maximum parallelism
        """
        if not stages:
            raise ValueError("No stages to process")

        if len(stages) > self._max_concurrency:
            # TODO: don't fail. queue them
            raise ValueError(
                f"Too many stages to process. Max concurrency is {self._max_concurrency}"
            )

        futures = {}

        for stage in stages:
            logger.info(f"Submitting stage {stage.stage_id}")
            stage.state = StageState.RUNNING
            futures[stage.stage_id] = self._pool.submit(
                lambda s=stage: execute_function(s, on_future_done)
            )

        wait(futures.values())

        return {stage_id: future.result() for stage_id, future in futures.items()}

    def shutdown(self):
        """
        Shuts down the thread pool executor.
        """
        self._pool.shutdown()
