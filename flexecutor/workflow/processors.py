from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Sequence

from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture

logger = logging.getLogger(__name__)


class ThreadPoolProcessor:
    """
    Processor that uses a thread pool to execute stages
    """

    def __init__(self, max_concurrency=256):
        super().__init__()
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
            raise ValueError('No stages to process')

        if len(stages) > self._max_concurrency:
            raise ValueError(f'Too many stages to process. Max concurrency is {self._max_concurrency}')

        ex_futures = {}

        for stage in stages:
            logger.info(f"Submitting stage {stage.stage_id}")

            # TODO: get optimal stage configuration
            # TODO: set optimal config to lithops executor

            stage.state = StageState.RUNNING
            ex_futures[stage.stage_id] = self._pool.submit(
                lambda: _process_stage(
                    stage,
                    on_future_done
                )
            )

        wait(ex_futures.values())

        return {stage_id: ex_future.result() for stage_id, ex_future in ex_futures.items()}

    def shutdown(self):
        self._pool.shutdown()


def _process_stage(
        stage: Stage,
        on_future_done: Callable[[Stage, StageFuture], None] = None,
        *args,
        **kwargs
) -> StageFuture:
    """
    Process a stage

    :param stage: stage to process
    :param on_future_done: Callback to execute every time a future is done
    """
    future = stage(*args, **kwargs)
    stage.executor.wait(future)
    future = StageFuture(future)

    stage.state = StageState.FAILED if future.error() else StageState.SUCCESS

    if on_future_done:
        on_future_done(stage, future)

    return future
