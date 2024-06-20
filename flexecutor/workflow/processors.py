from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Callable, Sequence, List, Tuple
import os
import logging

from lithops import FunctionExecutor

from flexecutor.storage.wrapper import worker_wrapper
from flexecutor.utils.iomanager import IOManager
from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.utils import setup_logging
# from flexecutor.storage.storage import DataSlice, S3Handler, InputS3Path

logger = setup_logging(level=logging.INFO)


# def split_txt_files(
#     file_paths: List[str],
#     chunk_size: int = None,
#     chunk_number: int = None,
#     obj_newline: str = "\n",
# ) -> List[Tuple[str, int, int]]:
#     """
#     Split multiple .txt files into multiple chunks.
#     """
#     partitions = []
#     total_size = sum(os.path.getsize(file_path) for file_path in file_paths)
#
#     if chunk_number:
#         chunk_rest = total_size % chunk_number
#         obj_chunk_size = (total_size // chunk_number) + (1 if chunk_rest else 0)
#     elif chunk_size:
#         obj_chunk_size = chunk_size
#     else:
#         obj_chunk_size = total_size
#
#     logger.debug(
#         f"Creating partitions from {len(file_paths)} files ({total_size} bytes)"
#     )
#
#     current_size = 0
#     current_file_index = 0
#
#     while current_size < total_size:
#         start_file = file_paths[current_file_index]
#         current_file_size = os.path.getsize(start_file)
#         start_position = current_size % current_file_size
#         end_position = min(start_position + obj_chunk_size, current_file_size)
#
#         if end_position < current_file_size:
#             partitions.append((start_file, start_position, end_position))
#             current_size += end_position - start_position
#         else:
#             partitions.append((start_file, start_position, current_file_size))
#             current_size += current_file_size - start_position
#             if current_file_index + 1 < len(file_paths):
#                 current_file_index += 1
#
#     return partitions


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

        # for input_path in stage.input_file:
        #     if input_path.partitioner:
        #         input_path.partitioner.partitionize()

        # s3_handler = S3Handler()

        map_iterdata = []
        num_workers = stage.optimal_config.workers
        for worker_id in range(num_workers):
            copy_inputs = [deepcopy(item) for item in stage.inputs]
            for input_item in copy_inputs:
                input_item.set_chunk_indexes(worker_id, num_workers)
            io = IOManager(worker_id, num_workers, copy_inputs, stage.outputs, stage.params)
            map_iterdata.append(io)

        future = self._executor.map(
            map_function=worker_wrapper(stage.map_func),
            map_iterdata=map_iterdata,
            runtime_memory=int(stage.optimal_config.memory),
        )

        self._executor.wait(future)
        future = StageFuture(stage.stage_id, future)

        # Update the state of the stage based on the future result
        stage.state = StageState.FAILED if future.error() else StageState.SUCCESS

        # Call the callback function if provided
        if on_future_done:
            on_future_done(stage, future)

        return future
