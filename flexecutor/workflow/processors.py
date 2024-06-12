from __future__ import annotations
import os
import logging

from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Sequence, List, Tuple

from lithops import FunctionExecutor

from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.utils import setup_logging
from flexecutor.storage import DataSlice, S3Handler

logger = setup_logging(level=logging.INFO)


def split_txt_file(
    file_path: str,
    chunk_size: int = None,
    chunk_number: int = None,
    obj_newline: str = "\n",
) -> List[Tuple[int, int]]:
    """
    Split a single .txt file into multiple chunks.
    """
    partitions = []
    obj_size = os.path.getsize(file_path)

    if chunk_number:
        chunk_rest = obj_size % chunk_number
        obj_chunk_size = (obj_size // chunk_number) + round(
            (chunk_rest / chunk_number) + 0.5
        )
    elif chunk_size:
        obj_chunk_size = chunk_size
    else:
        obj_chunk_size = obj_size

    logger.debug(f"Creating partitions from {file_path} ({obj_size} bytes)")

    with open(file_path, "rb") as f:
        size = 0
        while size < obj_size:
            start = size
            end = min(size + obj_chunk_size, obj_size)
            if end < obj_size:
                f.seek(end)
                while f.read(1) != obj_newline.encode() and f.tell() < obj_size:
                    end += 1

            partitions.append((start, end))
            size = end + 1

    return partitions


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
        # TODO:
        # 1. Do a predict call to the model to get the optimal number of workers, memory and vcpus (DONE, HARDCODED)
        # 2. Update the configuration of the executor (FIXME ISSUE WITH RUNTIME_NUMCPUS, FOR NOW WE CAN ONLY PASS THE RUNTIME_MEMORY)
        # 3.  PARTITIONING : Partition the file into multiple equal-sized chunks
        #   3.1 partitioning happens before the map call, this is the preprocessing needed for converting InputS3Paths to chunks as well as converting the InputS3Path to its corresponding OutputS3Path (iterdata)
        # 4. The lithops map function should be wrapped with a download-upload logic.
        # 5. Call the map which will return the iterdata, pass the iterdata to the map

        print(f"Found datasets: {stage.input_file}")

        input_file = stage.input_file
        print(f"stage input file: {input_file}")
        input_file.download_file()

        # Partition after the dataset is loaded
        # TODO: Partition here, for now we are just splitting the file into equal chunks and we only support .txt files
        partitions = split_txt_file(
            input_file.local_path, chunk_number=stage.optimal_config.workers
        )

        print("Partitions: ", partitions)

        s3_handler = S3Handler()

        map_iterdata = [
            DataSlice(
                bucket=input_file.bucket,
                key=input_file.key,
                output_bucket=input_file.bucket,  # Assuming output in the same bucket
                output_key=stage.output_path.generate_output_key(input_file.key, chunk),
                local_base_path=input_file.local_base_path,
                unique_id=input_file.unique_id,
                chunk=chunk,
                s3_handler=s3_handler,
            )
            for chunk in partitions
        ]

        future = self._executor.map(
            map_function=stage.map_func,
            map_iterdata=map_iterdata,
            runtime_memory=stage.optimal_config.memory,
        )

        self._executor.wait(future)
        future = StageFuture(stage.stage_id, future)

        # Update the state of the stage based on the future result
        stage.state = StageState.FAILED if future.error() else StageState.SUCCESS

        # Call the callback function if provided
        if on_future_done:
            on_future_done(stage, future)

        return future
