from __future__ import annotations

import logging
from lithops import FunctionExecutor
from typing import Callable, Tuple, Any
from functools import wraps

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.utils import setup_logging
from flexecutor.storage import InputS3File, OutputS3Path, DataSlice

logger = setup_logging(level=logging.INFO)


# This decorator does pre and post processing of the chunk in the remote worker.
def chunkprocessor(func: Callable[[DataSlice], Any]):
    @wraps(func)
    def wrapper(data_slice: DataSlice, *args, **kwargs):
        print(
            f"Downloading chunk: {data_slice.chunk} from bucket: {data_slice.bucket}, key: {data_slice.key}"
        )
        data_slice.s3_handler.download_chunk(data_slice)

        result = func(data_slice, *args, **kwargs)

        print(
            f"Uploading chunk: {data_slice.chunk} to bucket: {data_slice.output_bucket}, key: {data_slice.output_key}"
        )
        data_slice.s3_handler.upload_chunk(data_slice, data_slice.unique_output_path)

        return result

    return wrapper


NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"
input_file = InputS3File(f"{BUCKET_NAME}/tiny_shakespeare.txt", "/tmp", "1")
output_path = OutputS3Path(f"{BUCKET_NAME}/output", "/tmp", "1")


@flexorchestrator
def main():
    dag = DAG("mini-dag")

    @chunkprocessor
    def word_count(data_slice: DataSlice):
        print(f"Processing DataSlice: {data_slice}")

        data_slice.local_output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Reading from local input path: {data_slice.local_input_path}")
        with open(data_slice.local_input_path, "r") as f:
            content = f.read()
        word_count = len(content.split())

        unique_output_path = data_slice.local_output_path.with_name(
            f"{data_slice.local_output_path.stem}_{data_slice.chunk[0]}_{data_slice.chunk[1]}{data_slice.local_output_path.suffix}"
        )
        with open(unique_output_path, "w") as f:
            f.write(str(word_count))

        data_slice.unique_output_path = unique_output_path

        print(f"Finished processing DataSlice: {data_slice}")

    stage1 = Stage(
        "stage1",
        func=word_count,
        perf_model_type=PerfModelEnum.GENETIC,
        input_file=input_file,
        output_path=output_path,
    )

    dag.add_stages([stage1])

    executor = DAGExecutor(dag, executor=FunctionExecutor())

    logger.info("Starting DAG execution")
    executor.execute()
    executor.shutdown()
    logger.info("Tasks completed")


if __name__ == "__main__":
    main()
