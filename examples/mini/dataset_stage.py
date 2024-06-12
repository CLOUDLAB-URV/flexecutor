from __future__ import annotations

import logging

from functools import wraps
from lithops import FunctionExecutor, Storage
from typing import Callable, Tuple, Any

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.utils import setup_logging
from flexecutor.storage import InputS3File, OutputS3File, InputS3Chunk
from functools import wraps


logger = setup_logging(level=logging.INFO)


def manage_s3_io(func: Callable[[InputS3Chunk], Any]):
    @wraps(func)
    def wrapper(chunk: InputS3Chunk, *args, **kwargs):
        client = Storage()

        # Download the input chunk
        byte_range = chunk.chunk
        extra_get_args = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
        chunk_data = client.get_object(
            chunk.bucket, chunk.key, extra_get_args=extra_get_args
        )

        chunk.local_input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk.local_input_path, "wb") as f:
            f.write(chunk_data)

        result = func(chunk, *args, **kwargs)

        return result

    return wrapper


NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"
input_file = InputS3File(f"{BUCKET_NAME}/tiny_shakespeare.txt", "/tmp", "1")
output_file = OutputS3File(f"{BUCKET_NAME}/output.txt", "/tmp", "1")
if __name__ == "__main__":

    @flexorchestrator
    def main():
        dag = DAG("mini-dag")

        # function to be applied to each chunk, the input of the function has to always be of type InputS3Chunk
        @manage_s3_io
        def func(chunk: InputS3Chunk):
            print(chunk)

        stage1 = Stage(
            "stage1",
            func=func,
            perf_model_type=PerfModelEnum.GENETIC,
            input_file=input_file,
            output_file=output_file,
        )

        dag.add_stages([stage1])

        executor = DAGExecutor(dag, executor=FunctionExecutor())

        executor.execute()
        executor.shutdown()
        print("Tasks completed")

    main()
