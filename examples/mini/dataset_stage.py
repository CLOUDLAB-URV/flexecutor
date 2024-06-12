from __future__ import annotations

import logging

from lithops import FunctionExecutor
from typing import Callable, Tuple, Any

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.utils import setup_logging
from flexecutor.storage import InputS3File, OutputS3Path, DataSlice
from functools import wraps


logger = setup_logging(level=logging.INFO)


# This decorator does pre and post processing of the chunk in the remote worker
def manage_s3_io(func: Callable[[DataSlice], Any]):
    @wraps(func)
    def wrapper(data_slice: DataSlice, *args, **kwargs):
        data_slice.s3_handler.download_chunk(data_slice)

        result = func(data_slice, *args, **kwargs)

        # data_slice.s3_handler.upload_chunk(data_slice)

        return result

    return wrapper


NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"
input_file = InputS3File(f"{BUCKET_NAME}/tiny_shakespeare.txt", "/tmp", "1")
output_path = OutputS3Path(f"{BUCKET_NAME}/output", "/tmp", "1")

if __name__ == "__main__":

    @flexorchestrator
    def main():
        dag = DAG("mini-dag")

        # function to be applied to each chunk, the input of the function has to always be of type DataSlice
        @manage_s3_io
        def word_count(data_slice: DataSlice):
            # word counting logic
            with open(data_slice.local_input_path, "r") as f:
                words = f.read().split()
                word_count = len(words)
                print(f"Word count: {word_count}")

            # write the result to the output file
            # with open(data_slice.local_output_path, "w") as f:
            #     f.write(str(word_count))

        stage1 = Stage(
            "stage1",
            func=word_count,
            perf_model_type=PerfModelEnum.GENETIC,
            input_file=input_file,
            output_path=output_path,
        )

        dag.add_stages([stage1])

        executor = DAGExecutor(dag, executor=FunctionExecutor())

        executor.execute()
        executor.shutdown()
        print("Tasks completed")

    main()
