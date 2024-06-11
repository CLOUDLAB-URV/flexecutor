from __future__ import annotations

import logging
from lithops import FunctionExecutor
from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.storage import InputS3File, OutputS3File
from flexecutor.utils import setup_logging


logger = setup_logging(level=logging.INFO)

NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"
input_path = f"{BUCKET_NAME}/tiny_shakespeare.txt"
output_path = f"{BUCKET_NAME}/output.txt"
local_base_path = "/tmp"
unique_id = "1"

if __name__ == "__main__":

    @flexorchestrator
    def main():
        dag = DAG("mini-dag")

        def func(
            input_path: str, output_path: str, local_base_path: str, unique_id: str
        ):
            input_file = InputS3File(input_path, local_base_path, unique_id)
            output_file = OutputS3File(output_path, local_base_path, unique_id)
            input_file.download_file()
            # Process the file
            with open(input_file.local_path, "r") as f:
                data = f.read()
                print(data)
            # Simulate writing to output
            with open(output_file.local_path, "w") as f:
                f.write(data)
            output_file.upload_file()

        stage1 = Stage(
            "stage1",
            func=func,
            input_file_path=input_path,
            output_file_path=output_path,
            local_base_path=local_base_path,
            unique_id=unique_id,
            perf_model_type=PerfModelEnum.GENETIC,
        )

        dag.add_stages([stage1])

        executor = DAGExecutor(dag, executor=FunctionExecutor())

        executor.execute()
        executor.shutdown()
        print("Tasks completed")

    main()
