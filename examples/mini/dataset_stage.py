from __future__ import annotations

import logging

from lithops import FunctionExecutor

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.storage import Dataset
from flexecutor.utils import setup_logging
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}


logger = setup_logging(level=logging.INFO)

NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"


if __name__ == "__main__":

    @flexorchestrator
    def main():
        dag = DAG("mini-dag")

        # function to be applied to each chunk
        def func(chunk):
            print(chunk)

        stage1 = Stage(
            "stage1",
            func=func,
            perf_model_type=PerfModelEnum.GENETIC,
            input_dataset=Dataset.from_glob(BUCKET_NAME, "dir/*.txt"),
        )

        dag.add_stages([stage1])

        executor = DAGExecutor(dag, executor=FunctionExecutor())

        executor.execute()
        executor.shutdown()
        print("Tasks completed")

    main()
