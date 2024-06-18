from __future__ import annotations

import logging
import time

from lithops import FunctionExecutor

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.storage.storage import OutputS3Path
from flexecutor.storage.storage import InputS3File
from flexecutor.utils import setup_logging
from flexecutor.utils.iomanager import IOManager
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

logger = setup_logging(level=logging.INFO)


NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"
input_file = InputS3File(BUCKET_NAME, "dir/tiny-shakespeare.txt")
output_path = OutputS3Path(BUCKET_NAME, "count")


@flexorchestrator
def main():
    dag = DAG("mini-dag")

    def word_count(io: IOManager):
        worker_id = io.params["worker_id"]
        print(f"I'm worker_id #{worker_id}")

        [txt_path] = io.input_file_func("txt")
        with open(txt_path, "r") as f:
            content = f.read()

        count = len(content.split())

        count_path = io.output_paths("count")
        with open(count_path, "w") as f:
            f.write(str(count))

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
    results = executor.execute()
    times = results['stage1'].get_timings()
    print(times)
    executor.shutdown()
    logger.info("Tasks completed")


if __name__ == "__main__":
    main()
