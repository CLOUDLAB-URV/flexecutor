from __future__ import annotations

import logging
import time

from lithops import FunctionExecutor

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.storage.storage import FlexOutput
from flexecutor.storage.storage import FlexInput
from flexecutor.utils import setup_logging
from flexecutor.utils.iomanager import IOManager
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

logger = setup_logging(level=logging.INFO)


NUM_ITERATIONS = 1
BUCKET_NAME = "test-bucket"
input_file = FlexInput("txt", bucket=BUCKET_NAME, prefix="dir")
output_path = FlexOutput("count", bucket=BUCKET_NAME, prefix="count")


@flexorchestrator
def main():
    dag = DAG("mini-dag")

    def word_count(io: IOManager):
        txt_paths = io.input_paths("txt")
        for txt_path in txt_paths:
            with open(txt_path, "r") as f:
                content = f.read()

            count = len(content.split())

            count_path = io.next_output_path("count")
            with open(count_path, "w") as f:
                f.write(str(count))

    stage1 = Stage(
        "stage1",
        func=word_count,
        perf_model_type=PerfModelEnum.GENETIC,
        inputs=[input_file],
        outputs=[output_path],
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
