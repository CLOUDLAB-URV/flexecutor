from __future__ import annotations

import logging

from lithops import FunctionExecutor

from examples.mini.functions.word_count import (
    word_count,
    word_count_input,
    word_count_output,
)
from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils import setup_logging
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

logger = setup_logging(level=logging.INFO)


NUM_ITERATIONS = 1


@flexorchestrator
def main():
    dag = DAG("mini-dag")

    stage1 = Stage(
        "stage1",
        func=word_count,
        perf_model_type=PerfModelEnum.GENETIC,
        inputs=[word_count_input],
        outputs=[word_count_output],
    )

    dag.add_stages([stage1])

    executor = DAGExecutor(dag, executor=FunctionExecutor())

    logger.info("Starting DAG execution")
    results = executor.execute()
    times = results["stage1"].get_timings()
    print(times)
    executor.shutdown()
    logger.info("Tasks completed")


if __name__ == "__main__":
    main()
