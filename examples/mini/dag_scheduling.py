from __future__ import annotations

import logging

from lithops import FunctionExecutor

from functions.word_count import (
    word_count,
    sum_counts,
    flex_data_txt,
    flex_data_word_count,
    flex_data_reduce_count,
)
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.utils import setup_logging
from scheduling.jolteon import Jolteon

logger = setup_logging(level=logging.INFO)

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}

NUM_ITERATIONS = 2


if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("mini-dag")

        stage1 = Stage(
            "stage0",
            func=word_count,
            inputs=[flex_data_txt],
            outputs=[flex_data_word_count],
        )
        stage2 = Stage(
            "stage1",
            func=sum_counts,
            inputs=[flex_data_word_count],
            outputs=[flex_data_reduce_count],
            max_concurrency=1,
        )

        stage1 >> stage2

        dag.add_stages([stage1, stage2])

        executor = DAGExecutor(dag, executor=FunctionExecutor())
        scheduler = Jolteon(dag, total_parallelism=10, cpu_per_worker=1)

        executor.train()
        scheduler.schedule()

        executor.shutdown()
        print("Tasks completed")

    main()
