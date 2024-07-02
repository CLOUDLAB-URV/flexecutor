from __future__ import annotations

import logging

from lithops import FunctionExecutor

from examples.mini.functions.word_count import (
    word_count,
    sum_counts,
    word_count_input,
    word_count_output,
    reduce_input,
    reduce_output,
)
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor, StageConfig
from flexecutor.workflow.stage import Stage
from flexecutor.utils import setup_logging

logger = setup_logging(level=logging.INFO)

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}

NUM_ITERATIONS = 1
BUCKET_NAME = "lithops-manri-urv"


if __name__ == "__main__":

    @flexorchestrator()
    def main():
        config_space = [
            StageConfig(2, 1024, 3),
            StageConfig(1, 1024, 3),
            # ...
        ]

        dag = DAG("mini-dag")

        stage1 = Stage(
            "map",
            func=word_count,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )
        stage2 = Stage(
            "reduce",
            func=sum_counts,
            inputs=[reduce_input],
            outputs=[reduce_output],
            max_concurrency=1,
        )

        stage1

        dag.add_stages([stage1])

        executor = DAGExecutor(dag, executor=FunctionExecutor())
        executor.profile(config_space, num_iterations=NUM_ITERATIONS)
        # executor.execute()
        executor.shutdown()
        print("Tasks completed")

    main()
