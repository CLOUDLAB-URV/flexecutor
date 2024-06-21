from __future__ import annotations

import logging

from lithops import FunctionExecutor

from examples.mini.functions.word_count import (
    word_count,
    word_count_input,
    word_count_output,
)
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor, StageConfig
from flexecutor.workflow.stage import Stage

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)

NUM_ITERATIONS = 1
BUCKET_NAME = "lithops-manri-urv"


if __name__ == "__main__":

    @flexorchestrator()
    def main():
        config_spaces = [
            StageConfig(2, 1024, 3),
            StageConfig(0.5, 1568, 5),
            # ...
        ]

        dag = DAG("mini-dag")

        stage1 = Stage(
            "stage1",
            func=word_count,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )
        stage2 = Stage(
            "stage2",
            func=word_count,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )
        stage3 = Stage(
            "stage3",
            func=word_count,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )

        stage1 >> stage2 << stage3

        dag.add_stages([stage1, stage2, stage3])

        executor = DAGExecutor(dag, executor=FunctionExecutor())
        executor.profile(config_spaces, num_iterations=NUM_ITERATIONS)
        executor.shutdown()
        print("Tasks completed")

    main()
