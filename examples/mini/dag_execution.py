from __future__ import annotations

import logging

from lithops import FunctionExecutor

from functions.word_count import (
    word_count,
    word_count_input,
    word_count_output,
)
from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.utils import setup_logging

logger = setup_logging(level=logging.INFO)

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}


NUM_ITERATIONS = 1


if __name__ == "__main__":

    @flexorchestrator()
    def main():
        dag = DAG("mini-dag")

        stage1 = Stage(
            "stage1",
            func=word_count,
            perf_model_type=PerfModelEnum.GENETIC,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )
        stage2 = Stage(
            "stage2",
            func=word_count,
            perf_model_type=PerfModelEnum.GENETIC,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )
        stage3 = Stage(
            "stage3",
            func=word_count,
            perf_model_type=PerfModelEnum.GENETIC,
            inputs=[word_count_input],
            outputs=[word_count_output],
        )

        stage1 >> stage2 << stage3

        dag.add_stages([stage1, stage2, stage3])

        executor = DAGExecutor(dag, executor=FunctionExecutor())
        executor.execute()
        executor.shutdown()
        print("Tasks completed")

    main()
