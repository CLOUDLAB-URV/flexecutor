from __future__ import annotations

import logging

from lithops import FunctionExecutor

from examples.mini.functions.word_count import (
    word_count_input,
    word_count_output,
    word_count,
)
from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.dataclass import StageConfig
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)

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

        stage2 << stage1

        dag.add_stages([stage1, stage2])

        executor = DAGExecutor(dag, executor=FunctionExecutor())
        executor.train()

        executor.plot_model_performance(
            stage1,
            [
                StageConfig(cpu=2, memory=1024, workers=3),
                StageConfig(cpu=0.5, memory=1568, workers=5),
            ],
        )

        executor.shutdown()
        print("stages completed")

    main()
