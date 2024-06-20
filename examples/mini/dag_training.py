from __future__ import annotations

import logging

from lithops import FunctionExecutor

from examples.mini.functions.word_count import (
    word_count,
    word_count_output,
    word_count_input,
)
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

    @flexorchestrator
    def main():
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
        executor.train()

        prediction = executor.predict(
            [StageConfig(cpu=2, memory=1024, workers=3)], stage1
        )
        print(prediction)

        executor.shutdown()
        print("stages completed")

    main()
