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
from flexecutor.utils.utils import setup_logging, flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor, ConfigBounds
from flexecutor.workflow.stage import Stage

logger = setup_logging(level=logging.INFO)

config = {"lithops": {"backend": "localhost", "storage": "localhost"}}

NUM_ITERATIONS = 2


if __name__ == "__main__":

    @flexorchestrator()
    def main():
        dag_critical_path = ["map", "reduce"]

        dag = DAG("mini-dag")

        stage1 = Stage(
            "map",
            func=word_count,
            inputs=[flex_data_txt],
            outputs=[flex_data_word_count],
        )
        stage2 = Stage(
            "reduce",
            func=sum_counts,
            inputs=[flex_data_word_count],
            outputs=[flex_data_reduce_count],
            max_concurrency=1,
        )

        stage1 >> stage2

        dag.add_stages([stage1, stage2])

        executor = DAGExecutor(dag, executor=FunctionExecutor(runtime_cpus=1))
        config_bounds = ConfigBounds(
            cpu=(0.5, 4.5), memory=(1024, 4096), workers=(1, 10)
        )
        executor.train()
        executor.optimize(dag_critical_path, config_bounds)
        executor.shutdown()
        print("Tasks completed")

    main()
