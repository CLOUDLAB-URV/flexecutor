from __future__ import annotations

import logging

from lithops import FunctionExecutor

from functions.word_count import (
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

NUM_ITERATIONS = 2


if __name__ == "__main__":

    @flexorchestrator()
    def main():
        # FIXME: Look at how many cpus is the profiling setting at the runtime
        # FIXME: We want a list of tuples of stageconfigs, for each stage,
        # we want to try different configs
        dag_critical_path = ["map", "reduce"]

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

        stage1 >> stage2

        dag.add_stages([stage1, stage2])

        # FIXME: Lithop's map doesn't have a runtime_cpus to set them at runtime with optimal config,
        # Since I decided to keep the number of cpus fixed (for now), we just pass it to the function executor as an argument beforehand.
        # Related to issue: https://github.com/lithops-cloud/lithops/issues/1371

        executor = DAGExecutor(dag, executor=FunctionExecutor(runtime_cpus=1))
        executor.train()
        executor.optimize(
            dag_critical_path,
        )
        executor.shutdown()
        print("Tasks completed")

    main()
