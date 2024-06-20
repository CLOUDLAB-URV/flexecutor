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


@flexorchestrator
def main():
    config_spaces = [
        StageConfig(0.5, 256, 2),
        # ...
    ]

    dag = DAG("stage-obj-profiling")

    stage1 = Stage(
        "stage1",
        func=word_count,
        inputs=[word_count_input],
        outputs=[word_count_output],
    )

    dag.add_stages([stage1])

    executor = DAGExecutor(dag, executor=FunctionExecutor())
    executor.profile(config_spaces, num_iterations=1)
    executor.shutdown()

    print("stages completed")


if __name__ == "__main__":
    main()
