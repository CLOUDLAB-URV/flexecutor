from typing import Any

from lithops import FunctionExecutor

from examples.ml.functions import pca, train_with_multiprocessing, aggregate, test
from flexecutor.storage.storage import FlexInput, StrategyEnum, FlexOutput
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage


if __name__ == "__main__":

    @flexorchestrator
    def main():
        dag = DAG("machine-learning")

        stage0 = Stage(
            stage_id="stage0",
            func=pca,
            inputs=[
                FlexInput("training-data", bucket="test-bucket", prefix="training-data")
            ],
            outputs=[
                FlexOutput(
                    "vectors-pca",
                    bucket="test-bucket",
                    prefix="vectors-pca",
                ),
                FlexOutput(
                    "training-data-transform",
                    bucket="test-bucket",
                    prefix="training-data-transform",
                ),
            ],
            params={"n_components": 2},
            max_concurrency=1,
        )

        stage1 = Stage(
            stage_id="stage1",
            func=train_with_multiprocessing,
            inputs=[
                FlexInput(
                    "training-data-transform",
                    bucket="test-bucket",
                    prefix="training-data-transform",
                    strategy=StrategyEnum.BROADCAST,
                )
            ],
            outputs=[
                FlexOutput(
                    "models",
                    bucket="test-bucket",
                    prefix="models",
                )
            ],
        )

        stage2 = Stage(
            stage_id="stage2",
            func=aggregate,
            inputs=[
                FlexInput(
                    "training-data-transform",
                    bucket="test-bucket",
                    prefix="training-data-transform",
                    strategy=StrategyEnum.BROADCAST,
                ),
                FlexInput("models", bucket="test-bucket", prefix="models"),
            ],
            outputs=[
                FlexOutput(
                    "forests",
                    bucket="test-bucket",
                    prefix="forests",
                ),
                FlexOutput(
                    "predictions",
                    bucket="test-bucket",
                    prefix="predictions",
                ),
            ],
        )

        stage3 = Stage(
            stage_id="stage3",
            func=test,
            inputs=[
                FlexInput("predictions", bucket="test-bucket", prefix="predictions"),
                FlexInput(
                    "training-data-transform",
                    bucket="test-bucket",
                    prefix="training-data-transform",
                    strategy=StrategyEnum.BROADCAST,
                ),
            ],
            outputs=[
                FlexOutput(
                    "accuracies",
                    bucket="test-bucket",
                    prefix="accuracies",
                    suffix=".txt",
                )
            ],
        )

        stage0 >> [stage1, stage2, stage3]
        stage1 >> stage2
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])

        executor = DAGExecutor(dag, executor=FunctionExecutor(log_level="INFO"))
        results = executor.execute()
        print(results["stage1"].get_timings())

    main()
