from lithops import FunctionExecutor

from examples.ml.functions import pca, train_with_multiprocessing, aggregate, test
from flexecutor.storage.storage import FlexInput, StrategyEnum, FlexOutput
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("machine-learning")

        stage0 = Stage(
            stage_id="stage0",
            func=pca,
            inputs=[FlexInput(prefix="training-data")],
            outputs=[
                FlexOutput(
                    prefix="vectors-pca",
                ),
                FlexOutput(
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
                    prefix="training-data-transform",
                    strategy=StrategyEnum.BROADCAST,
                )
            ],
            outputs=[
                FlexOutput(
                    prefix="models",
                )
            ],
        )

        stage2 = Stage(
            stage_id="stage2",
            func=aggregate,
            inputs=[
                FlexInput(
                    prefix="training-data-transform",
                    strategy=StrategyEnum.BROADCAST,
                ),
                FlexInput("models", prefix="models"),
            ],
            outputs=[
                FlexOutput(
                    prefix="forests",
                ),
                FlexOutput(
                    prefix="predictions",
                ),
            ],
        )

        stage3 = Stage(
            stage_id="stage3",
            func=test,
            inputs=[
                FlexInput(prefix="predictions"),
                FlexInput(
                    prefix="training-data-transform",
                    strategy=StrategyEnum.BROADCAST,
                ),
            ],
            outputs=[
                FlexOutput(
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
