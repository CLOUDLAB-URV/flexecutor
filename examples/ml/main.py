from lithops import FunctionExecutor

from examples.ml.functions import pca, train_with_multiprocessing, aggregate, test
from flexecutor.storage.storage import FlexData, StrategyEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from scheduling.jolteon import Jolteon

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("machine-learning")

        data_training = FlexData("training-data")
        data_vectors_pca = FlexData("vectors-pca")
        data_training_transform = FlexData(
            "training-data-transform", read_strategy=StrategyEnum.BROADCAST
        )
        data_models = FlexData("models")
        data_forests = FlexData("forests")
        data_predictions = FlexData("predictions")
        data_accuracies = FlexData("accuracies", suffix=".txt")

        stage0 = Stage(
            stage_id="0",
            func=pca,
            inputs=[data_training],
            outputs=[data_vectors_pca, data_training_transform],
            params={"n_components": 2},
            max_concurrency=1,
        )

        stage1 = Stage(
            stage_id="1",
            func=train_with_multiprocessing,
            inputs=[data_training_transform],
            outputs=[data_models],
        )

        stage2 = Stage(
            stage_id="2",
            func=aggregate,
            inputs=[data_training_transform, data_models],
            outputs=[data_forests, data_predictions],
        )

        stage3 = Stage(
            stage_id="3",
            func=test,
            inputs=[data_predictions, data_training_transform],
            outputs=[data_accuracies],
            max_concurrency=1
        )

        stage0 >> [stage1, stage2, stage3]
        stage1 >> stage2
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])

        executor = DAGExecutor(dag, executor=FunctionExecutor(log_level="INFO"))
        # results = executor.execute(num_workers=6)
        # print(results["stage1"].get_timings())

        scheduler = Jolteon(dag, total_parallelism=10, cpu_per_worker=1)

        executor.train()
        scheduler.schedule()

        executor.shutdown()

    main()
