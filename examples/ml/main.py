from examples.ml.functions import pca, train_with_multiprocessing, aggregate, test
from flexecutor.storage.storage import FlexData, StrategyEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from scheduling.jolteon import Jolteon
from scheduling.orion import Orion
from utils.dataclass import StageConfig

if __name__ == "__main__":

    @flexorchestrator(bucket="flexecutor-data")
    def main():
        dag = DAG("machine_learning")

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
            stage_id="stage0",
            func=pca,
            inputs=[data_training],
            outputs=[data_vectors_pca, data_training_transform],
            params={"n_components": 2},
            max_concurrency=1,
        )

        stage1 = Stage(
            stage_id="stage1",
            func=train_with_multiprocessing,
            inputs=[data_training_transform],
            outputs=[data_models],
        )

        stage2 = Stage(
            stage_id="stage2",
            func=aggregate,
            inputs=[data_training_transform, data_models],
            outputs=[data_forests, data_predictions],
        )

        stage3 = Stage(
            stage_id="stage3",
            func=test,
            inputs=[data_predictions, data_training_transform],
            outputs=[data_accuracies],
            max_concurrency=1,
        )

        stage0 >> [stage1, stage2, stage3]
        stage1 >> stage2
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])

        entry_point = [
            StageConfig(workers=workers, cpu=cpu)
            for workers, cpu in zip([1, 16, 8, 1], [3] * 4)
        ]
        x_bounds = [
            StageConfig(workers=workers, cpu=cpu)
            for workers, cpu in zip(
                [1, 0.5] + [4, 0.5] * 2 + [1, 0.5], [2, 4.1] + [32, 4.1] * 2 + [2, 4.1]
            )
        ]

        executor = DAGExecutor(
            dag,
            # scheduler=Jolteon(
            #     dag,
            #     bound=20,
            #     bound_type="latency",
            #     cpu_search_space=[0.6, 1, 1.5, 2, 2.5, 3, 4],
            #     entry_point=entry_point,
            #     x_bounds=x_bounds,
            # ),
            scheduler=Orion(dag, latency=30, total_parallelism=16),
            # scheduler=Ditto(
            #     total_parallelism=10,
            #     dag=dag,
            #     cpu_per_worker=0.6,
            #     objective="latency"
            # )
            # scheduler=Caerus(
            #     total_parallelism=16,
            #     dag=dag,
            #     cpu_per_worker=2.5,
            # )
        )

        # executor.profile(config_space=[
        #     {
        #         "stage0": StageConfig(cpu=1, memory=2048, workers=4),
        #         "stage1": StageConfig(cpu=1, memory=2048, workers=4),
        #         "stage2": StageConfig(cpu=1, memory=2048, workers=4),
        #         "stage3": StageConfig(cpu=1, memory=2048, workers=4),
        #
        #     },
        # ], callback=clean_ml_data, num_reps=2)

        executor.optimize()
        executor.shutdown()

    main()
