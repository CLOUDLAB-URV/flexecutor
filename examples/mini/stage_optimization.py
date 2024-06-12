from lithops import LocalhostExecutor

from examples.mini.functions.word_occurrence import word_occurrence_count
from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.dataclass import StageConfig, ConfigBounds
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

NUM_CONFIGS = 5

if __name__ == "__main__":
    @flexorchestrator
    def main(num_configs=NUM_CONFIGS):
        dag = DAG('large-example-dag')

        stage1 = Stage(
            'stage1',
            func=word_occurrence_count,
            perf_model_type=PerfModelEnum.GENETIC,
            # input_file="test-bucket/corpus.txt"
        )

        dag.add_stages([stage1])
        executor = DAGExecutor(dag, executor=LocalhostExecutor())

        config_space = [
            (3, 1024, 2),  # 1 vCPU, 512 MB per worker, 10 workers
            (1, 200, 10),  # 1 vCPU, 200 MB per worker, 10 workers
            (2, 2048, 7),  # 2 vCPUs, 2048 MB per worker, 7 workers
            (3, 3072, 5),  # 3 vCPUs, 3072 MB per worker, 5 workers
            (1, 512, 15),  # 1 vCPU, 512 MB per worker, 15 workers
            (1, 1024, 10),  # 1 vCPU, 1024 MB per worker, 10 workers
            (2, 2048, 5),  # 2 vCPUs, 2048 MB per worker, 5 workers
            (2, 1707, 6),  # 2 vCPUs, 1707 MB per worker, 6 workers
            (3, 3413, 3),  # 3 vCPUs, 3413 MB per worker, 3 workers
            (4, 4096, 4),  # 4 vCPUs, 4096 MB per worker, 4 workers
            (4, 5120, 2),  # 4 vCPUs, 5120 MB per worker, 2 workers
            (6, 10240, 1),  # 6 vCPUs, 10240 MB per worker, 1 worker
            (1, 2048, 10),  # 1 vCPU, 2048 MB per worker, 10 workers
            (2, 4096, 8),  # 2 vCPUs, 4096 MB per worker, 8 workers
            (3, 6144, 6),  # 3 vCPUs, 6144 MB per worker, 6 workers
            (4, 8192, 4),  # 4 vCPUs, 8192 MB per worker, 4 workers
            (5, 10240, 2),  # 5 vCPUs, 10240 MB per worker, 2 workers
            (6, 12288, 1),  # 6 vCPUs, 12288 MB per worker, 1 worker
        ]
        config_space_obj = [StageConfig(*resource_config) for resource_config in config_space]
        num_configs = min(num_configs, len(config_space_obj) - 1)
        config_space_obj = config_space_obj[:num_configs]

        # Profile the DAG
        executor.profile(config_space_obj, num_iterations=2)

        # Train the stage models
        executor.train()

        # Print the objective function
        objective_function = stage1.perf_model.objective_func
        print(f"Objective function {objective_function}")

        bounds = ConfigBounds(*[(1, 6), (512, 4096), (1, 3)])

        # Get the optimal configuration for the stage
        [optimal_config] = executor.optimize(bounds, stage1)
        print(optimal_config)
        [predicted_latency] = executor.predict([optimal_config], stage1)
        print("Predicted latency", predicted_latency)

        # Execute the stage with the optimal config
        stage1.resource_config = optimal_config
        timings = executor.execute_stage(stage1)
        executor.shutdown()

        # Print metrics
        actual_latency = max([i.read + i.cold_start + i.compute + i.write for i in timings])
        print("Actual latency", actual_latency)
        print(f"Accuracy {100 - (actual_latency - predicted_latency.total) / predicted_latency.total * 100} %")

    main()
