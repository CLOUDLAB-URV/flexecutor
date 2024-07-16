from ast import List
import numpy as np


from flexecutor.optimization.solver import OptimizationSolver
from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.dag import DAG


class BruteforceSolver(OptimizationSolver):
    def __init__(self, problem):
        super().__init__(problem)

    def solve(self, dag: DAG, dag_critical_path: List, config_bounds: StageConfig):

        # range num_workers = 1...10, range_cpu 0.5 to 4, whatever amazon gives you
        # step 1: profile stages with tuples of stages, each stage could be trained on a different set of stageconfigs
        # step 2: train the analytical model with those profiled configurations.
        # step 3: once trained, the optimize function should predict the latency for each stage in the critical path a bruteforce manner for the given config_bounds
        def calculate_memory_for_cpus(cpus: int) -> int:
            memory = cpus * 1769
            return memory

        cpu_combinations = np.arange(
            config_bounds.cpu[0], config_bounds.cpu[1] + 0.5, 0.5
        )
        worker_combinations = np.arange(
            config_bounds.workers[0], config_bounds.workers[1] + 1, 1
        )

        for stage in dag:
            if stage.stage_id in dag_critical_path:
                for cpu in cpu_combinations:
                    for worker in worker_combinations:
                        if stage.max_concurrency == 1:
                            worker = 1
                        memory = calculate_memory_for_cpus(cpu)
                        if memory <= config_bounds.memory[1]:
                            predicted_time = stage.perf_model.predict_time(
                                StageConfig(cpu=cpu, memory=memory, workers=worker)
                            )
                            if predicted_time < stage.perf_model.predict_time(
                                stage.resource_config
                            ):
                                stage.resource_config = StageConfig(
                                    cpu=cpu, memory=memory, workers=worker
                                )

        for stage in dag:
            print(
                f"Optimal configuration for stage {stage.stage_id}: {stage.resource_config}"
            )
