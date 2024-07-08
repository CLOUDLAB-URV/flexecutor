import math

from collections import deque
from enum import Enum

from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.dag import DAG


class StrategyEnum(Enum):
    SCATTER = 1
    BROADCAST = 2


class DittoScheduler:
    def __init__(self, total_slots: int, dag: DAG):
        self.total_slots = total_slots
        self.dag = dag

    def set_time_weights(self, mode: str = "RCW"):
        for stage in self.dag:
            prediction = stage.perf_model.predict_partial_factor(mode)
            print(f"Prediction for stage {stage.stage_id}: {prediction}")
            if prediction is None:
                print(
                    f"Failed to get a prediction for stage {stage.stage_id}, setting default time weight."
                )
                prediction = 1.0  # Set a default weight in case of failure to predict
            stage.time_weight = abs(prediction)

    def distribute_parallelism_by_jct(self):
        """
        Distribute parallelism by job completion time, adjusting parallelism allocation based on job characteristics and dependencies.
        """
        # From: https://github.com/pkusys/Ditto/blob/main/include/scheduler.hpp#L603
        parallelism_allocation_ratios = {stage.stage_id: 0 for stage in self.dag}
        cumulative_loads = {stage.stage_id: 0 for stage in self.dag}

        sorted_stages = self.topological_sort()[
            ::-1
        ]  # Reverse topologically sorted order

        for stage in sorted_stages:
            print(f"Processing stage {stage.stage_id}")
            if len(stage.children) == 0:  # Leaf stage
                stage.time_weight = abs(stage.perf_model.predict_partial_factor("RCW"))
                cumulative_loads[stage.stage_id] = stage.time_weight
                parallelism_allocation_ratios[stage.stage_id] = 1
                print(
                    f"Leaf stage {stage.stage_id}: time_weight={stage.time_weight}, cumulative_load={cumulative_loads[stage.stage_id]}"
                )
            elif stage.max_concurrency == 1:  # Single concurrency stage
                child = next(iter(stage.children))
                stage.time_weight = abs(stage.perf_model.predict_partial_factor("RCW"))
                cumulative_loads[stage.stage_id] = (
                    cumulative_loads[child.stage_id]
                    + stage.time_weight / parallelism_allocation_ratios[child.stage_id]
                )
                parallelism_allocation_ratios[stage.stage_id] = (
                    parallelism_allocation_ratios[child.stage_id]
                )
                print(
                    f"Single concurrency stage {stage.stage_id}: updated cumulative_load={cumulative_loads[stage.stage_id]}"
                )
            elif len(stage.parents) == 1:  # Stages with exactly one parent
                child = next(iter(stage.children))
                offsprings = []
                queue = deque([stage])
                pw = math.sqrt(stage.time_weight)
                cw = math.sqrt(cumulative_loads[child.stage_id])
                cumulative_loads[stage.stage_id] = (
                    cumulative_loads[child.stage_id] / cw + stage.time_weight / pw
                ) * (cw + pw)
                parallelism_allocation_ratios[stage.stage_id] = pw / (cw + pw)
                sf = cw / (cw + pw)

                while queue:
                    current_stage = queue.popleft()
                    for child_stage in current_stage.children:
                        queue.append(child_stage)
                        offsprings.append(child_stage)

                for offspring in offsprings:
                    parallelism_allocation_ratios[offspring.stage_id] *= sf
                    print(
                        f"Updated parallelism ratio for offspring {offspring.stage_id} = {parallelism_allocation_ratios[offspring.stage_id]}"
                    )
            else:  # Stages with multiple parents
                children_weights = [
                    cumulative_loads[child.stage_id] for child in stage.children
                ]
                total_children_weight = sum(children_weights)
                normalized_weights = [
                    weight / total_children_weight for weight in children_weights
                ]

                offsprings = []
                alloffsprings = []
                for weight, child in zip(normalized_weights, stage.children):
                    queue = deque([child])
                    while queue:
                        current_stage = queue.popleft()
                        alloffsprings.append(current_stage)
                        for grandchild in current_stage.children:
                            queue.append(grandchild)

                    for offspring in alloffsprings:
                        parallelism_allocation_ratios[offspring.stage_id] *= weight

                child = next(iter(stage.children))
                cw = math.sqrt(cumulative_loads[child.stage_id] / normalized_weights[0])
                pw = math.sqrt(stage.time_weight)
                cumulative_loads[stage.stage_id] = (
                    cumulative_loads[child.stage_id] / normalized_weights[0] / cw
                    + stage.time_weight / pw
                ) * (cw + pw)
                parallelism_allocation_ratios[stage.stage_id] = pw / (cw + pw)
                sf = cw / (cw + pw)

                for offspring in alloffsprings:
                    parallelism_allocation_ratios[offspring.stage_id] *= sf

                print(
                    f"Multi-parent stage {stage.stage_id}: cumulative_load updated to {cumulative_loads[stage.stage_id]}"
                )

        total_weight = sum(
            parallelism_allocation_ratios[stage.stage_id]
            for stage in self.dag
            if stage.max_concurrency != 1
        )
        for stage in self.dag:
            if stage.max_concurrency == 1:
                stage.optimal_config = StageConfig(cpu=1, memory=1024, workers=1)
            else:
                allocated_workers = max(
                    int(
                        (parallelism_allocation_ratios[stage.stage_id] / total_weight)
                        * self.total_slots
                    ),
                    1,
                )
                if not stage.optimal_config:
                    stage.optimal_config = StageConfig(
                        cpu=1, memory=1024, workers=allocated_workers
                    )
                else:
                    stage.optimal_config.workers = allocated_workers

        allocated_slots = sum(stage.optimal_config.workers for stage in self.dag)
        while allocated_slots > self.total_slots:
            for stage in sorted(
                self.dag, key=lambda x: x.optimal_config.workers, reverse=True
            ):
                if stage.optimal_config.workers > 1:
                    stage.optimal_config.workers -= 1
                    allocated_slots -= 1
                    if allocated_slots <= self.total_slots:
                        break

        for stage in self.dag:
            workers = (
                stage.optimal_config.workers if stage.optimal_config else "Not set"
            )
            print(
                f"Final allocation - Stage {stage.stage_id}: Time Weight = {stage.time_weight}, Workers = {workers}"
            )

        print(f"Tasks completed with total_slots={self.total_slots}")

    def topological_sort(self):
        """
        Perform topological sort on the DAG and return a list of stages in topologically sorted order.
        """

        in_degree = {stage.stage_id: 0 for stage in self.dag}
        for stage in self.dag:
            for child in stage.children:
                in_degree[child.stage_id] += 1

        queue = deque([stage for stage in self.dag if in_degree[stage.stage_id] == 0])
        sorted_stages = []

        while queue:
            current_stage = queue.popleft()
            sorted_stages.append(current_stage)
            for child in current_stage.children:
                in_degree[child.stage_id] -= 1
                if in_degree[child.stage_id] == 0:
                    queue.append(child)

        return sorted_stages

    # adjusts worker allocation based on the input data, assumes chunks are already created in object storage
    # finetune_degree_alloc in ditto
    def adjust_worker_allocation(self):
        pass
