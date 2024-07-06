import math
from collections import deque
from flexecutor.workflow.stage import Stage
from flexecutor.utils.dataclass import StageConfig


class DAG:
    """
    Class to represent a DAG

    :param dag_id: DAG ID
    """

    def __init__(self, dag_id):
        self._dag_id = dag_id
        self._stages = set()

    @property
    def dag_id(self):
        """Return the DAG ID"""
        return self._dag_id

    @property
    def stages(self) -> set[Stage]:
        """Return all stages in the DAG"""
        return self._stages

    @property
    def root_stages(self) -> set[Stage]:
        """
        Return all root stages in the DAG

        A root stage is a stage that has no parents.
        """
        return {stage for stage in self.stages if not stage.parents}

    @property
    def leaf_stages(self) -> set[Stage]:
        """
        Return all leaf stages in the DAG

        A leaf stage is a stage that has no children.
        """
        return {stage for stage in self.stages if not stage.children}

    def add_stage(self, stage: Stage):
        """
        Add a stage to this DAG

        :param stage: Stage to add
        :raises ValueError: if the stage is already in the DAG
        """
        stage.dag_id = self.dag_id

        if stage.stage_id in {t.stage_id for t in self.stages}:
            raise ValueError(
                f"Stage with id {stage.stage_id} already exists in DAG {self._dag_id}"
            )

        self._stages.add(stage)

    def add_stages(self, stages: list[Stage]):
        """
        Add a list of stages to this DAG

        :param stages: List of stages to add
        :raises ValueError: if any of the stages is already in the DAG
        """
        for stage in stages:
            self.add_stage(stage)

    def set_time_weights(self, mode: str):
        for stage in self.stages:
            prediction = stage.perf_model.predict_partial_factor(mode)
            print(f"Prediction for stage {stage.stage_id}: {prediction}")
            if prediction is None:
                print(
                    f"Failed to get a prediction for stage {stage.stage_id}, setting default time weight."
                )
                prediction = 1.0  # Set a default weight in case of failure to predict
            stage.time_weight = abs(prediction)

    # def distribute_parallelism_by_jct(self):
    #     """
    #     Distribute parallelism by job completion time
    #     """
    #     # https://github.com/pkusys/Ditto/blob/main/include/scheduler.hpp#L658
    #     for stage in self.stages:
    #         # If a stage is a leaf
    #         if len(stage.children) == 0:
    #             stage.ratio = stage.time_weight
    #             stage.parallelism_ratio = 1
    #             continue
    #         # If it's "single"
    #         if stage.max_concurrency == 1:
    #             stage.ratio = stage.time_weight
    #             stage.parallelism_ratio = 1
    #             continue
    #         if len(stage.parents) == 1:
    #             # Since it's only one parent, get the first parent
    #             parent = stage.parents[0]
    #             offspring = []
    #             queue = []
    #             queue.push(stage)
    #             pw = math.sqrt(stage.time_weight)
    #             cw = math.sqrt(parent.ratio)
    #             stage.ratio = (parent.ratio / cw + stage.time_weight / pw) * (cw + pw)
    #             stage.parallelism_ratio = pw / (cw + pw)
    #             sf = cw / (cw + pw)
    #             while not queue.empty():
    #                 cur = queue.front()
    #                 queue.pop()
    #                 for

    def distribute_parallelism_by_jct(self, total_slots=20):
        """
        Distribute parallelism by job completion time
        """
        para_ratios = {stage.stage_id: 0 for stage in self.stages}
        r = {stage.stage_id: 0 for stage in self.stages}

        # https://github.com/pkusys/Ditto/blob/main/include/scheduler.hpp#L658

        # Calculate initial weights and parallelism ratios for all stages
        for stage in self.stages:
            if len(stage.children) == 0:  # Leaf stage
                stage.time_weight = abs(stage.perf_model.predict_partial_factor("RCW"))
                r[stage.stage_id] = stage.time_weight
                para_ratios[stage.stage_id] = 1
            elif stage.max_concurrency == 1:  # Single concurrency stage
                child = next(iter(stage.children))
                stage.time_weight = abs(stage.perf_model.predict_partial_factor("RCW"))
                r[stage.stage_id] = (
                    r[child.stage_id] + stage.time_weight / para_ratios[child.stage_id]
                )
                para_ratios[stage.stage_id] = para_ratios[child.stage_id]
            elif len(stage.parents) == 1:  # Stages with exactly one parent
                child = next(iter(stage.children))
                offsprings = []
                queue = deque([stage])
                pw = math.sqrt(stage.time_weight)
                cw = math.sqrt(r[child.stage_id])
                r[stage.stage_id] = (
                    r[child.stage_id] / cw + stage.time_weight / pw
                ) * (cw + pw)
                para_ratios[stage.stage_id] = pw / (cw + pw)
                sf = cw / (cw + pw)

                while queue:
                    current_stage = queue.popleft()
                    for child_stage in current_stage.children:
                        queue.append(child_stage)
                        offsprings.append(child_stage)

                for offspring in offsprings:
                    para_ratios[offspring.stage_id] *= sf
            else:  # Stages with multiple parents
                children_weights = [r[child.stage_id] for child in stage.children]
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
                        para_ratios[offspring.stage_id] *= weight

                child = next(iter(stage.children))
                cw = math.sqrt(r[child.stage_id] / normalized_weights[0])
                pw = math.sqrt(stage.time_weight)
                r[stage.stage_id] = (
                    r[child.stage_id] / normalized_weights[0] / cw
                    + stage.time_weight / pw
                ) * (cw + pw)
                para_ratios[stage.stage_id] = pw / (cw + pw)
                sf = cw / (cw + pw)

                for offspring in alloffsprings:
                    para_ratios[offspring.stage_id] *= sf

        # Calculate total slots and total weight
        total_weight = sum(
            para_ratios[stage.stage_id]
            for stage in self.stages
            if stage.max_concurrency != 1
        )
        for stage in self.stages:
            if stage.max_concurrency == 1:
                stage.optimal_config = StageConfig(cpu=1, memory=1024, workers=1)
            else:
                allocated_workers = max(
                    int((para_ratios[stage.stage_id] / total_weight) * total_slots), 1
                )
                if not stage.optimal_config:
                    stage.optimal_config = StageConfig(
                        cpu=1, memory=1024, workers=allocated_workers
                    )
                else:
                    stage.optimal_config.workers = allocated_workers

        allocated_slots = sum(stage.optimal_config.workers for stage in self.stages)
        while allocated_slots > total_slots:
            for stage in sorted(
                self.stages, key=lambda x: x.optimal_config.workers, reverse=True
            ):
                if stage.optimal_config.workers > 1:
                    stage.optimal_config.workers -= 1
                    allocated_slots -= 1
                    if allocated_slots <= total_slots:
                        break

        for stage in self.stages:
            workers = (
                stage.optimal_config.workers if stage.optimal_config else "Not set"
            )
            print(
                f"Stage {stage.stage_id}: Time Weight = {stage.time_weight}, Workers = {workers}"
            )

    def draw(self, filename="dag.png"):
        """
        Draw the DAG for user visualization and save it to a file.

        Parameters:
            filename (str): The name of the file to save the image to.
        """
        import networkx as nx
        from matplotlib import pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout

        graph = nx.DiGraph()
        for stage in self.stages:
            graph.add_node(stage.stage_id, label=stage.stage_id)
            for parent in stage.parents:
                graph.add_edge(parent.stage_id, stage.stage_id)

        pos = graphviz_layout(graph, prog="dot")
        labels = nx.get_node_attributes(graph, "label")

        plt.figure(figsize=(8, 6))
        plt.title(self.dag_id, fontsize=15, fontweight="bold")
        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            node_size=2000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            arrows=True,
        )
        plt.savefig(filename)
        plt.close()
