from typing import Optional
from flexecutor.workflow.stage import Stage


class DAG:
    """
    Class to represent a DAG

    :param dag_id: DAG ID
    """

    def __init__(self, dag_id):
        self._dag_id = dag_id
        self._stages = list()
        self._stage_counter = 0

    def __iter__(self):
        return iter(self.stages)

    def get_stage_by_id(self, stage_id: str) -> Stage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage

    @property
    def dag_id(self):
        """Return the DAG ID"""
        return self._dag_id

    @property
    def stages(self) -> list[Stage]:
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
        stage.stage_idx = self._stage_counter
        self._stage_counter += 1

        if stage.stage_id in {t.stage_id for t in self.stages}:
            raise ValueError(
                f"Stage with id {stage.stage_id} already exists in DAG {self._dag_id}"
            )

        self._stages.append(stage)

    def add_stages(self, stages: list[Stage]):
        """
        Add a list of stages to this DAG

        :param stages: List of stages to add
        :raises ValueError: if any of the stages is already in the DAG
        """
        for stage in stages:
            self.add_stage(stage)

    def draw(self, filename: Optional[str] = None):
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
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
