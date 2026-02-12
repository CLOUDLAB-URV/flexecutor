from abc import ABC, abstractmethod

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.dag import DAG


class Scheduler(ABC):
    def __init__(self, dag: DAG, perf_model_type: PerfModelEnum):
        self._dag = dag
        self._perf_model_type = perf_model_type
        for stage in self._dag.stages:
            perf_model = stage.init_perf_model(perf_model_type)
            perf_model.init_from_dag(self._dag)

    @abstractmethod
    def schedule(self) -> list[StageConfig]:
        """
        This method purpose is to calculate and set the resource configuration of the
        different stages in the DAG.
        """
        raise NotImplementedError
