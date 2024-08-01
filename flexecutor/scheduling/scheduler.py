from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, dag, perf_model_type):
        self._dag = dag
        self._perf_model_type = perf_model_type
        for stage in self._dag.stages:
            stage.init_perf_model(perf_model_type)

    @abstractmethod
    def schedule(self):
        """
        This method purpose is to calculate and set the resource configuration of the
        different stages in the DAG.
        """
        raise NotImplementedError
