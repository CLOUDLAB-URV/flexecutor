from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, dag):
        self._dag = dag

    @abstractmethod
    def schedule(self):
        """
        This method purpose is to calculate and set the resource configuration of the
        different stages in the DAG.
        """
        raise NotImplementedError
