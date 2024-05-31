from abc import abstractmethod, ABC
from typing import Dict

from flexecutor.modelling.prediction import Prediction


class PerfModel(ABC):
    def __init__(self, model_type=None):
        self._model_type = model_type
        self._objective_func = None

    @abstractmethod
    def train(self, profiling_results: Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, num_cpu, runtime_memory, num_workers, chunk_size=None) -> Prediction:
        raise NotImplementedError

    @property
    def objective_func(self):
        return self._objective_func

    @property
    def model_type(self):
        return self._model_type

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    @classmethod
    def instance(cls, model_type):
        if model_type == 'analytic':
            from flexecutor.modelling.anaperfmodel import AnaPerfModel
            return AnaPerfModel(stage_id=0, stage_name='stage')
        elif model_type == 'genetic':
            from flexecutor.modelling.gaperfmodel import GAPerfModel
            return GAPerfModel()
        else:
            raise ValueError("Invalid model type")
