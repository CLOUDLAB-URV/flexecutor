from abc import abstractmethod, ABC
from typing import Dict

from flexecutor.modelling.prediction import Prediction


class PerfModel(ABC):
    def __init__(self, model_type, model_name, model_dst):
        self._model_name = model_name
        self._model_dst = model_dst
        self._model_type = model_type
        self._objective_func = None

    @abstractmethod
    def train(self, profiling_results: Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, num_cpu, runtime_memory, num_workers, chunk_size=None) -> Prediction:
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
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
    def instance(cls, model_type, model_name='default', model_dst=None):
        if model_dst is None:
            model_dst = 'models' + '/' + model_name + '.pkl'
        if model_type == 'analytic':
            from flexecutor.modelling.anaperfmodel import AnaPerfModel
            return AnaPerfModel(stage_id=0, stage_name='stage', model_name=model_name, model_dst=model_dst)
        elif model_type == 'genetic':
            from flexecutor.modelling.gaperfmodel import GAPerfModel
            return GAPerfModel(model_name=model_name, model_dst=model_dst)
        else:
            raise ValueError("Invalid model type")
