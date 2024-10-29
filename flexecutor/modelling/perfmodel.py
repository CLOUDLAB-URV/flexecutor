from abc import abstractmethod, ABC
from enum import Enum
from typing import Dict

from flexecutor.utils.dataclass import FunctionTimes, StageConfig
from flexecutor.utils.utils import get_my_exec_path


class PerfModelEnum(Enum):
    ANALYTIC = "analytic"
    GENETIC = "genetic"
    DISTRIBUTION = "distribution"
    MIXED = "mixed"


class PerfModel(ABC):
    def __init__(self, model_type, stage):
        model_name = stage.stage_unique_id or "default"
        model_dst = get_my_exec_path() + "/models/" + model_name + ".pkl"

        self._stage_name = stage.stage_unique_id
        self._stage_id = stage.stage_id
        self._stage_idx = stage.stage_idx

        self.allow_parallel = stage.max_concurrency > 1
        # self.allow_parallel = True

        self.has_parent = stage.parents is not None and len(stage.parents) > 0
        self.parent_idx = max([int(parent.stage_idx) for parent in stage.parents]) if self.has_parent else None
        self._model_name = model_name
        self._model_dst = model_dst
        self._model_type = model_type
        self._objective_func = None

    @abstractmethod
    def train(self, stage_profile_data: Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_time(self, config: StageConfig) -> FunctionTimes:
        raise NotImplementedError

    # @abstractmethod
    # def optimize(self, config: ConfigBounds) -> StageConfig:
    #     raise NotImplementedError

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

    def init_from_dag(self, dag):
        """
        After declaring the Scheduler, some models need to be initialized with the DAG.
        This method is used to initialize the model with the DAG.
        Override this method to implement the logic.
        @param dag: the DAG object
        """
        pass

    @classmethod
    def instance(cls, model_type: PerfModelEnum, stage):
        model_type = model_type.value
        if model_type == PerfModelEnum.ANALYTIC.value:
            from flexecutor.modelling.anaperfmodel import AnaPerfModel

            return AnaPerfModel(stage)
        elif model_type == PerfModelEnum.GENETIC.value:
            from flexecutor.modelling.gaperfmodel import GAPerfModel

            return GAPerfModel(stage)
        elif model_type == PerfModelEnum.DISTRIBUTION.value:
            from flexecutor.modelling.distperfmodel import DistPerfModel

            return DistPerfModel(stage)

        elif model_type == PerfModelEnum.MIXED.value:
            from flexecutor.modelling.mixedperfmodel import MixedPerfModel

            return MixedPerfModel(stage)
        else:
            raise ValueError("Invalid model type")
