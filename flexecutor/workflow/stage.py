from __future__ import annotations

from enum import Enum
from typing import Any, Set, List, Optional, Callable

from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum
from flexecutor.storage.storage import FlexInput
from flexecutor.storage.storage import FlexOutput
from flexecutor.utils.dataclass import StageConfig


class StageState(Enum):
    """
    State of a stage
    """

    NONE = 0
    SCHEDULED = 1
    WAITING = 2
    RUNNING = 3
    SUCCESS = 4
    FAILED = 5


class Stage:
    """
    :param stage_id: Stage ID
    :param inputs: List of InputS3Path instances for the operator
    """

    def __init__(
        self,
        stage_id: str,
        func: Callable[..., Any],
        inputs: List[FlexInput],
        outputs: List[FlexOutput],
        perf_model_type: PerfModelEnum = PerfModelEnum.ANALYTIC,
        params: Optional[dict[str, Any]] = None,
        max_concurrency: int = 1024,
    ):
        if params is None:
            params = {}
        self._stage_unique_id = None
        self._stage_id = stage_id
        self._perf_model = None  # Lazy init
        self._perf_model_type = perf_model_type
        self._inputs = inputs
        self._outputs = outputs
        self._params = params
        self._children: Set[Stage] = set()
        self._parents: Set[Stage] = set()
        self._state = StageState.NONE
        self._map_func = func
        self._max_concurrency = max_concurrency
        self.dag_id = None
        self.optimal_config: Optional[StageConfig] = StageConfig(
            cpu=1, memory=1024, workers=1
        )
        self.resource_config: Optional[StageConfig] = StageConfig(
            cpu=1, memory=2048, workers=1
        )
        ############################################
        # FIXME: Encapsulate the parameters in a class, since each scheduler we implement has it's own
        # Ditto parameters
        self._time_weight = None
        self._ratio = None
        self._parallelism_ratio = None
        # self.degree_alloc is our optimal_config
        ############################################

    def __repr__(self) -> str:
        return f"Stage({self._stage_id}, resource_config={self.resource_config}) "

    @property
    def dag_id(self) -> str:
        """Return the DAG ID."""
        return self._dag_id

    @property
    def time_weight(self) -> int:
        "return the time weight of the stage"
        return self._time_weight

    @property
    def ratio(self) -> int:
        "return the ratio of the stage"
        return self._ratio

    @property
    def parallelism_ratio(self) -> int:
        "return the parallelism ratio of the stage"
        return self._parallelism_ratio

    @ratio.setter
    def ratio(self, value: int):
        self._ratio = value

    @parallelism_ratio.setter
    def parallelism_ratio(self, value: int):
        self._parallelism_ratio = value

    @time_weight.setter
    def time_weight(self, value: int):
        self._time_weight = value

    @property
    def map_func(self) -> Callable[..., Any]:
        """Return the map function."""
        return self._map_func

    @dag_id.setter
    def dag_id(self, value: str):
        self._dag_id = value
        self._stage_unique_id = f"{self._dag_id}-{self._stage_id}"
        self._perf_model = PerfModel.instance(
            model_type=self._perf_model_type, model_name=self._stage_unique_id
        )

    @property
    def perf_model(self) -> PerfModel:
        return self._perf_model

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def stage_id(self) -> str:
        """Return the stage ID."""
        return self._stage_id

    @property
    def parents(self) -> Set[Stage]:
        """Return the parents of this operator."""
        return self._parents

    @property
    def children(self) -> Set[Stage]:
        """Return the children of this operator."""
        return self._children

    @property
    def inputs(self) -> List[FlexInput]:
        """Return the list of input paths."""
        return self._inputs

    @property
    def outputs(self) -> List[FlexOutput]:
        """Return the output path."""
        return self._outputs

    @property
    def params(self) -> dict[str, Any]:
        """Return the parameters of the stage."""
        return self._params

    @property
    def state(self) -> StageState:
        """Return the state of the stage."""
        return self._state

    @state.setter
    def state(self, value):
        """Set the state of the stage."""
        self._state = value

    def _set_relation(
        self, operator_or_operators: Stage | List[Stage], upstream: bool = False
    ):
        """
        Set relation between this operator and another operator or list of operator

        :param operator_or_operators: Operator or list of operator
        :param upstream: Whether to set the relation as upstream or downstream
        """
        if isinstance(operator_or_operators, Stage):
            operator_or_operators = [operator_or_operators]

        for operator in operator_or_operators:
            if upstream:
                self.parents.add(operator)
                operator.children.add(self)
            else:
                self.children.add(operator)
                operator.parents.add(self)

    def add_parent(self, operator: Stage | List[Stage]):
        """
        Add a parent to this operator.
        :param operator: Operator or list of operator
        """
        self._set_relation(operator, upstream=True)

    def add_child(self, operator: Stage | List[Stage]):
        """
        Add a child to this operator.
        :param operator: Operator or list of operator
        """
        self._set_relation(operator, upstream=False)

    def __lshift__(self, other: Stage | List[Stage]) -> Stage | List[Stage]:
        """Overload the << operator to add a parent to this operator."""
        self.add_parent(other)
        return other

    def __rshift__(self, other: Stage | List[Stage]) -> Stage | List[Stage]:
        """Overload the >> operator to add a child to this operator."""
        self.add_child(other)
        return other

    def __rrshift__(self, other: Stage | List[Stage]) -> Stage:
        """Overload the >> operator for lists of operator."""
        self.add_parent(other)
        return self

    def __rlshift__(self, other: Stage | List[Stage]) -> Stage:
        """Overload the << operator for lists of operator."""
        self.add_child(other)
        return self
