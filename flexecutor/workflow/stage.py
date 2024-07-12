from __future__ import annotations

from enum import Enum
from typing import Any, Set, List, Optional, Callable
from copy import deepcopy
from lithops import FunctionExecutor

from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum
from flexecutor.storage.storage import FlexInput
from flexecutor.storage.storage import FlexOutput
from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.storage.wrapper import worker_wrapper
from flexecutor.workflow.stagecontext import InternalStageContext


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
    Represents a stage in a data processing or computational workflow.

    :param stage_id: Unique identifier for the stage.
    :param func: The function to be executed in this stage.
    :param inputs: List of inputs for the function.
    :param outputs: List of outputs for the function.
    :param executor: Optional executor to run the stage's function. Defaults to None if not provided.
    :param perf_model_type: Performance model type for the stage.
    :param params: Additional parameters for stage configuration, defaults to an empty dictionary if None is provided.
    :param max_concurrency: Maximum number of concurrent executions allowed for this stage.
    """

    def __init__(
        self,
        stage_id: str,
        func: Callable[..., Any],
        inputs: List[FlexInput],
        outputs: List[FlexOutput],
        executor: Optional[FunctionExecutor] = None,
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
        self._max_concurrency: int = max_concurrency
        self._executor = executor
        self.dag_id = None
        # FIXME: resource config is a fallback for now, in the future it would be better if it was provided explicitly, so the user understands what is happening and why its creating X workers (if no optimize is executed)
        self._resource_config: StageConfig = StageConfig(cpu=1, memory=2048, workers=1)

    def __repr__(self) -> str:
        return f"Stage({self._stage_id}, resource_config={self.resource_config}), executor={self._executor}) "

    @property
    def executor(self) -> FunctionExecutor:
        return self._executor

    @executor.setter
    def executor(self, fexec: FunctionExecutor):
        self._executor = fexec

    @property
    def dag_id(self) -> str:
        """Return the DAG ID."""
        return self._dag_id

    @property
    def resource_config(self):
        return self._resource_config

    @resource_config.setter
    def resource_config(self, value: StageConfig):
        self._resource_config = value

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

    def execute(
        self, on_future_done: Callable[[Stage, StageFuture], None] = None
    ) -> StageFuture:
        """
        Process a stage

        :param on_future_done: Callback to execute every time a future is done
        """

        # STATIC PARTITIONING ???
        # for input_path in stage.input_file:
        #     if input_path.partitioner:
        #         input_path.partitioner.partitionize()

        if self._executor is None:
            raise ValueError("No executor provided for the stage.")

        map_iterdata = []
        num_workers = min(self.resource_config.workers, self.max_concurrency)
        for worker_id in range(num_workers):
            copy_inputs = [deepcopy(item) for item in self.inputs]
            copy_outputs = [deepcopy(item) for item in self.outputs]
            for input_item in copy_inputs:
                input_item.scan_objects(worker_id, num_workers)
            ctx = InternalStageContext(
                worker_id, num_workers, copy_inputs, copy_outputs, self.params
            )
            map_iterdata.append(ctx)

        future = self._executor.map(
            map_function=worker_wrapper(self.map_func),
            map_iterdata=map_iterdata,
            runtime_memory=int(self.resource_config.memory),
        )

        self._executor.wait(future)
        future = StageFuture(self.stage_id, future)

        # Update the state of the stage based on the future result
        self.state = StageState.FAILED if future.error() else StageState.SUCCESS

        # Call the callback function if provided
        if on_future_done:
            on_future_done(self, future)

        return future

    def profile(self, stage_config: StageConfig):
        self.resource_config = stage_config
        future = self.execute()
        print(future.get_timings())
        # save the results here

    def train(self, profiling_data):
        pass

    def predict(self):
        pass

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
