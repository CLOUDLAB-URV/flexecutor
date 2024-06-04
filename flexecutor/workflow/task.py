from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Set, List, Optional, Callable

from lithops import FunctionExecutor
from lithops.utils import FuturesList

from flexecutor.future import Future
from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum


class TaskState(Enum):
    """
    State of a task
    """
    NONE = 0
    SCHEDULED = 1
    WAITING = 2
    RUNNING = 3
    SUCCESS = 4
    FAILED = 5


class Task:
    """

    :param task_id: Task ID
    :param executor: Executor to use
    :param input_data: Input data for the operator
    :param args: Arguments to pass to the operator
    :param kwargs: Keyword arguments to pass to the operator
    """

    def __init__(
            self,
            task_id: str,
            func: Callable[[...], Any],
            perf_model_type: PerfModelEnum = PerfModelEnum.ANALYTIC,
            executor: FunctionExecutor | None = None,
            input_data: Optional[Dict[str, Future] | Future] = None,
            output_data: Optional[Dict[str, Future] | Future] = None,
            *args,
            **kwargs
    ):
        self._task_unique_id = None
        self._task_id = task_id
        self._executor = executor
        self._perf_model = None     # Lazy init
        self._perf_model_type = perf_model_type
        self._input_data = input_data if isinstance(input_data, dict)\
            else {'root': input_data} if input_data else dict()
        self._output_data = output_data
        self._args = args
        self._kwargs = kwargs
        self._children: Set[Task] = set()
        self._parents: Set[Task] = set()
        self._state = TaskState.NONE
        self._map_func = func
        self.dag_id = None

    @property
    def executor(self) -> FunctionExecutor:
        """Return the executor."""
        return self._executor

    @executor.setter
    def executor(self, value: FunctionExecutor):
        """Set the executor."""
        self._executor = value

    @property
    def dag_id(self) -> str:
        """Return the DAG ID."""
        return self._dag_id

    @dag_id.setter
    def dag_id(self, value: str):
        self._dag_id = value
        self._task_unique_id = f'{self._dag_id}-{self._task_id}'
        self._perf_model = PerfModel.instance(model_type=self._perf_model_type,
                                              model_name=self._task_unique_id,
                                              model_dst=f"models/{self._dag_id}/{self._task_id}.pkl")

    def __call__(
            self,
            input_data: Dict[str, Future] = None,
            *args,
            **kwargs
    ) -> FuturesList:
        """
        Execute the operator and return a future object.

        :param input_data: Input data
        :return: the future object
        """

        input_data = input_data or self._input_data

        if 'obj' in input_data:
            iterdata = input_data['obj'].data
            self._kwargs['obj_chunk_number'] = self._executor.config['workers']
        else:
            iterdata = [(v, k) for k, v in input_data.items()]

        return self._executor.map(
            self._map_func,
            iterdata,
            *self._args,
            **self._kwargs
        )

    @property
    def perf_model(self) -> PerfModel:
        return self._perf_model

    @property
    def task_id(self) -> str:
        """Return the task ID."""
        return self._task_id

    @property
    def parents(self) -> Set[Task]:
        """Return the parents of this operator."""
        return self._parents

    @property
    def children(self) -> Set[Task]:
        """Return the children of this operator."""
        return self._children

    @property
    def input_data(self) -> Dict[str, Future]:
        """Return the input data."""
        return self._input_data

    @property
    def state(self) -> TaskState:
        """Return the state of the task."""
        return self._state

    @state.setter
    def state(self, value):
        """Set the state of the task."""
        self._state = value

    def _set_relation(self, operator_or_operators: Task | List[Task], upstream: bool = False):
        """
        Set relation between this operator and another operator or list of operator

        :param operator_or_operators: Operator or list of operator
        :param upstream: Whether to set the relation as upstream or downstream
        """
        if isinstance(operator_or_operators, Task):
            operator_or_operators = [operator_or_operators]

        for operator in operator_or_operators:
            if upstream:
                self.parents.add(operator)
                operator.children.add(self)
            else:
                self.children.add(operator)
                operator.parents.add(self)

    def add_parent(self, operator: Task | List[Task]):
        """
        Add a parent to this operator.
        :param operator: Operator or list of operator
        """
        self._set_relation(operator, upstream=True)

    def add_child(self, operator: Task | List[Task]):
        """
        Add a child to this operator.
        :param operator: Operator or list of operator
        """
        self._set_relation(operator, upstream=False)

    def __lshift__(self, other: Task | List[Task]) -> Task | List[Task]:
        """Overload the << operator to add a parent to this operator."""
        self.add_parent(other)
        return other

    def __rshift__(self, other: Task | List[Task]) -> Task | List[Task]:
        """Overload the >> operator to add a child to this operator."""
        self.add_child(other)
        return other

    def __rrshift__(self, other: Task | List[Task]) -> Task:
        """Overload the >> operator for lists of operator. """
        self.add_parent(other)
        return self

    def __rlshift__(self, other: Task | List[Task]) -> Task:
        """Overload the << operator for lists of operator."""
        self.add_child(other)
        return self
