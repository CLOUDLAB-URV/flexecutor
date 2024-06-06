from __future__ import annotations

import os
from enum import Enum
from typing import Any, Set, List, Optional, Callable

import numpy as np
import pandas as pd
from lithops import FunctionExecutor
from lithops.utils import FuturesList
from matplotlib import pyplot as plt
from pandas import DataFrame

from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum
from flexecutor.utils.dataclass import ResourceConfig, Prediction, ConfigBounds
from flexecutor.utils.utils import load_profiling_results
from flexecutor.workflow.taskfuture import TaskFuture


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
    :param input_file: Input data for the operator
    :param args: Arguments to pass to the operator
    :param kwargs: Keyword arguments to pass to the operator
    """

    def __init__(
            self,
            task_id: str,
            func: Callable[[...], Any],
            perf_model_type: PerfModelEnum = PerfModelEnum.ANALYTIC,
            executor: FunctionExecutor | None = None,
            input_file: Optional[TaskFuture] = None,
            output_file: Optional[TaskFuture] = None,
            *args,
            **kwargs
    ):
        self._task_unique_id = None
        self._task_id = task_id
        self._executor = executor
        self._perf_model = None  # Lazy init
        self._perf_model_type = perf_model_type
        self._input_file = input_file
        self._output_file = output_file
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
            *args,
            **kwargs
    ) -> FuturesList:
        """
        Execute the operator and return a future object.

        :param input_file: Input data
        :return: the future object
        """

        file_key = self._input_file.file
        self._kwargs['obj_chunk_number'] = self._executor.config['workers']

        return self._executor.map(
            map_function=self._map_func,
            map_iterdata=file_key,
            *self._args,
            **self._kwargs
        )

    def predict(self, config_space: ResourceConfig) -> Prediction:
        return self._perf_model.predict(config_space)

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
    def input_file(self) -> TaskFuture:
        """Return the input data."""
        return self._input_file

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

    def optimize(self, config_bounds: ConfigBounds) -> ResourceConfig:
        return self._perf_model.optimize(config_bounds)

    def model_perf_metrics(self, config_spaces: List[ResourceConfig]) -> DataFrame:
        actual_latencies, predicted_latencies = self._prediction_vs_actual(config_spaces)

        actual_latencies = np.array(actual_latencies)
        predicted_latencies = np.array(predicted_latencies)

        data = np.array([
            [config.workers, config.cpu, config.memory, actual, predicted, abs(actual - predicted),
             (actual - predicted) ** 2]
            for config, actual, predicted in zip(config_spaces, actual_latencies, predicted_latencies)
        ])

        df = pd.DataFrame(data, columns=["Workers",
                                         "CPU",
                                         "Memory",
                                         "Actual latency",
                                         "Predicted latency",
                                         "MAE",
                                         "MSE"])

        return df

    def plot_model_performance(self, config_spaces: List[ResourceConfig]):
        actual_latencies, predicted_latencies = self._prediction_vs_actual(config_spaces)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(config_spaces))

        ax.plot(x, predicted_latencies, label="Predicted Latencies", marker="x")

        if any(actual_latencies):
            ax.plot(x, actual_latencies, label="Actual Latencies", marker="o")

        ax.set_xlabel("Configurations")
        ax.set_ylabel("Latency")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i.key) for i in config_spaces], rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()

        folder = f"images/{self.dag_id}"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"images/{self.dag_id}/{self.task_id}.png")

    def _prediction_vs_actual(self, config_spaces: List[ResourceConfig]):
        actual_latencies = []
        predicted_latencies = []
        profiling_data = load_profiling_results(f"profiling/{self.dag_id}/{self.task_id}.json")
        self.perf_model.train(profiling_data)
        for config in config_spaces:
            if config.key in profiling_data:
                executions = profiling_data[config.key]
                total_latencies = [
                    sum(lats)
                    for breaks in zip(
                        executions["read"],
                        executions["compute"],
                        executions["write"],
                        executions["cold_start_time"],
                    )
                    for lats in zip(*breaks)
                ]
                avg_actual_latency = np.mean(total_latencies)
                actual_latencies.append(avg_actual_latency)
            else:
                actual_latencies.append(None)

            predicted_latency = self.perf_model.predict(config).total_time
            predicted_latencies.append(predicted_latency)
        return actual_latencies, predicted_latencies
