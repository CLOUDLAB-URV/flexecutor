from __future__ import annotations

import logging

from lithops import LocalhostExecutor

from examples.functions.sleepy_func import sleepy_func
from flexecutor.future import InputData
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.dagexecutor import DAGExecutor, ConfigSpace
from flexecutor.workflow.task import Task

config = {'lithops': {'backend': 'localhost', 'storage': 'localhost'}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)

NUM_ITERATIONS = 1


if __name__ == '__main__':
    config_spaces = [
        ConfigSpace(2, 1024, 3),
        ConfigSpace(0.5, 1568, 5),
        # ...
    ]

    dag = DAG('mini-dag')

    task1 = Task(
        'task1',
        func=sleepy_func,
        input_data={'1': InputData(1), '2': InputData(2), '3': InputData(3)}
    )
    task2 = Task(
        'task2',
        func=sleepy_func,
        input_data={'1': InputData(4), '2': InputData(5), '3': InputData(6)}
    )

    task2 << task1

    dag.add_tasks([task1, task2])

    executor = DAGExecutor(dag, task_executor=LocalhostExecutor())
    executor.profile(config_spaces, num_iterations=NUM_ITERATIONS)
    executor.shutdown()
    print('Tasks completed')


