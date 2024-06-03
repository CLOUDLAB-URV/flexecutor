from __future__ import annotations

import logging
from typing import Dict

from flexecutor.future import Future, InputData
from lithops import Storage, LocalhostExecutor

from flexecutor.operator import Operator
from flexecutor.workflow import DAG
from flexecutor.scheduling import Scheduler

config = {'lithops': {'backend': 'localhost', 'storage': 'localhost'}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)


def map_func(input_data: Future, *args, **kwargs):
    logger.info(f'Executing map_func with input data: {input_data}')
    map_result = input_data.result()
    if isinstance(map_result, int):
        return map_result + 1
    else:
        return sum(map_result) + 1
    # return input_data.result() + 1, input_data.result() + 2


if __name__ == '__main__':
    dag = DAG('dag')

    ex = LocalhostExecutor()
    storage = Storage()
    task1 = Operator(
        'task1',
        executor=ex,
        func=map_func,
        input_data=InputData(1)
    )
    task2 = Operator(
        'task2',
        executor=ex,
        func=map_func,
    )
    task3 = Operator(
        'task3',
        executor=ex,
        func=map_func,
    )
    task4 = Operator(
        'task4',
        executor=ex,
        func=map_func,
    )
    task5 = Operator(
        'task5',
        executor=ex,
        func=map_func,
    )
    task6 = Operator(
        'task6',
        executor=ex,
        func=map_func,
    )

    task1 >> task2 >> [task3, task4] >> task5 >> task6

    dag.add_tasks([task1, task2, task3, task4, task5, task6])

    executor = Scheduler(dag)
    futures = executor.execute()
    executor.shutdown()

    result = futures['task2'].result()
    print('Tasks completed')
    print(result)


