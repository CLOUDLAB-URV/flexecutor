from __future__ import annotations

import logging

from lithops import LocalhostExecutor

from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.utils.dataclass import ConfigSpace
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.dagexecutor import DAGExecutor
from flexecutor.workflow.task import Task
from flexecutor.workflow.taskfuture import InputFile

config = {'lithops': {'backend': 'localhost', 'storage': 'localhost'}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
BUCKET_NAME = "lithops-manri-urv"


NUM_ITERATIONS = 1

if __name__ == '__main__':
    dag = DAG('mini-dag')

    task1 = Task(
        'task1',
        func=word_occurrence_count,
        input_file=InputFile(f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt")
    )
    task2 = Task(
        'task2',
        func=word_occurrence_count,
        input_file=InputFile(f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt")
    )

    task2 << task1

    dag.add_tasks([task1, task2])

    executor = DAGExecutor(dag, task_executor=LocalhostExecutor())
    executor.train()

    prediction = task1.predict(ConfigSpace(cpu=2, memory=1024, workers=3))
    print(prediction)

    executor.shutdown()
    print('Tasks completed')
