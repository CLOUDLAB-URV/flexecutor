from __future__ import annotations

import logging

from lithops import LocalhostExecutor

from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.utils.dataclass import ResourceConfig
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.workflow.stagefuture import InputFile

config = {'lithops': {'backend': 'localhost', 'storage': 'localhost'}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
BUCKET_NAME = "lithops-manri-urv"


NUM_ITERATIONS = 1

if __name__ == '__main__':
    dag = DAG('mini-dag')

    stage1 = Stage(
        'stage1',
        func=word_occurrence_count,
        input_file=InputFile(f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt")
    )
    stage2 = Stage(
        'stage2',
        func=word_occurrence_count,
        input_file=InputFile(f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt")
    )

    stage2 << stage1

    dag.add_stages([stage1, stage2])

    executor = DAGExecutor(dag, executor=LocalhostExecutor())
    executor.train()

    prediction = executor.predict(ResourceConfig(cpu=2, memory=1024, workers=3), stage1)
    print(prediction)

    executor.shutdown()
    print('stages completed')
