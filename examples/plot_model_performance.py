from __future__ import annotations

import logging

from lithops import LocalhostExecutor

from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.dataclass import ResourceConfig
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

config = {'lithops': {'backend': 'localhost', 'storage': 'localhost'}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)

NUM_ITERATIONS = 1
BUCKET_NAME = "lithops-manri-urv"


if __name__ == '__main__':
    dag = DAG('mini-dag')

    stage1 = Stage(
        'stage1',
        func=word_occurrence_count,
        perf_model_type=PerfModelEnum.GENETIC,
        input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
    )
    stage2 = Stage(
        'stage2',
        func=word_occurrence_count,
        perf_model_type=PerfModelEnum.GENETIC,
        input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
    )

    stage2 << stage1

    dag.add_stages([stage1, stage2])

    executor = DAGExecutor(dag, executor=LocalhostExecutor())
    executor.train()

    executor.plot_model_performance(stage1, [
        ResourceConfig(cpu=2, memory=1024, workers=3),
        ResourceConfig(cpu=0.5, memory=1568, workers=5),
    ])

    executor.shutdown()
    print('stages completed')
