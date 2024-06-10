from __future__ import annotations

import logging

from lithops import LocalhostExecutor

from examples.mini.functions.word_occurrence import word_occurrence_count
from flexecutor.utils.dataclass import StageConfig
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

config = {'lithops': {'backend': 'localhost', 'storage': 'localhost'}}

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s -- %(message)s"
logging.basicConfig(format=LOGGER_FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
BUCKET_NAME = "lithops-manri-urv"


NUM_ITERATIONS = 1

if __name__ == "__main__":
    @flexorchestrator
    def main():
        dag = DAG('mini-dag')

        stage1 = Stage(
            'stage1',
            func=word_occurrence_count,
            # input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
        )
        stage2 = Stage(
            'stage2',
            func=word_occurrence_count,
            # input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
        )
        stage3 = Stage(
            'stage3',
            func=word_occurrence_count,
            # input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
        )

        stage1 >> stage2 << stage3

        dag.add_stages([stage1, stage2, stage3])

        executor = DAGExecutor(dag, executor=LocalhostExecutor())
        executor.train()

        prediction = executor.predict(StageConfig(cpu=2, memory=1024, workers=3), stage1)
        print(prediction)

        executor.shutdown()
        print('stages completed')

    main()
