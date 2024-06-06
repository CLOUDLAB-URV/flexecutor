from lithops import LocalhostExecutor

from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor, ResourceConfig
from flexecutor.workflow.stage import Stage

BUCKET_NAME = "lithops-manri-urv"

if __name__ == "__main__":
    config_spaces = [
        ResourceConfig(0.5, 256, 32),
        # ...
    ]

    dag = DAG('stage-obj-profiling')

    stage1 = Stage(
        'stage1',
        func=word_occurrence_count,
        input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
    )

    dag.add_stages([stage1])

    executor = DAGExecutor(dag, executor=LocalhostExecutor())
    executor.profile(config_spaces, num_iterations=1)
    executor.shutdown()

    print('stages completed')
