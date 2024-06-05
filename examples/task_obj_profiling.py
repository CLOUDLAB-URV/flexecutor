from lithops import LocalhostExecutor

from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.workflow.taskfuture import InputFile
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.dagexecutor import DAGExecutor, ConfigSpace
from flexecutor.workflow.task import Task

BUCKET_NAME = "lithops-manri-urv"

if __name__ == "__main__":
    config_spaces = [
        ConfigSpace(0.5, 256, 32),
        # ...
    ]

    dag = DAG('task-obj-profiling')

    task1 = Task(
        'task1',
        func=word_occurrence_count,
        input_file=InputFile(f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt")
    )

    dag.add_tasks([task1])

    executor = DAGExecutor(dag, task_executor=LocalhostExecutor())
    executor.profile(config_spaces, num_iterations=1)
    executor.shutdown()

    print('Tasks completed')
