from lithops import LocalhostExecutor

from examples.mini.functions.word_occurrence import word_occurrence_count
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor, StageConfig
from flexecutor.workflow.stage import Stage

BUCKET_NAME = "lithops-manri-urv"

if __name__ == "__main__":
    @flexorchestrator
    def main():
        config_spaces = [
            StageConfig(0.5, 256, 32),
            # ...
        ]

        dag = DAG('stage-obj-profiling')

        stage1 = Stage(
            'stage1',
            func=word_occurrence_count,
            # input_file=f"/tmp/{BUCKET_NAME}/test-bucket/tiny_shakespeare.txt"
        )

        dag.add_stages([stage1])

        executor = DAGExecutor(dag, executor=LocalhostExecutor())
        executor.profile(config_spaces, num_iterations=1)
        executor.shutdown()

        print('stages completed')

    main()
