from dataplug.formats.generic.csv import partition_num_chunks, CSV
from lithops import FunctionExecutor

from examples.titanic.functions import train_model
from flexecutor.storage.chunker import ChunkerTypeEnum, Chunker
from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("titanic")

        stage = Stage(
            stage_id="stage",
            func=train_model,
            inputs=[
                FlexInput(
                    "titanic",
                    # chunker=Chunker(
                    #     prefix="titanic",
                    #     chunker_type=ChunkerTypeEnum.STATIC,
                    #     strategy=preprocess_static_csv,
                    # ),
                    chunker=Chunker(
                        prefix="titanic",
                        chunker_type=ChunkerTypeEnum.DYNAMIC,
                        strategy=partition_num_chunks,
                        cloud_object_format=CSV,
                    ),
                )
            ],
            outputs=[
                FlexOutput(
                    prefix="titanic-accuracy",
                    suffix=".txt",
                )
            ],
        )

        dag.add_stage(stage)
        executor = DAGExecutor(dag, executor=FunctionExecutor())
        results = executor.execute()
        print(results["stage"].get_timings())

    main()
