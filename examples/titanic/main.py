from dataplug.formats.generic.csv import CSV
from dataplug.formats.generic.csv import partition_num_chunks as preprocess_dynamic_csv
from lithops import FunctionExecutor

from examples.titanic.functions import train_model
from flexecutor.storage.chunker import ChunkerTypeEnum, Chunker
from flexecutor.storage.chunking_strategies import preprocess_static_csv
from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

CHUNKER_TYPE = "STATIC"


if __name__ == "__main__":

    if CHUNKER_TYPE == "STATIC":
        chunker = Chunker(
            prefix="titanic",
            chunker_type=ChunkerTypeEnum.STATIC,
            strategy=preprocess_static_csv,
        )
    elif CHUNKER_TYPE == "DYNAMIC":
        chunker = Chunker(
            prefix="titanic",
            chunker_type=ChunkerTypeEnum.DYNAMIC,
            strategy=preprocess_dynamic_csv,
            cloud_object_format=CSV,
        )
    else:
        raise ValueError(f"Chunker type {CHUNKER_TYPE} not supported")

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("titanic")

        stage = Stage(
            stage_id="stage",
            func=train_model,
            inputs=[FlexInput("titanic", chunker=chunker)],
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
