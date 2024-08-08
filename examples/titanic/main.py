from dataplug.formats.generic.csv import CSV
from dataplug.formats.generic.csv import partition_num_chunks as chunking_dynamic_csv
from lithops import FunctionExecutor

from examples.titanic.functions import train_model
from flexecutor.storage.chunker import Chunker
from flexecutor.storage.chunking_strategies import chunking_static_csv
from flexecutor.storage.storage import FlexData
from flexecutor.utils.enums import ChunkerTypeEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

CHUNKER_TYPE = "DYNAMIC"


if __name__ == "__main__":

    if CHUNKER_TYPE == "STATIC":
        chunker = Chunker(
            chunker_type=ChunkerTypeEnum.STATIC,
            chunking_strategy=chunking_static_csv,
        )

    elif CHUNKER_TYPE == "DYNAMIC":
        chunker = Chunker(
            chunker_type=ChunkerTypeEnum.DYNAMIC,
            chunking_strategy=chunking_dynamic_csv,
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
            inputs=[FlexData(prefix="titanic", chunker=chunker)],
            outputs=[
                FlexData(
                    prefix="titanic-accuracy",
                    suffix=".txt",
                )
            ],
        )

        dag.add_stage(stage)
        executor = DAGExecutor(dag, executor=FunctionExecutor())
        results = executor.execute(num_workers=7)
        print(results["stage"].get_timings())

    main()
