from lithops import FunctionExecutor

from examples.video.functions import (
    split_videos,
    extract_frames,
    sharpening_filter,
    classify_images,
)
from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("video-analytics")

        # TODO: avoid repeating the same bucket name in every input/output

        stage0 = Stage(
            stage_id="stage0",
            func=split_videos,
            inputs=[FlexInput(prefix="videos")],
            outputs=[
                FlexOutput(
                    prefix="video-chunks",
                    suffix=".mp4",
                )
            ],
        )
        stage1 = Stage(
            stage_id="stage1",
            func=extract_frames,
            inputs=[FlexInput(prefix="video-chunks")],
            outputs=[
                FlexOutput(
                    prefix="mainframes",
                    suffix=".jpg",
                )
            ],
        )
        stage2 = Stage(
            stage_id="stage2",
            func=sharpening_filter,
            inputs=[FlexInput(prefix="mainframes")],
            outputs=[
                FlexOutput(
                    prefix="filtered-frames",
                    suffix=".jpg",
                )
            ],
        )
        stage3 = Stage(
            stage_id="stage3",
            func=classify_images,
            inputs=[FlexInput(prefix="filtered-frames")],
            outputs=[
                FlexOutput(
                    prefix="classification",
                    suffix=".json",
                )
            ],
        )

        stage0 >> stage1 >> [stage2, stage3]
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])
        executor = DAGExecutor(dag, executor=FunctionExecutor(log_level="INFO"))
        results = executor.execute()
        print(results["stage1"].get_timings())

    main()
