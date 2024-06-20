from typing import Any

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

    @flexorchestrator
    def main():
        dag = DAG("video-analytics")

        stage0 = Stage(
            stage_id="stage0",
            func=split_videos,
            inputs=[FlexInput("videos", bucket="test-bucket", prefix="videos")],
            outputs=[
                FlexOutput("video-chunks", bucket="test-bucket", prefix="video-chunks", suffix=".mp4")
            ],
        )
        stage1 = Stage(
            stage_id="stage1",
            func=extract_frames,
            inputs=[
                FlexInput("video-chunks", bucket="test-bucket", prefix="video-chunks")
            ],
            outputs=[
                FlexOutput("mainframes", bucket="test-bucket", prefix="mainframes", suffix=".jpg")
            ],
        )
        stage2 = Stage(
            stage_id="stage2",
            func=sharpening_filter,
            inputs=[FlexInput("mainframes", bucket="test-bucket", prefix="mainframes")],
            outputs=[
                FlexOutput("filtered-frames", bucket="test-bucket", prefix="filtered-frames", suffix=".jpg")
            ],
        )
        stage3 = Stage(
            stage_id="stage3",
            func=classify_images,
            inputs=[
                FlexInput("filtered-frames", bucket="test-bucket", prefix="filtered-frames")
            ],
            outputs=[
                FlexOutput("classification", bucket="test-bucket", prefix="classification", suffix=".json")
            ],
        )

        stage0 >> stage1 >> [stage2, stage3]
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])
        executor = DAGExecutor(dag, executor=FunctionExecutor(log_level="INFO"))
        results = executor.execute()
        print(results["stage1"].get_timings())

    main()
