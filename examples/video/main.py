from lithops import FunctionExecutor

from examples.video.functions import (
    split_videos,
    extract_frames,
    sharpening_filter,
    classify_images,
)
from flexecutor.storage.storage import FlexData
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("video-analytics")

        data_videos = FlexData("videos")
        data_video_chunks = FlexData("video-chunks", suffix=".mp4")
        data_mainframes = FlexData("mainframes", suffix=".jpg")
        data_filtered_frames = FlexData("filtered-frames", suffix=".jpg")
        data_classification = FlexData("classification", suffix=".json")

        stage0 = Stage(
            stage_id="stage0",
            func=split_videos,
            inputs=[data_videos],
            outputs=[data_video_chunks],
        )
        stage1 = Stage(
            stage_id="stage1",
            func=extract_frames,
            inputs=[data_video_chunks],
            outputs=[data_mainframes],
        )
        stage2 = Stage(
            stage_id="stage2",
            func=sharpening_filter,
            inputs=[data_mainframes],
            outputs=[data_filtered_frames],
        )
        stage3 = Stage(
            stage_id="stage3",
            func=classify_images,
            inputs=[data_filtered_frames],
            outputs=[data_classification],
        )

        stage0 >> stage1 >> [stage2, stage3]
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])
        executor = DAGExecutor(dag, executor=FunctionExecutor(log_level="INFO"))
        results = executor.execute(num_workers=8)
        print(results["stage1"].get_timings())

    main()
