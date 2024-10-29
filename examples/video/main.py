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
from scheduling.jolteon import Jolteon
from utils.dataclass import StageConfig

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("video")

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

        entry_point = [
            StageConfig(workers=workers, cpu=cpu)
            for workers, cpu in zip([16, 8, 8, 8], [2] * 4)
        ]
        x_bounds = [
            StageConfig(workers=workers, cpu=cpu)
            for workers, cpu in zip([4, 1] * 4, [32, 5.1] * 4)
        ]
        executor = DAGExecutor(
            dag,
            executor=FunctionExecutor(log_level="INFO"),
            scheduler=Jolteon(
                dag,
                bound=20,
                bound_type="latency",
                entry_point=entry_point,
                x_bounds=x_bounds,
            ),
        )

        executor.optimize()
        executor.shutdown()

    main()
