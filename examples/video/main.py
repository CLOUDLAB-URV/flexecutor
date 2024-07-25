from lithops import FunctionExecutor

from functions import (
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
from flexecutor.workflow.executor import DAGExecutor, StageConfig

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("video-analytics")

        config_space = [
            {
                "video_segmentation": StageConfig(cpu=2, memory=2048, workers=1),
                "key_frame_extraction": StageConfig(cpu=2, memory=2048, workers=1),
                "image_enhancement": StageConfig(cpu=2, memory=2048, workers=1),
                "classification": StageConfig(cpu=2, memory=2048, workers=1),
            },
        ]

        data_videos = FlexData("videos")
        data_video_chunks = FlexData("video-chunks", suffix=".mp4")
        data_mainframes = FlexData("mainframes", suffix=".jpg")
        data_filtered_frames = FlexData("filtered-frames", suffix=".jpg")
        data_classification = FlexData("classification", suffix=".json")

        # FIXME: It would be better if the ids used in the chunks were
        # the same across different stages, then we could associate the
        # classification jsons to the original videos to understand what
        # was classified and if it was correctly classified.
        stage0 = Stage(
            stage_id="video_segmentation",
            func=split_videos,
            inputs=[data_videos],
            outputs=[data_video_chunks],
        )
        stage1 = Stage(
            stage_id="key_frame_extraction",
            func=extract_frames,
            inputs=[data_video_chunks],
            outputs=[data_mainframes],
        )
        stage2 = Stage(
            stage_id="image_enhancement",
            func=sharpening_filter,
            inputs=[data_mainframes],
            outputs=[data_filtered_frames],
        )
        stage3 = Stage(
            stage_id="classification",
            func=classify_images,
            inputs=[data_filtered_frames],
            outputs=[data_classification],
        )

        stage0 >> stage1 >> [stage2, stage3]
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])
        executor = DAGExecutor(dag, executor=FunctionExecutor(log_level="INFO"))
        # results = executor.profile(config_space=config_space, num_reps=2)

        executor.train()
        predictions = executor.predict(
            [
                StageConfig(cpu=2, memory=2048, workers=1),
                StageConfig(cpu=2, memory=2048, workers=1),
                StageConfig(cpu=2, memory=2048, workers=1),
                StageConfig(cpu=2, memory=2048, workers=1),
            ]
        )
        print("predictions")
        print(predictions)

    main()
