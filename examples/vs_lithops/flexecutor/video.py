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


import json
from pathlib import Path

import numpy as np

from flexecutor import StageContext


def split_videos(ctx: StageContext):
    from moviepy.editor import VideoFileClip

    video_paths = ctx.get_input_paths("videos")
    chunk_size = 10

    for index, video_path in enumerate(video_paths):
        vc = VideoFileClip(video_path)
        video_len = int(vc.duration)
        start_size = 0
        while start_size < video_len:
            end_size = min(start_size + chunk_size, video_len)
            chunk_path = f"{ctx.next_output_path('video-chunks')}"
            clip_vc = vc.subclip(start_size, end_size)
            clip_vc.write_videofile(
                chunk_path, codec="libx264", logger=None, ffmpeg_params=["-f", "mp4"],
                temp_audiofile='/tmp/temp-audio.mp4'
            )
            del clip_vc
            start_size += chunk_size
        vc.close()


def extract_frames(ctx: StageContext):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from PIL import Image

    def calculate_average_pixel_value(image):
        # Convert image to grayscale image
        gray_image = np.mean(image, axis=2).astype(np.uint8)
        # Calculate the average value of pixels
        average_pixel_value = np.mean(gray_image)
        return average_pixel_value

    chunk_paths = ctx.get_input_paths("video-chunks")

    for index, chunk_path in enumerate(chunk_paths):
        best_frame = None
        best_metric = float("-inf")
        video_clip = VideoFileClip(chunk_path)

        for frame in video_clip.iter_frames(fps=0.5, dtype="uint8"):
            frame_metric = calculate_average_pixel_value(frame)
            if frame_metric > best_metric:
                best_metric = frame_metric
                best_frame = frame

        pil_image = Image.fromarray(best_frame)
        frame_path = ctx.next_output_path("mainframes")
        pil_image.save(frame_path)
        video_clip.close()


def sharpening_filter(ctx: StageContext):
    import cv2

    frame_paths = ctx.get_input_paths("mainframes")
    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(frame_path)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
        cv2.imwrite(ctx.next_output_path("filtered-frames"), sharpened_image)


def classify_images(ctx: StageContext):
    from imageai.Detection import ObjectDetection

    frame_paths = ctx.get_input_paths("filtered-frames")

    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    folder = Path("/function")
    # folder = Path(__file__).parent
    detector.setModelPath(str(folder / "tiny-yolov3.pt"))
    detector.loadModel()

    for index, frame_path in enumerate(frame_paths):
        detection = detector.detectObjectsFromImage(
            input_image=frame_path,
            output_image_path="/tmp/dest_image.jpg",
            minimum_percentage_probability=2,
        )

        json_data = json.dumps(detection, indent=4)
        tmp_filename = ctx.next_output_path("classification")
        with open(tmp_filename, "w") as json_file:
            json_file.write(json_data)

import time
import boto3

def clean():
    s3 = boto3.client('s3')
    bucket_name = 'octavio-flexecutor-bucket'
    prefixes = ['video-chunks/', 'mainframes/', 'filtered-frames/', 'classification/']

    for prefix in prefixes:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})

    print("Clean up (video) completed.")


if __name__ == "__main__":

    @flexorchestrator(bucket="octavio-flexecutor-bucket")
    def main():
        for i in range(10):
            t0 = time.time()
            fexec = FunctionExecutor(log_level="DEBUG")

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
                executor=fexec,
                scheduler=Jolteon(
                    dag,
                    bound=20,
                    bound_type="latency",
                    entry_point=entry_point,
                    x_bounds=x_bounds,
                ),
            )

            executor.execute(num_workers=4)

            t1 = time.time()
            stats = [f.stats for f in fexec.futures]
            # add to each stat the total time
            for stat in stats:
                stat['total_time'] = t1 - t0
                stat['init_timestamp'] = t0
                stat['end_timestamp'] = t1
            # Save stats to a JSON file
            with open(f'run_{i}.json', 'w') as f:
                json.dump(stats, f, indent=4)
            clean()

            del fexec
            del dag



    main()
