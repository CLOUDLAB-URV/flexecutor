import json
from pathlib import Path

import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

from flexecutor import StageContext


def split_videos(ctx: StageContext):
    video_paths = ctx.get_input_paths("videos")
    chunk_size = 10

    for index, video_path in enumerate(video_paths):
        vc = VideoFileClip(video_path, verbose=False)
        video_len = int(vc.duration)
        start_size = 0
        while start_size < video_len:
            end_size = min(start_size + chunk_size, video_len)
            chunk_path = f"{ctx.next_output_path('video-chunks')}"
            clip_vc = vc.subclip(start_size, end_size)
            clip_vc.write_videofile(
                chunk_path, codec="libx264", logger=None, ffmpeg_params=["-f", "mp4"]
            )
            del clip_vc
            start_size += chunk_size
        vc.close()


def extract_frames(ctx: StageContext):
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
        video_clip = VideoFileClip(chunk_path, verbose=False)

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
    frame_paths = ctx.get_input_paths("mainframes")
    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(frame_path)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
        cv2.imwrite(ctx.next_output_path("filtered-frames"), sharpened_image)


def classify_images(ctx: StageContext):
    frame_paths = ctx.get_input_paths("filtered-frames")

    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(str(Path(__file__).parent / "tiny-yolov3.pt"))
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
