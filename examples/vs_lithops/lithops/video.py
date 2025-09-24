from lithops import FunctionExecutor

from examples.video.functions import (
    split_videos,
    extract_frames,
    sharpening_filter,
    classify_images,
)
from lithops import Storage
import lithops


import json
import time
import os
import uuid
from pathlib import Path
import boto3

import numpy as np

bucket = "octavio-flexecutor-bucket"
num_workers = 4

storage = lithops.Storage()


def split_videos(keys):
    from moviepy.editor import VideoFileClip

    storage = Storage()

    read_init = time.time()

    paths = []
    os.makedirs("/tmp/videos", exist_ok=True)
    for item in keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

    chunk_size = 10
    output_prefix = "video-chunks/"

    for index, video_path in enumerate(paths):
        vc = VideoFileClip(video_path)
        video_len = int(vc.duration)
        start_size = 0
        while start_size < video_len:
            end_size = min(start_size + chunk_size, video_len)
            # generate uuid[0:8]
            chunk_path = f"{'/tmp/video_' + str(uuid.uuid4())[:8]}.mp4"
            clip_vc = vc.subclip(start_size, end_size)
            clip_vc.write_videofile(
                chunk_path, codec="libx264", logger=None, ffmpeg_params=["-f", "mp4"],
                temp_audiofile='/tmp/temp-audio.mp4'
            )
            storage.upload_file(chunk_path, bucket, output_prefix +
                                f"video_{index}_part_{start_size}_{end_size}_{str(uuid.uuid4())[:8]}.mp4")
            del clip_vc
            start_size += chunk_size
        vc.close()

    compute_end = time.time()
    read = read_end - read_init
    compute = compute_end - read_end

    return (read, compute, None)


def extract_frames(keys):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from PIL import Image

    def calculate_average_pixel_value(image):
        # Convert image to grayscale image
        gray_image = np.mean(image, axis=2).astype(np.uint8)
        # Calculate the average value of pixels
        average_pixel_value = np.mean(gray_image)
        return average_pixel_value

    storage = Storage()

    read_init = time.time()

    paths = []
    os.makedirs("/tmp/video-chunks", exist_ok=True)
    for item in keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

    for index, chunk_path in enumerate(paths):
        best_frame = None
        best_metric = float("-inf")
        video_clip = VideoFileClip(chunk_path)

        for frame in video_clip.iter_frames(fps=0.5, dtype="uint8"):
            frame_metric = calculate_average_pixel_value(frame)
            if frame_metric > best_metric:
                best_metric = frame_metric
                best_frame = frame

        pil_image = Image.fromarray(best_frame)
        uid = str(uuid.uuid4())[:8]
        frame_path = f"/tmp/mainframe_{uid}.jpg"
        pil_image.save(frame_path)
        storage.upload_file(
            frame_path, bucket, "mainframes/" + f"mainframe_{uid}.jpg"
        )
        video_clip.close()

    compute_end = time.time()
    read = read_end - read_init
    compute = compute_end - read_end

    return (read, compute, None)


def sharpening_filter(keys):
    import cv2

    storage = Storage()
    read_init = time.time()

    paths = []
    os.makedirs("/tmp/mainframes", exist_ok=True)
    for item in keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

    for index, frame_path in enumerate(paths):
        image = cv2.imread(frame_path)
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
        uid = str(uuid.uuid4())[:8]
        output_path = f"/tmp/filtered_frame_{uid}.jpg"
        cv2.imwrite(output_path, sharpened_image)
        storage.upload_file(
            output_path, bucket, "filtered-frames/" +
            f"filtered_frame_{uid}.jpg"
        )

    compute_end = time.time()
    read = read_end - read_init
    compute = compute_end - read_end

    return (read, compute, None)


def classify_images(keys):

    from imageai.Detection import ObjectDetection
    storage = Storage()

    read_init = time.time()

    paths = []
    os.makedirs("/tmp/filtered-frames", exist_ok=True)
    for item in keys:
        path = "/tmp/" + item
        paths.append(path)
        storage.download_file(bucket, item, path)

    read_end = time.time()

    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    folder = Path("/function")
    # folder = Path(__file__).parent
    detector.setModelPath(str(folder / "tiny-yolov3.pt"))
    detector.loadModel()

    for index, frame_path in enumerate(paths):
        detection = detector.detectObjectsFromImage(
            input_image=frame_path,
            output_image_path="/tmp/dest_image.jpg",
            minimum_percentage_probability=2,
        )

        json_data = json.dumps(detection, indent=4)
        uid = str(uuid.uuid4())[:8]
        tmp_filename = f"/tmp/classification_{uid}.json"
        with open(tmp_filename, "w") as json_file:
            json_file.write(json_data)
        storage.upload_file(
            tmp_filename, bucket, "classification/" +
            f"classification_{uid}.json"
        )

    compute_end = time.time()
    read = read_end - read_init
    compute = compute_end - read_end

    return (read, compute, None)


def explicit_scatter(prefix):
    # explicit scatter the files
    objects = storage.list_objects(bucket, prefix=prefix)
    keys = [obj["Key"] for obj in objects if obj["Key"][-1] != "/"]

    # split keys in list of length num_workers (no def fucntion)
    iterdata = [keys[i: i + len(keys)//num_workers]
                for i in range(0, len(keys), len(keys)//num_workers)]

    return iterdata

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

    def main():
        for i in range(10):
            t0 = time.time()

            fexec = FunctionExecutor()

            split_iterdata = explicit_scatter("videos/")
            fexec.map(split_videos, split_iterdata)
            fexec.wait()
            profilings = fexec.get_result()

            sharp_iterdata = explicit_scatter("video-chunks/")
            fexec.map(extract_frames, sharp_iterdata)
            fexec.wait()
            profilings = fexec.get_result()

            filter_iterdata = explicit_scatter("mainframes/")
            fexec.map(sharpening_filter, filter_iterdata)
            fexec.wait()
            profilings = fexec.get_result()

            classify_iterdata = explicit_scatter("filtered-frames/")
            fexec.map(classify_images, classify_iterdata)
            fexec.wait()
            profilings = fexec.get_result()

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


    main()
