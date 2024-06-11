import os
import time
from pathlib import Path
from typing import Callable

from lithops import Storage
from flexecutor.utils import setup_logging
from flexecutor.utils.utils import initialize_timings


def measure_operation(op_type: str):
    def decorator(func: Callable):
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "timings"):
                self.timings = initialize_timings()

            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()

            self.timings[op_type] += end_time - start_time
            return result

        return wrapper

    return decorator


class S3File:
    def __init__(self, path: str, local_base_path: str, unique_id: str):
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Path must be in the format 'bucket/key'")
        self.bucket = parts[0]
        self.key = parts[1]
        self.unique_id = unique_id
        self.local_base_path = Path(local_base_path)
        self.local_path = self._calculate_local_path()

    def _calculate_local_path(self):
        local_path = self.local_base_path / self.unique_id / Path(self.key).name
        return local_path

    def __repr__(self):
        return f"{self.bucket}/{self.key} ({self.unique_id})"

    def __str__(self):
        return self.__repr__()


class InputS3File(S3File):
    def __init__(
        self, path: str, local_base_path: str, unique_id: str, *args, **kwargs
    ):
        self.client = Storage()
        if not self._file_exists_in_s3(path):
            raise FileNotFoundError(f"File {path} does not exist in bucket")
        super().__init__(path, local_base_path, unique_id)
        self.logger = setup_logging("INFO")
        self.logger.info(
            f"InputS3File initialized with S3 path {self} and local base path {self.local_base_path}"
        )

    def _file_exists_in_s3(self, path: str):
        bucket, key = path.split("/", 1)
        try:
            self.client.head_object(bucket, key)
            return True
        except Exception:
            return False

    @measure_operation("read")
    def download_file(self):
        self.validate_paths()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading {self.key} to {self.local_path}")
        try:
            self.client.download_file(self.bucket, self.key, str(self.local_path))
            self.logger.info(f"Downloaded {self.key} to {self.local_path}")
        except Exception as e:
            self.logger.error(f"Failed to download {self.key}: {e}")
            raise

    def validate_paths(self):
        if not self.bucket or not self.key:
            raise ValueError("Both bucket and key must be defined in the S3 path.")
        if not self.local_base_path.is_dir():
            raise ValueError("Local base path must be a valid directory.")


class OutputS3File(S3File):
    def __init__(
        self, path: str, local_base_path: str, unique_id: str, *args, **kwargs
    ):
        super().__init__(path, local_base_path, unique_id)
        self.client = Storage()
        self.logger = setup_logging("INFO")
        self.logger.info(
            f"OutputS3File initialized with S3 path {self} and local base path {self.local_base_path}"
        )

    def file_exists_locally(self):
        return self.local_path.exists()

    @measure_operation("write")
    def upload_file(self):
        self.validate_paths()
        if not self.file_exists_locally():
            self.logger.error(f"Local file {self.local_path} does not exist")
            raise FileNotFoundError(f"Local file {self.local_path} does not exist")
        self.logger.info(f"Uploading {self.local_path} to {self.key}")
        try:
            self.client.upload_file(str(self.local_path), self.bucket, self.key)
            self.logger.info(f"Uploaded {self.local_path} to {self.key}")
        except Exception as e:
            self.logger.error(f"Failed to upload {self.local_path}: {e}")
            raise

    def validate_paths(self):
        if not self.bucket or not self.key:
            raise ValueError("Both bucket and key must be defined in the S3 path.")
        if not self.local_base_path.is_dir():
            raise ValueError("Local base path must be a valid directory.")


if __name__ == "__main__":
    try:
        input_file = InputS3File("test-bucket/dir/file.txt", "/tmp", unique_id="step1")
    except FileNotFoundError as e:
        print(e)
