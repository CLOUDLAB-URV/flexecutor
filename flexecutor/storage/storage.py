import os
import time
from pathlib import Path
from typing import Callable, Tuple, Any

from lithops import Storage
from flexecutor.utils import setup_logging, initialize_timings


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


class DataSlice:
    def __init__(
        self,
        bucket: str,
        key: str,
        output_bucket: str,
        output_key: str,
        local_base_path: str,
        unique_id: str,
        chunk: Tuple[int, int],
        s3_handler: S3Handler,
    ):
        self.bucket = bucket
        self.key = key
        self.output_bucket = output_bucket
        self.output_key = output_key
        self.local_base_path = Path(local_base_path)
        self.unique_id = unique_id
        self.chunk = chunk
        self.s3_handler = s3_handler
        self.local_input_path = self._calculate_local_path(
            self.bucket, self.key, "input"
        )
        self.local_output_path = self._calculate_local_path(
            self.output_bucket, self.key, "output"
        )
        self.output_key = self.s3_handler.generate_output_key(
            self.key, self.chunk, output_bucket
        )

    def _calculate_local_path(self, bucket, key, path_type):
        local_path = self.local_base_path / self.unique_id / path_type / bucket / key
        return local_path

    def __repr__(self):
        return f"DataSlice(bucket={self.bucket}, key={self.key}, chunk={self.chunk})"


class S3Handler:
    def __init__(self):
        self.client = Storage()

    def download_chunk(self, data_slice: DataSlice):
        byte_range = data_slice.chunk
        extra_get_args = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
        chunk_data = self.client.get_object(
            data_slice.bucket, data_slice.key, extra_get_args=extra_get_args
        )

        data_slice.local_input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_slice.local_input_path, "wb") as f:
            f.write(chunk_data)

    def upload_chunk(self, data_slice: DataSlice):
        self.client.upload_file(
            str(data_slice.local_output_path),
            data_slice.output_bucket,
            data_slice.output_key,
        )

    def generate_output_key(
        self, input_key: str, chunk_range: Tuple[int, int], base_output_path: str
    ) -> str:
        output_key = (
            f"{base_output_path}/{input_key}_chunk_{chunk_range[0]}_{chunk_range[1]}"
        )
        return output_key


class InputS3File:
    def __init__(self, path: str, local_base_path: str, unique_id: str):
        self.client = Storage()
        self.bucket, self.key = self._split_s3_path(path)
        self.unique_id = unique_id
        self.local_base_path = Path(local_base_path)
        self.local_path = self._calculate_local_path()

        if not self._file_exists_in_s3():
            raise FileNotFoundError(f"File {path} does not exist in bucket")

        self.logger = setup_logging("INFO")
        self.logger.info(
            f"InputS3File initialized with S3 path {self.bucket}/{self.key} and local base path {self.local_base_path}"
        )

    def _split_s3_path(self, s3_path: str) -> Tuple[str, str]:
        parts = s3_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Path must be in the format 'bucket/key'")
        return parts[0], parts[1]

    def _file_exists_in_s3(self) -> bool:
        try:
            self.client.head_object(self.bucket, self.key)
            return True
        except Exception:
            return False

    def _calculate_local_path(self):
        return self.local_base_path / self.unique_id / Path(self.key).name

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


class OutputS3Path:
    def __init__(self, base_output_path: str, local_base_path: str, unique_id: str):
        self.base_output_path = base_output_path
        self.local_base_path = Path(local_base_path)
        self.unique_id = unique_id

    def generate_output_key(self, input_key: str, chunk_range: Tuple[int, int]) -> str:
        return f"{self.base_output_path}/{input_key}_chunk_{chunk_range[0]}_{chunk_range[1]}"


if __name__ == "__main__":
    try:
        input_file = InputS3File("test-bucket/dir/file1.txt", "/tmp", unique_id="1")
    except FileNotFoundError as e:
        print(e)
