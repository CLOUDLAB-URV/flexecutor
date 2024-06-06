from botocore.client import Config
from s3path import S3Path, register_configuration_parameter
from pathlib import Path
import boto3
import fnmatch
from flexecutor.utils import setup_logging
from lithops import Storage
from typing import Callable
import time
import re
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


class S3Client:
    _config = None
    _resource = None
    logger = setup_logging(level="INFO")

    @classmethod
    def configure(cls, config):
        if cls._config is None or cls._resource is None:
            backend = config["backend"]
            cls._config = config[backend]
            cls.logger.info(f"Configuration: {cls._config}")
            cls._resource = cls.create_s3_resource(cls._config)
            cls.logger.info(f"Configured {backend} backend")

    @classmethod
    def create_s3_resource(cls, config):
        cls.logger.info(f"Creating S3 resource with config: {config}")
        resource = boto3.resource(
            "s3",
            endpoint_url=config.get("endpoint", None),
            aws_access_key_id=config["access_key_id"],
            aws_secret_access_key=config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )
        cls.logger.info("S3 resource created successfully")
        return resource

    def __init__(self):
        if S3Client._config is None or S3Client._resource is None:
            storage = Storage()
            S3Client.configure(storage.config)

        self.resource = S3Client._resource
        self.logger = S3Client.logger


class Dataset:
    def __init__(
        self, bucket, pattern, local_base_path="/tmp", s3_client=None, *args, **kwargs
    ):
        self.s3_client = s3_client or S3Client()
        self.bucket_name = bucket
        self.pattern = pattern
        self.regex = re.compile(fnmatch.translate(self.pattern))
        self.local_base_path = Path(local_base_path).resolve()
        self.paths = self.find_paths()
        self.logger = setup_logging(level="INFO")
        self.logger.info(
            f"Dataset initialized with pattern: {self.pattern} and local base path: {self.local_base_path}"
        )

    def find_paths(self):
        matching_paths = []
        paginator = self.s3_client.resource.meta.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name):
            for obj in page.get("Contents", []):
                s3_full_path = f"/{self.bucket_name}/{obj['Key']}"
                self.logger.debug(f"Checking S3 path: {s3_full_path}")
                if self.regex.match(s3_full_path):
                    self.logger.debug(f"Matched S3 path: {s3_full_path}")
                    matching_paths.append(S3Path(s3_full_path))
        if not matching_paths:
            self.logger.info("No matching files found.")
        return matching_paths

    def download_all(self):
        for s3_path in self.paths:
            local_file_path = self.s3_to_local_path(s3_path)
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Downloading {s3_path} to {local_file_path}")
            try:
                self.s3_client.resource.Bucket(s3_path.bucket_name).download_file(
                    s3_path.key, str(local_file_path)
                )
                self.logger.info(f"Downloaded {s3_path} to {local_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to download {s3_path}: {e}")
                raise

    def s3_to_local_path(self, s3_path):
        bucket_name = s3_path.parts[1]
        relative_path = s3_path.relative_to(f"/{bucket_name}")
        return self.local_base_path / bucket_name / relative_path


if __name__ == "__main__":

    bucket = "test-bucket"
    pattern = "*_file.txt"

    dataset = Dataset(bucket=bucket, pattern=pattern)
    print("Matched files:", [str(p) for p in dataset.paths])
    dataset.download_all()
