import boto3
import fnmatch
import time

from botocore.client import Config
from s3path import S3Path
from pathlib import Path
from lithops import Storage
from typing import Callable

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


class S3Client:
    _config = None
    _resource = None
    # class logger
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
        self, s3_client, bucket, pattern, local_base_path="/tmp", *args, **kwargs
    ):
        self.s3_client = s3_client
        self.bucket_name = bucket
        self.pattern = pattern.strip("/")
        self.local_base_path = Path(local_base_path).resolve()
        self.logger = setup_logging(level="INFO")
        self.paths = self.find_paths()
        self.logger.info(
            f"Dataset initialized with pattern: {self.pattern} and local base path: {self.local_base_path}"
        )

    @classmethod
    def from_directory(cls, bucket, directory, local_base_path="/tmp"):
        pattern = f"{directory.strip('/')}/*"
        s3_client = S3Client()
        return cls(s3_client, bucket, pattern, local_base_path)

    @classmethod
    def from_glob(cls, bucket, glob_pattern, local_base_path="/tmp"):
        s3_client = S3Client()
        return cls(s3_client, bucket, glob_pattern, local_base_path)

    def find_paths(self):
        matching_paths = set()
        paginator = self.s3_client.resource.meta.client.get_paginator("list_objects_v2")
        prefix = self.pattern.split("*")[0]
        self.logger.info(
            f"Listing objects in bucket: {self.bucket_name} with prefix: {prefix}"
        )

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_full_path = f"/{self.bucket_name}/{obj['Key']}"
                s3_path_obj = S3Path(
                    s3_full_path
                )  # wrapping the found paths as s3paths
                self.logger.debug(f"Found S3 object: {s3_path_obj}")

                if fnmatch.fnmatch(obj["Key"], self.pattern):
                    self.logger.debug(f"Matched S3 path: {s3_path_obj}")
                    matching_paths.add(s3_path_obj)
                else:
                    self.logger.debug(f"Did not match S3 path: {s3_path_obj}")

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

    dataset_glob = Dataset.from_glob(bucket, "dir/*.txt")
    print("Matched files from glob pattern:", [p for p in dataset_glob.paths])
