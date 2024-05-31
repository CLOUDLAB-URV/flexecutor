from botocore.client import Config
from s3path import S3Path, register_configuration_parameter
from pathlib import Path
import boto3
import os
from flexecutor.utils import setup_logging
from lithops import Storage
from typing import Callable
import time

from flexecutor.utils.utils import initialize_timings


# Helper functions to retrieve and reset timings
def get_timings(timings: dict):
    return timings


def reset_timings(timings: dict):
    for key in timings:
        timings[key] = 0


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


class BaseS3Path:
    _config = None
    _resource = None
    logger = setup_logging(log_level="INFO")

    @classmethod
    def configure(cls, config):
        if cls._config is None or cls._resource is None:
            backend = config["backend"]
            cls._config = config[backend]
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

    def __init__(self, path, local_base_path, *args, **kwargs):
        if BaseS3Path._config is None or BaseS3Path._resource is None:
            storage = Storage()
            BaseS3Path.configure(storage.config)

        if isinstance(path, S3Path):
            self.s3_path = path
        else:
            self.s3_path = S3Path(f"/{path.strip('/')}")

        self.local_base_path = Path(local_base_path).resolve()
        register_configuration_parameter(self.s3_path, resource=BaseS3Path._resource)
        self.logger = setup_logging()
        self.logger.info(
            f"Initialized BaseS3Path with S3 path: {self.s3_path}, local base path: {self.local_base_path}"
        )


class InputS3Path(BaseS3Path):
    def __init__(self, path, local_base_path, *args, **kwargs):
        super().__init__(path, local_base_path)
        self.logger.info(
            f"InputS3Path initialized with S3 path {self.s3_path} and local base path {self.local_base_path}"
        )

    def s3_to_local_path(self):
        bucket_name = self.s3_path.parts[1]
        relative_path = self.s3_path.relative_to(f"/{bucket_name}")
        local_path = self.local_base_path / bucket_name / relative_path
        return local_path

    @measure_operation("read")
    def download_directory(self):
        bucket = BaseS3Path._resource.Bucket(self.s3_path.parts[1])
        prefix = str(self.s3_path).lstrip("/")
        for obj in bucket.objects.filter(Prefix=prefix):
            s3_key = obj.key
            self.s3_path = S3Path(f"/{obj.bucket_name}/{s3_key}")
            local_file_path = self.s3_to_local_path()
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Downloading {s3_key} to {local_file_path}")
            try:
                bucket.download_file(s3_key, str(local_file_path))
                self.logger.info(f"Downloaded {s3_key} to {local_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to download {s3_key}: {e}")
                raise


class OutputS3Path(BaseS3Path):
    def __init__(self, path, local_base_path, *args, **kwargs):
        super().__init__(path, local_base_path)
        self.logger.info(
            f"OutputS3Path initialized with S3 path {self.s3_path} and local base path {self.local_base_path}"
        )

    def local_to_s3_key(self):
        bucket_name = self.s3_path.parts[1]
        relative_path = self.local_base_path.relative_to(
            self.local_base_path / bucket_name
        ).as_posix()
        s3_key = str(self.s3_path).lstrip("/") + "/" + relative_path
        return s3_key

    @measure_operation("write")
    def upload_directory(self):
        bucket = BaseS3Path._resource.Bucket(self.s3_path.parts[1])
        for root, _, files in os.walk(self.local_base_path):
            for file in files:
                local_file_path = Path(root) / file
                self.local_base_path = local_file_path
                s3_key = self.local_to_s3_key()
                self.logger.info(f"Uploading {local_file_path} to {s3_key}")
                try:
                    bucket.upload_file(str(local_file_path), s3_key)
                    self.logger.info(f"Uploaded {local_file_path} to {s3_key}")
                except Exception as e:
                    self.logger.error(f"Failed to upload {local_file_path}: {e}")
                    raise
