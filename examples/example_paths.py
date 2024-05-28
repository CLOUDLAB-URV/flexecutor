import boto3
from botocore.client import Config
from s3path import S3Path, register_configuration_parameter
from pathlib import Path
from lithops import Storage
import logging


def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    return logger


class BaseS3Path:
    _config = None
    _resource = None
    logger = setup_logging()

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

        self.s3_path = S3Path(
            f"/{path.strip('/')}"
        )  # Ensure S3 path is properly formatted
        self.local_base_path = Path(
            local_base_path
        ).resolve()  # Ensure local path is absolute
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

    def s3_to_local_path(self, s3_path):
        relative_path = s3_path.relative_to(self.s3_path)
        local_path = self.local_base_path / relative_path.as_posix()
        return local_path


class OutputS3Path(BaseS3Path):
    def __init__(self, path, local_base_path, *args, **kwargs):
        super().__init__(path, local_base_path)
        self.logger.info(
            f"OutputS3Path initialized with S3 path {self.s3_path} and local base path {self.local_base_path}"
        )

    def local_to_s3_path(self, local_path):
        relative_path = local_path.relative_to(self.local_base_path).as_posix()
        s3_path = self.s3_path / relative_path
        return s3_path


if __name__ == "__main__":
    local_base_path = "/tmp"
    input_s3_prefix = "/test-bucket/dir/"

    # Initialize Input and Output S3 paths
    input_s3_path = InputS3Path(input_s3_prefix, local_base_path)

    for e in input_s3_path.s3_path.iterdir():
        local_path = input_s3_path.s3_to_local_path(e)
        print(f"Local path: {local_path}")

    
