from enum import Enum
from typing import Callable, List

from dataplug import CloudObject
from dataplug.entities import CloudObjectSlice


class ChunkerTypeEnum(Enum):
    STATIC = 1
    DYNAMIC = 2


class Chunker:
    def __init__(
        self,
        prefix,
        chunker_type: ChunkerTypeEnum,
        strategy: Callable,
        cloud_object_format=None,
    ):
        """
        The Chunker class is responsible for chunking the data before processing it in the workers.
        @param prefix: the object storage prefix in which the data is stored
        @param chunker_type: STATIC or DYNAMIC.
         Static chunking is used when the data is downloaded, chunked, and then uploaded into smaller parts.
         Dynamic chunking is used when the data is chunked using on-the-fly partitioning via Dataplug.
        @param strategy: the function that will be used to chunk the data.
         If chunker_type is STATIC, the strategy will implement downloading, chunking, and uploading of the data.
         If chunker_type is DYNAMIC, the strategy will be a partitioning function already implemented by Dataplug.
        @param cloud_object_format: the format of the Dataplug cloud object
         Only used when chunker_type is DYNAMIC (default: None)
        """
        if prefix and prefix[-1] != "/":
            prefix += "/"
        self.prefix = prefix
        self.chunker_type: ChunkerTypeEnum = chunker_type
        self.strategy: Callable = strategy
        self.data_slices: List[CloudObjectSlice] = []
        self.cloud_object_format = cloud_object_format

    def preprocess(self, flex_input, num_workers):
        if self.chunker_type == ChunkerTypeEnum.STATIC:
            self.strategy(self.prefix, flex_input, num_workers)
            return None
        else:  # DYNAMIC
            # TODO: un-hardcode this
            file = f"s3://{flex_input.bucket}/titanic/titanic.csv"
            cloud_object = CloudObject.from_s3(
                self.cloud_object_format,
                file,
                s3_config={
                    "region_name": "us-east-1",
                    "endpoint_url": "http://172.17.0.1:9000",
                    "credentials": {
                        "AccessKeyId": "admin",
                        "SecretAccessKey": "password"
                    }
                },
            )
            cloud_object.preprocess()
            self.data_slices = cloud_object.partition(
                self.strategy, num_chunks=num_workers
            )
