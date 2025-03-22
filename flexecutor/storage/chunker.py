import os
from pathlib import Path
from typing import List

from dataplug import CloudObject
from dataplug.entities import CloudObjectSlice
from lithops import Storage

from flexecutor.storage.storage import FlexData
from flexecutor.utils.enums import ChunkerTypeEnum


class Chunker:
    def __init__(
        self, chunker_type: ChunkerTypeEnum, chunking_strategy, cloud_object_format=None
    ):
        """
        The Chunker class is responsible for chunking the data before processing it in the workers.
        @param chunker_type: STATIC or DYNAMIC.
         Static chunking is used when the data is downloaded, chunked, and then uploaded into smaller parts.
         Dynamic chunking is used when the data is chunked using on-the-fly partitioning via Dataplug.
        @param chunking_strategy: the function that will be used to chunk the data.
         If chunker_type is STATIC, the strategy will implement downloading, chunking, and uploading of the data.
         If chunker_type is DYNAMIC, the strategy will be a partitioning function already implemented by Dataplug.
        @param cloud_object_format: the format of the Dataplug cloud object
         Only used when chunker_type is DYNAMIC (default: None)
        """
        # if prefix and prefix[-1] != "/":
        #     prefix += "/"
        # self.prefix = prefix
        self.chunker_type: ChunkerTypeEnum = chunker_type
        self.chunking_strategy = chunking_strategy
        self.data_slices: List[CloudObjectSlice] = []
        self.cloud_object_format = cloud_object_format

    def chunk(self, flex_data, num_workers):
        if self.chunker_type == ChunkerTypeEnum.STATIC:
            self._static_chunking(flex_data, num_workers)
        elif self.chunker_type == ChunkerTypeEnum.DYNAMIC:
            self._dynamic_chunking(flex_data, num_workers)
        else:
            raise ValueError("Invalid chunker type")

    def _static_chunking(self, flex_data, num_workers):
        chunker_ctx = InternalChunkerContext(flex_data, num_workers)
        # Download the files to the local storage
        storage = Storage()
        flex_data = chunker_ctx.flex_data
        os.makedirs(flex_data.local_base_path, exist_ok=True)
        for index in range(len(flex_data.keys)):
            storage.download_file(
                flex_data.bucket,
                flex_data.keys[index],
                flex_data.local_paths[index],
            )

        # Execute the chunker function
        self.chunking_strategy(ChunkerContext(chunker_ctx))

        # Upload the chunked files to the object storage
        for index in range(len(chunker_ctx.output_paths)):
            storage.upload_file(
                chunker_ctx.output_paths[index],
                flex_data.bucket,
                chunker_ctx.output_keys[index],
            )

        # Adapt the flex_data object to the new state
        flex_data.flush()
        flex_data.prefix = chunker_ctx.prefix_output

    def _dynamic_chunking(self, flex_data, num_workers):
        if self.cloud_object_format.is_folder:
            # only first level
            files = [f"s3://{flex_data.bucket}/partitions-nozip/partition_1.ms"]
        else:
            files = [f"s3://{flex_data.bucket}/{file}" for file in flex_data.keys]

        storage = Storage()
        storage_dict = storage.config[storage.config["backend"]]

        # Calculate the number of chunks per file
        num_chunks = int(num_workers / len(files))
        chunk_list = [num_chunks] * len(files)
        for i in range(num_workers % len(files)):
            chunk_list[i] += 1

        # Create the cloud objects and partition them
        for file, num_chunks_file in zip(files, chunk_list):
            #3 preprocessing for 1 file??
            cloud_object = CloudObject.from_s3(
                self.cloud_object_format,
                file,
                s3_config={
                    "region_name": "us-east-1",
                    "endpoint_url": storage_dict["endpoint"],
                    "credentials": {
                        "AccessKeyId": storage_dict["access_key_id"],
                        "SecretAccessKey": storage_dict["secret_access_key"],
                    },
                },
            )
            cloud_object.preprocess()
            self.data_slices.extend(
                cloud_object.partition(self.chunking_strategy, num_chunks=num_chunks_file)
            )


class InternalChunkerContext:
    def __init__(self, flex_data: FlexData, num_workers: int):
        self.flex_data = flex_data
        self.prefix_output = flex_data.prefix.removesuffix("/") + "-chunks"
        self.num_workers = num_workers
        self.output_paths = []
        self.output_keys = []
        self.counter = 0

    def get_input_paths(self):
        return self.flex_data.local_paths

    def next_chunk_path(self):
        new_local_base_path = Path(
            str(self.flex_data.local_base_path).replace(
                self.flex_data.prefix.removesuffix("/"), self.prefix_output
            )
        )
        os.makedirs(new_local_base_path, exist_ok=True)
        local_path = f"{new_local_base_path}/part{self.counter}.{self._get_suffix()}"
        self.output_paths.append(local_path)
        key = f"{self.prefix_output}/part{self.counter}.{self._get_suffix()}"
        self.output_keys.append(key)
        self.counter += 1
        return local_path

    def _get_suffix(self):
        return self.flex_data.keys[0].split(".")[-1]


# ChunkerContext is a facade for InternalChunkerContext
class ChunkerContext:
    def __init__(self, ctx: InternalChunkerContext):
        self._ctx = ctx

    def get_input_paths(self):
        return self._ctx.get_input_paths()

    def next_chunk_path(self):
        return self._ctx.next_chunk_path()

    def get_num_workers(self):
        return self._ctx.num_workers
