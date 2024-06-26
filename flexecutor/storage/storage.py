import os
from enum import Enum
from pathlib import Path
from typing import Optional

from lithops import Storage

from flexecutor.storage.chunker import Chunker, ChunkerTypeEnum


class StrategyEnum(Enum):
    SCATTER = 1
    BROADCAST = 2


class FlexInput:
    def __init__(
        self,
        prefix: str,
        bucket=None,
        custom_input_id=None,
        strategy: StrategyEnum = StrategyEnum.SCATTER,
        chunker: Optional[Chunker] = None,
        local_base_path: str = "/tmp",
    ):
        """
        A class for define inputs in flex stages.
        ...
        """
        self._input_id = custom_input_id or prefix
        self.bucket = bucket if bucket else os.environ.get("FLEX_BUCKET")
        if prefix[-1] != "/":
            prefix += "/"
        self.prefix = prefix or ""
        self.strategy = strategy
        self.chunk_indexes: Optional[(int, int)] = None
        self.chunker = chunker
        self.local_base_path = Path(local_base_path) / self.prefix
        self.keys = []
        self.local_paths = []

    @property
    def id(self):
        return self._input_id

    def scan_objects(self, worker_id, num_workers) -> None:
        # Update keys and local_paths
        self.keys = [
            obj["Key"]
            for obj in Storage().list_objects(self.bucket, prefix=self.prefix)
        ]
        self.local_paths = [
            str(self.local_base_path / key.split("/")[-1]) for key in self.keys
        ]
        # Define chunk indexes
        if self.has_chunker_type(ChunkerTypeEnum.DYNAMIC):
            self.chunk_indexes = (0, num_workers)
            return
        if self.has_chunker_type(ChunkerTypeEnum.STATIC):
            self.chunk_indexes = (worker_id, worker_id + 1)
            return
        if self.strategy == StrategyEnum.BROADCAST:
            start = 0
            end = len(self.local_paths)
        else:  # SCATTER
            num_files = len(self.local_paths)
            start = (worker_id * num_files) // num_workers
            end = ((worker_id + 1) * num_files) // num_workers
        self.chunk_indexes = (start, end)

    def has_chunker_type(self, chunking_type: ChunkerTypeEnum):
        return (
                self.chunker is not None
                and self.chunker.chunker_type is chunking_type
        )


class FlexOutput:
    def __init__(
        self,
        prefix,
        bucket=None,
        custom_output_id=None,
        suffix=".file",
        local_base_path="/tmp",
    ):
        self._output_id = custom_output_id or prefix
        self.prefix = prefix
        self.suffix = suffix
        self.local_base_path = Path(local_base_path) / prefix
        self.bucket = bucket if bucket else os.environ.get("FLEX_BUCKET")
        self.keys = []
        self.local_paths = []

    @property
    def id(self):
        return self._output_id
