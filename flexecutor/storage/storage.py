import os
from enum import Enum
from pathlib import Path
from typing import Optional

from lithops import Storage

from flexecutor.storage.chunker import Chunker


class StrategyEnum(Enum):
    SCATTER = 1
    BROADCAST = 2


class FlexInput:
    def __init__(
        self,
        input_id: str,
        key: str = None,
        prefix: str = None,
        bucket=None,
        strategy: StrategyEnum = StrategyEnum.SCATTER,
        chunker: Optional[Chunker] = None,
        local_base_path: str = "/tmp",
    ):
        """
        A class for define inputs in flex stages.
        ...
        """
        self._input_id = input_id
        self.bucket = bucket if bucket else os.environ.get("FLEX_BUCKET")
        if prefix and prefix[-1] != "/":
            prefix += "/"
        self.prefix = prefix or ""
        self.keys = [key] if key else []
        self.strategy = strategy
        self.chunk_indexes: Optional[(int, int)] = None
        self.chunker = chunker
        self.local_base_path = Path(local_base_path) / self.prefix
        self.local_paths = [
            str(self.local_base_path / key.split("/")[-1]) for key in self.keys
        ]

    @property
    def id(self):
        return self._input_id

    def set_chunk_indexes(self, worker_id, num_workers) -> None:
        if self.keys:  # single file input
            self.chunk_indexes = (0, len(self.keys))
            return
        self.keys = [
            obj["Key"]
            for obj in Storage().list_objects(self.bucket, prefix=self.prefix)
        ]
        self.local_paths = [
            str(self.local_base_path / key.split("/")[-1]) for key in self.keys
        ]
        if self.chunker:
            self.chunk_indexes = (0, num_workers)
            self.local_paths = [
                str(self.local_base_path / key.split("/")[-1]) for key in self.keys
            ]
            return
        if self.strategy == StrategyEnum.BROADCAST:
            start = 0
            end = len(self.local_paths)
        else:  # SCATTER
            num_files = len(self.local_paths)
            start = (worker_id * num_files) // num_workers
            end = ((worker_id + 1) * num_files) // num_workers
        self.chunk_indexes = (start, end)


class FlexOutput:
    def __init__(
        self, output_id, prefix, bucket=None, suffix=".file", local_base_path="/tmp"
    ):
        self._output_id = output_id
        self.prefix = prefix
        self.suffix = suffix
        self.local_base_path = Path(local_base_path) / prefix
        self.bucket = bucket if bucket else os.environ.get("FLEX_BUCKET")
        self.keys = []
        self.local_paths = []

    @property
    def id(self):
        return self._output_id
