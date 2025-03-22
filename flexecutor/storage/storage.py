import os
from pathlib import Path
from typing import Optional, Tuple

from lithops import Storage

from flexecutor.utils.enums import StrategyEnum, ChunkerTypeEnum


class FlexData:
    def __init__(
        self,
        prefix: str,
        bucket=None,
        custom_data_id=None,
        read_strategy: StrategyEnum = StrategyEnum.SCATTER,
        chunker=None,
        local_base_path: str = "/tmp",
        suffix=".file",
    ):
        """
        A class to define inputs in flex stages.
        ...
        """
        self._input_id = custom_data_id or prefix
        self.bucket = bucket if bucket else os.environ.get("FLEX_BUCKET")
        if prefix[-1] != "/":
            prefix += "/"
        self.prefix = prefix or ""
        self.read_strategy = read_strategy
        self.file_indexes: Optional[Tuple[int, int]] = None
        self.chunker = chunker
        self.local_base_path = Path(local_base_path) / self.prefix
        self.suffix = suffix
        self.keys = []
        self.local_paths = []

    def __repr__(self):
        return (f"FlexData(prefix={self.prefix}, bucket={self.bucket}, strategy={self.read_strategy}, "
                f"chunker={self.chunker}, local_base_path={self.local_base_path}, file_indexes={self.file_indexes})")

    @property
    def id(self):
        return self._input_id

    @id.setter
    def id(self, value):
        self._input_id = value

    def has_chunker_type(self, chunking_type: ChunkerTypeEnum):
        return self.chunker is not None and self.chunker.chunker_type is chunking_type

    def scan_keys(self):
        objects = Storage().list_objects(self.bucket, prefix=self.prefix)
        if self.chunker and self.chunker.chunker_type is ChunkerTypeEnum.DYNAMIC and self.chunker.cloud_object_format.is_folder:
            # get common string between all objects
            common_prefix = os.path.commonprefix([obj["Key"] for obj in objects])
            self.keys = [common_prefix]
        else:
            self.keys = [obj["Key"] for obj in objects if obj["Key"][-1] != "/"]


    def set_local_paths(self, override_local_paths: Optional[list[str]] = None):
        if (
            self.has_chunker_type(ChunkerTypeEnum.DYNAMIC)
            and override_local_paths is None
        ):
            return
        if override_local_paths:
            self.local_paths = override_local_paths
            return
        self.local_paths = [
            str(self.local_base_path / key.split("/")[-1]) for key in self.keys
        ]

    def set_file_indexes(self, worker_id, num_workers):
        if self.has_chunker_type(ChunkerTypeEnum.DYNAMIC):
            self.file_indexes = (0, 1)
            return
        if self.has_chunker_type(ChunkerTypeEnum.STATIC):
            self.file_indexes = (worker_id, worker_id + 1)
            return
        if self.read_strategy == StrategyEnum.BROADCAST:
            start = 0
            end = len(self.local_paths)
        else:  # SCATTER
            num_files = len(self.local_paths)
            start = (worker_id * num_files) // num_workers
            end = ((worker_id + 1) * num_files) // num_workers
        self.file_indexes = (start, end)

    def __hash__(self):
        return hash(self.prefix)

    def flush(self):
        self.prefix = None
        self.keys = []
        self.local_paths = []
        self.file_indexes = None
        self.chunker = None
