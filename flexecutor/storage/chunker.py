from abc import ABC
from dataclasses import dataclass
from enum import Enum

from lithops import Storage


class ChunkerTypeEnum(Enum):
    STATIC = 1
    DYNAMIC = 2


@dataclass
class ChunkerInfo:
    key: str
    start: int
    end: int


class Chunker(ABC):
    def __init__(self, name, chunker_type=ChunkerTypeEnum.DYNAMIC):
        self.name = name
        self.chunker_type = chunker_type

    def my_byte_range(
        self, flex_input, worker_id, num_workers, *args, **kwargs
    ) -> list[ChunkerInfo]:
        raise NotImplementedError


class FileChunker(Chunker):
    def __init__(self):
        super().__init__("dynamic txt word counter", ChunkerTypeEnum.DYNAMIC)
        self.separator = "\n"

    def my_byte_range(
        self, flex_input, worker_id, num_workers, *args, **kwargs
    ) -> list[ChunkerInfo]:
        # TODO: support more than one file scenarios
        [key] = flex_input.keys
        file_size = int(Storage().head_object(flex_input.bucket, key)["content-length"])
        start = (worker_id * file_size) // num_workers
        end = ((worker_id + 1) * file_size) // num_workers
        return [ChunkerInfo(key, start, end)]
