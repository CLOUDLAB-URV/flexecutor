import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
from lithops import Storage


class ChunkerTypeEnum(Enum):
    STATIC = 1
    DYNAMIC = 2


@dataclass
class ChunkerInfo:
    key: str
    start: Optional[int]
    end: Optional[int]


class Chunker(ABC):
    def __init__(self, prefix, chunker_type: ChunkerTypeEnum):
        if prefix and prefix[-1] != "/":
            prefix += "/"
        self.prefix = prefix
        self.chunker_type = chunker_type

    @abstractmethod
    def preprocess(self, flex_input, worker_id, num_workers) -> None:
        pass

    @abstractmethod
    def get_my_chunk(self, flex_input, worker_id, num_workers) -> ChunkerInfo:
        pass


class CarelessFileChunker(Chunker):
    def __init__(self, ):
        super().__init__("", ChunkerTypeEnum.DYNAMIC)

    def preprocess(self, flex_input, worker_id, num_workers) -> None:
        """
        CarelessFileChunker does not require any preprocessing,
        because it does not care about the file structure.
        """
        pass

    def get_my_chunk(self, flex_input, worker_id, num_workers) -> list[ChunkerInfo]:
        # TODO: support more than one file scenarios
        [key] = flex_input.keys
        file_size = int(Storage().head_object(flex_input.bucket, key)["content-length"])
        start = (worker_id * file_size) // num_workers
        end = ((worker_id + 1) * file_size) // num_workers
        return [ChunkerInfo(key, start, end)]


class WordCounterChunker(Chunker):
    def __init__(self, prefix):
        super().__init__(prefix, ChunkerTypeEnum.STATIC)

    def preprocess(self, flex_input, worker_id, num_workers) -> None:
        storage = Storage()
        filename = "tiny-shakespeare.txt"
        key = f"{self.prefix}{filename}"
        file_size = int(storage.head_object(flex_input.bucket, key)["content-length"])
        file = storage.get_object(flex_input.bucket, key)
        text = file.decode("utf-8")
        start = 0
        for worker_id in range(num_workers):
            end = ((worker_id + 1) * file_size) // num_workers
            end = min(text.rfind(" ", start, end), end)
            storage.put_object(
                flex_input.bucket,
                f"{flex_input.prefix}{filename}.part{worker_id}",
                text[start:end].encode("utf-8"),
            )
            start = end + 1
        return

    def get_my_chunk(self, flex_input, worker_id, num_workers) -> list[ChunkerInfo]:
        pass


class CsvStaticChunker(Chunker):
    def __init__(self, prefix):
        super().__init__(prefix, ChunkerTypeEnum.STATIC)

    def preprocess(self, flex_input, worker_id, num_workers) -> None:
        storage = Storage()
        filename = "titanic.csv"
        key = f"{self.prefix}{filename}"
        df = pd.read_csv(io.BytesIO(storage.get_object(flex_input.bucket, key)))
        chunk_size = len(df) // num_workers
        chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
        for worker_id, chunk in enumerate(chunks):
            storage.put_object(
                flex_input.bucket,
                f"{flex_input.prefix}{filename}.part{worker_id}",
                chunk.to_csv(index=False).encode("utf-8"),
            )
        return

    def get_my_chunk(self, flex_input, worker_id, num_workers) -> list[ChunkerInfo]:
        pass


class CsvDynamicChunker(Chunker):
    def __init__(self, prefix):
        super().__init__(prefix, ChunkerTypeEnum.DYNAMIC)

    def preprocess(self, flex_input, worker_id, num_workers) -> None:
        pass

    def get_my_chunk(self, flex_input, worker_id, num_workers) -> list[ChunkerInfo]:
        pass


class ChunkerManager:
    def __init__(self, chunker: Chunker, flex_input, dst):
        self.input = flex_input
        self.chunker = chunker
        self.dst = dst
