import os
from pathlib import Path

from flexecutor.storage.storage import FlexInput


class ChunkerContext:
    def __init__(self, flex_input: FlexInput, prefix_output: str, num_workers: int):
        self.chunk_dir = None
        self.flex_input = flex_input
        self.prefix_output = prefix_output
        self.num_workers = num_workers
        self.output_paths = []
        self.output_keys = []
        self.counter = 0

    def get_input_paths(self):
        return self.flex_input.local_paths

    def next_chunk_path(self):
        # TODO: un-hardcode suffix
        new_local_base_path = Path(
            str(self.flex_input.local_base_path).replace(
                self.flex_input.prefix.removesuffix("/"), self.prefix_output
            )
        )
        self.chunk_dir = new_local_base_path
        os.makedirs(self.chunk_dir, exist_ok=True)
        local_path = (
            f"{new_local_base_path}/part{self.counter}.csv"
        )
        self.output_paths.append(local_path)
        key = f"{self.prefix_output}/part{self.counter}.csv"
        self.output_keys.append(key)
        self.counter += 1
        return local_path
