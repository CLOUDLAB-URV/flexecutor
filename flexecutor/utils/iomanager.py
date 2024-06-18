import os
import uuid

from flexecutor.storage.storage import InputS3File, OutputS3Path


class IOManager:
    def __init__(self, worker_id, input_file: InputS3File, output_path: OutputS3Path):
        self.worker_id = worker_id
        self.input_file = input_file
        self.output_path = output_path
        self.output_file = None
        self.output_key = None

    def input_file_func(self, param="foo") -> list[str]:
        return [self.input_file.local_path]

    @property
    def params(self):
        return {"worker_id": 0}

    def output_paths(self, param="foo") -> str:
        if self.output_file is None:
            os.makedirs(self.output_path.local_base_path, exist_ok=True)
            file = f"{str(uuid.uuid4())[0:8]}.count"
            self.output_file = str(self.output_path.local_base_path) + "/" + file
            self.output_key = file
        return self.output_file

    def next_output_path(self, param):
        pass
