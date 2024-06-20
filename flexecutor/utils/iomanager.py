import os
import uuid

from flexecutor.storage.storage import FlexInput, FlexOutput


class IOManager:
    def __init__(
        self, worker_id, num_workers, inputs: list[FlexInput], outputs: list[FlexOutput]
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.inputs: dict[str, FlexInput] = {i.id: i for i in inputs}
        self.outputs: dict[str, FlexOutput] = {o.id: o for o in outputs}
        self.write_counters = {o.id: 0 for o in outputs}

    def input_paths(self, input_id) -> list[str]:
        start, end = self.inputs[input_id].chunk_indexes
        return self.inputs[input_id].local_paths[start:end]

    @property
    def params(self):
        return {"worker_id": 0}

    def next_output_path(self, param):
        os.makedirs(self.outputs[param].local_base_path, exist_ok=True)
        serial = str(uuid.uuid4())[0:8] + self.outputs[param].suffix
        local_path = f"{self.outputs[param].local_base_path}/{serial}"
        self.outputs[param].local_paths.append(local_path)
        self.outputs[param].keys.append(f"{self.outputs[param].prefix}/{serial}")
        return local_path
