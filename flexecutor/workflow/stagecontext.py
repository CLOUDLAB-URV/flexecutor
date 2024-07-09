import os
import uuid
from typing import Optional, Any

from flexecutor.storage.storage import FlexInput, FlexOutput


class InternalStageContext:
    def __init__(
        self,
        worker_id,
        num_workers,
        inputs: list[FlexInput],
        outputs: list[FlexOutput],
        params: Optional[dict[str, Any]],
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.inputs: dict[str, FlexInput] = {i.id: i for i in inputs}
        self.outputs: dict[str, FlexOutput] = {o.id: o for o in outputs}
        self._params = params

    def __repr__(self):
        return f"InternalStageContext(worker_id={self.worker_id}, num_workers={self.num_workers}, inputs={self.inputs}, outputs={self.outputs}, params={self._params})"

    def input_paths(self, input_id: str) -> list[str]:
        start, end = self.inputs[input_id].file_index
        return self.inputs[input_id].local_paths[start:end]

    def get_param(self, key: str) -> Any:
        return self._params[key]

    def next_output_path(self, param: str) -> str:
        os.makedirs(self.outputs[param].local_base_path, exist_ok=True)
        serial = str(uuid.uuid4())[0:8] + self.outputs[param].suffix
        local_path = f"{self.outputs[param].local_base_path}/{serial}"
        self.outputs[param].local_paths.append(local_path)
        self.outputs[param].keys.append(f"{self.outputs[param].prefix}/{serial}")
        return local_path


class StageContext:
    def __init__(self, context: InternalStageContext):
        self._context = context

    def __repr__(self):
        return f"StageContext({self._context})"

    def get_input_paths(self, input_id: str) -> list[str]:
        return self._context.input_paths(input_id)

    def get_param(self, key: str) -> Any:
        return self._context.get_param(key)

    def next_output_path(self, param: str) -> str:
        return self._context.next_output_path(param)
