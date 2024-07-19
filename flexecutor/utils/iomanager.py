import os
import uuid
from typing import Optional, Any

from lithops import Storage

from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.enums import StrategyEnum, ChunkerTypeEnum


class InternalIOManager:
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

    def input_paths(self, input_id: str) -> list[str]:
        start, end = self.inputs[input_id].chunk_indexes
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

    def download_files(self):
        storage = Storage()
        # TODO: parallelize download?
        for input_id, flex_input in self.inputs.items():
            os.makedirs(flex_input.local_base_path, exist_ok=True)
            if (
                len(flex_input.keys) >= self.num_workers
                or flex_input.strategy is StrategyEnum.BROADCAST
                or flex_input.has_chunker_type(ChunkerTypeEnum.STATIC)
            ):  # More files than workers and scattering
                start_index, end_index = flex_input.chunk_indexes
                for index in range(start_index, end_index):
                    storage.download_file(
                        flex_input.bucket,
                        flex_input.keys[index],
                        flex_input.local_paths[index],
                    )
            else:  # Dynamic partitioning
                chunker = flex_input.chunker
                output = chunker.data_slices[self.worker_id].get()
                filename = f"{flex_input.local_base_path}_worker_{self.worker_id}"
                with open(filename, "wb") as f:
                    f.write(output.encode("utf-8"))
                flex_input.set_local_paths([filename])

    def upload_files(self):
        storage = Storage()
        # TODO: parallelize upload?
        for output_id, flex_output in self.outputs.items():
            for index in range(len(flex_output.local_paths)):
                storage.upload_file(
                    flex_output.local_paths[index],
                    flex_output.bucket,
                    flex_output.keys[index],
                )


# IOManager is a facade for InternalIOManager
class IOManager:
    def __init__(self, manager: InternalIOManager):
        self._manager = manager

    def get_input_paths(self, input_id: str) -> list[str]:
        return self._manager.input_paths(input_id)

    def get_param(self, key: str) -> Any:
        return self._manager.get_param(key)

    def next_output_path(self, param: str) -> str:
        return self._manager.next_output_path(param)
