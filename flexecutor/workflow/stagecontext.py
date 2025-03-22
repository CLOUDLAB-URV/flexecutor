import os
import uuid
from typing import Optional, Any

from lithops import Storage

from flexecutor.storage.storage import FlexData
from flexecutor.utils.enums import StrategyEnum, ChunkerTypeEnum


class InternalStageContext:
    def __init__(
        self,
        worker_id,
        num_workers,
        inputs: list[FlexData],
        outputs: list[FlexData],
        params: Optional[dict[str, Any]],
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.inputs: dict[str, FlexData] = {i.id: i for i in inputs}
        self.outputs: dict[str, FlexData] = {o.id: o for o in outputs}
        self._params = params

    def __repr__(self):
        return f"InternalStageContext(worker_id={self.worker_id}, num_workers={self.num_workers}, inputs={self.inputs}, outputs={self.outputs}, params={self._params})"

    def input_paths(self, input_id: str) -> list[str]:
        start, end = self.inputs[input_id].file_indexes
        return self.inputs[input_id].local_paths[start:end]

    def get_param(self, key: str) -> Any:
        return self._params[key]

    def next_output_path(self, param: str) -> str:
        os.makedirs(self.outputs[param].local_base_path, exist_ok=True)
        serial = str(uuid.uuid4())[0:8] + self.outputs[param].suffix
        local_path = f"{self.outputs[param].local_base_path}/{serial}"
        self.outputs[param].local_paths.append(local_path)
        self.outputs[param].keys.append(f"{self.outputs[param].prefix}{serial}")
        return local_path

    def download_files(self):
        storage = Storage()
        # TODO: parallelize download?
        for input_id, flex_data in self.inputs.items():
            os.makedirs(flex_data.local_base_path, exist_ok=True)
            # if (
            #     len(flex_data.keys) >= self.num_workers
            #     or flex_data.read_strategy is StrategyEnum.BROADCAST
            #     or flex_data.has_chunker_type(ChunkerTypeEnum.STATIC)
            # ):  # More files than workers and scattering
            #     start_index, end_index = flex_data.file_indexes
            #     for index in range(start_index, end_index):
            #         storage.download_file(
            #             flex_data.bucket,
            #             flex_data.keys[index],
            #             flex_data.local_paths[index],
            #         )
            # else:  # Dynamic partitioning
            #     chunker = flex_data.chunker
            #     output = chunker.data_slices[self.worker_id].get()
            #     filename = f"{flex_data.local_base_path}_worker_{self.worker_id}"
            #     with open(filename, "wb") as f:
            #         f.write(output.encode("utf-8"))
            #     flex_data.set_local_paths([filename])
            if flex_data.has_chunker_type(ChunkerTypeEnum.DYNAMIC):
                chunker = flex_data.chunker
                output = chunker.data_slices[self.worker_id].get()
                filename = f"{flex_data.local_base_path}_worker_{self.worker_id}"
                with open(filename, "wb") as f:
                    f.write(output.encode("utf-8"))
                flex_data.set_local_paths([filename])
            else:
                for index in range(len(flex_data.keys)):
                    storage.download_file(
                        flex_data.bucket,
                        flex_data.keys[index],
                        flex_data.local_paths[index],
                    )

    def upload_files(self):
        storage = Storage()
        # TODO: parallelize upload?
        for output_id, flex_data in self.outputs.items():
            for index in range(len(flex_data.local_paths)):
                storage.upload_file(
                    flex_data.local_paths[index],
                    flex_data.bucket,
                    flex_data.keys[index],
                )


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
    
    def is_dynamic_chunker(self, input_id: str) -> bool:
        return self._context.inputs[input_id].has_chunker_type(ChunkerTypeEnum.DYNAMIC)
