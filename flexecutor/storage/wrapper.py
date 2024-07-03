import time
import os
from functools import wraps
from typing import Callable, Any

import numpy as np
from botocore.response import StreamingBody
from lithops import Storage

from flexecutor.storage.chunker import ChunkerInfo
from flexecutor.storage.storage import StrategyEnum
from flexecutor.utils.dataclass import FunctionTimes
from flexecutor.utils.iomanager import InternalIOManager, IOManager


def worker_wrapper(func: Callable[[...], Any]):
    @wraps(func)
    def wrapper(io: InternalIOManager, *args, **kwargs):
        before_read = time.time()
        storage = Storage()
        # TODO: parallelize download?
        for input_id, flex_input in io.inputs.items():
            print("Creating local folder for input" + str(flex_input.local_base_path))
            os.makedirs(flex_input.local_base_path, exist_ok=True)
            print(f"Downloading {flex_input.keys} to {flex_input.local_paths}")
            if (
                len(flex_input.keys) >= io.num_workers
                or flex_input.strategy is StrategyEnum.BROADCAST
            ):  # More files than workers and scattering
                start_index, end_index = flex_input.chunk_indexes
                for index in range(start_index, end_index):
                    print(f"Downloading {flex_input.keys[index]} to {flex_input.local_paths[index]}")
                    storage.download_file(
                        flex_input.bucket,
                        flex_input.keys[index],
                        flex_input.local_paths[index],
                    )
            else:  # Fewer files than workers --> chunking
                if flex_input.chunker is None:
                    raise Exception(
                        "Chunker is required for scatter strategy with more workers than files."
                    )
                # TODO: fix, only works for one file
                chunker: ChunkerInfo = flex_input.chunker.my_byte_range(
                    flex_input, io.worker_id, io.num_workers
                )[0]
                extra_args = {"Range": f"bytes={chunker.start}-{chunker.end}"}
                chunk = storage.get_object(
                    flex_input.bucket,
                    flex_input.keys[0],
                    flex_input.local_paths[0],
                    extra_get_args=extra_args,
                )
                flex_input.local_paths[0] += ".part" + str(io.worker_id)
                with open(flex_input.local_paths[0], "wb") as f:
                    if isinstance(chunk, StreamingBody):
                        f.write(chunk.read())

        after_read = time.time()

        func_io = IOManager(io)
        result = func(func_io, *args, **kwargs)

        before_write = time.time()
        # TODO: parallelize upload?
        for output_id, flex_output in io.outputs.items():
            for index in range(len(flex_output.local_paths)):
                storage.upload_file(
                    flex_output.local_paths[index],
                    flex_output.bucket,
                    flex_output.keys[index],
                )
        after_write = time.time()

        times = {
            "read": after_read - before_read,
            "compute": before_write - after_read,
            "write": after_write - before_write,
        }
        times["total"] = np.mean(list(times.values()))
        func_times = FunctionTimes(**times)

        return result, func_times

    return wrapper
