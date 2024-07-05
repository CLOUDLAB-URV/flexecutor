import time
import os
from functools import wraps
from typing import Callable, Any

import numpy as np
from botocore.response import StreamingBody
from lithops import Storage

from flexecutor.storage.chunker import ChunkInfo
from flexecutor.storage.storage import StrategyEnum
from flexecutor.utils.dataclass import FunctionTimes
from flexecutor.utils.storagecontext import (
    InternalStorageContext,
    StorageContext,
)


def worker_wrapper(func: Callable[..., Any]):
    @wraps(func)
    def wrapper(st_context: InternalStorageContext, *args, **kwargs):
        before_read = time.time()
        storage = Storage()
        # TODO: parallelize download?
        for input_id, flex_input in st_context.inputs.items():
            os.makedirs(flex_input.local_base_path, exist_ok=True)
            if (
                len(flex_input.keys) >= st_context.num_workers
                or flex_input.strategy is StrategyEnum.BROADCAST
            ):  # More files than workers and scattering
                start_index, end_index = flex_input.file_index
                for index in range(start_index, end_index):
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
                chunker: ChunkInfo = flex_input.chunker.my_byte_range(
                    flex_input, st_context.worker_id, st_context.num_workers
                )[0]
                extra_args = {"Range": f"bytes={chunker.start}-{chunker.end}"}
                chunk = storage.get_object(
                    flex_input.bucket,
                    flex_input.keys[0],
                    flex_input.local_paths[0],
                    extra_get_args=extra_args,
                )
                flex_input.local_paths[0] += ".part" + str(st_context.worker_id)
                with open(flex_input.local_paths[0], "wb") as f:
                    if isinstance(chunk, StreamingBody):
                        f.write(chunk.read())

        after_read = time.time()

        func_st_context = StorageContext(st_context)
        result = func(func_st_context, *args, **kwargs)

        before_write = time.time()
        # TODO: parallelize upload?
        for output_id, flex_output in st_context.outputs.items():
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
        times["total"] = float(np.mean(list(times.values())))
        func_times = FunctionTimes(**times)

        return result, func_times

    return wrapper
