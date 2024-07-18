import os
import time
from functools import wraps
from typing import Callable, Any

import numpy as np
from lithops import Storage

from flexecutor.utils.chunker_context import ChunkerContext
from flexecutor.utils.dataclass import FunctionTimes
from flexecutor.utils.enums import ChunkerTypeEnum, StrategyEnum
from flexecutor.utils.iomanager import InternalIOManager, IOManager


def worker_wrapper(func: Callable[[...], Any]):
    @wraps(func)
    def wrapper(io: InternalIOManager, *args, **kwargs):
        before_read = time.time()
        storage = Storage()
        # TODO: parallelize download?
        for input_id, flex_input in io.inputs.items():
            os.makedirs(flex_input.local_base_path, exist_ok=True)
            if (
                len(flex_input.keys) >= io.num_workers
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
                output = chunker.data_slices[io.worker_id].get()
                filename = f"{flex_input.local_base_path}_worker_{io.worker_id}"
                with open(filename, "wb") as f:
                    f.write(output.encode("utf-8"))
                flex_input.set_local_paths([filename])

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


def chunker_wrapper(func: Callable[[...], Any], ctx: ChunkerContext, *args, **kwargs):
    # Download the files to the local storage
    storage = Storage()
    flex_input = ctx.flex_input
    os.makedirs(flex_input.local_base_path, exist_ok=True)
    for index in range(len(flex_input.keys)):
        storage.download_file(
            flex_input.bucket, flex_input.keys[index], flex_input.local_paths[index]
        )

    # Execute the chunker function
    result = func(ctx, *args, **kwargs)

    # Upload the chunked files to the object storage
    for index in range(len(ctx.output_paths)):
        storage.upload_file(
            ctx.output_paths[index], flex_input.bucket, ctx.output_keys[index]
        )

    # Adapt the flex_input object to the new state
    flex_input.custom_output_id = flex_input.prefix
    flex_input.prefix = ctx.prefix_output
    flex_input.chunker = None
    flex_input.flush()

    return
