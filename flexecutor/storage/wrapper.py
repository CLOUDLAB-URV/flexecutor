import time
import os
from functools import wraps
from typing import Callable, Any

import numpy as np
from lithops import Storage

from flexecutor.utils.dataclass import FunctionTimes
from flexecutor.utils.iomanager import IOManager


def worker_wrapper(func: Callable[[...], Any]):
    @wraps(func)
    def wrapper(io: IOManager, *args, **kwargs):
        before_read = time.time()
        storage = Storage()
        # TODO: parallelize download?
        for input_id, flex_input in io.inputs.items():
            start_index, end_index = flex_input.chunk_indexes
            os.makedirs(flex_input.local_base_path, exist_ok=True)
            for index in range(start_index, end_index):
                storage.download_file(
                    flex_input.bucket,
                    flex_input.keys[index],
                    flex_input.local_paths[index],
                )
        after_read = time.time()

        result = func(io, *args, **kwargs)

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
