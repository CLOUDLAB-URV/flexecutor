import time
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
        storage.download_file(
            io.input_file.bucket, io.input_file.key, io.input_file.local_path
        )
        after_read = time.time()

        result = func(io, *args, **kwargs)

        before_write = time.time()
        storage.upload_file(
            io.output_paths("foo"), io.output_path.bucket, io.output_key
        )
        after_write = time.time()

        times = {
            "read": after_read - before_read,
            "compute": before_write - after_read,
            "write": after_write - before_write,
        }
        times['total'] = np.mean(list(times.values()))
        func_times = FunctionTimes(**times)

        return result, func_times

    return wrapper
