import time
from functools import wraps
from typing import Callable, Any

import numpy as np

from flexecutor.utils.dataclass import FunctionTimes
from flexecutor.utils.iomanager import InternalIOManager, IOManager


def worker_wrapper(func: Callable[..., Any]):
    @wraps(func)
    def wrapper(io: InternalIOManager, *args, **kwargs):
        before_read = time.time()
        io.download_files()
        after_read = time.time()
        func_io = IOManager(io)
        result = func(func_io, *args, **kwargs)
        before_write = time.time()
        io.upload_files()
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
