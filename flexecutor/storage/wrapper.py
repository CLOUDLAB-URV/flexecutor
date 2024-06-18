from functools import wraps
from typing import Callable, Any

from lithops import Storage

from flexecutor.utils.iomanager import IOManager


def worker_wrapper(func: Callable[[...], Any]):
    @wraps(func)
    def wrapper(io: IOManager, *args, **kwargs):
        storage = Storage()
        storage.download_file(
            io.input_file.bucket, io.input_file.key, io.input_file.local_path
        )
        result = func(io, *args, **kwargs)
        storage.upload_file(
            io.output_paths("foo"), io.output_path.bucket, io.output_key
        )
        return result
    return wrapper
