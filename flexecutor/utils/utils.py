import inspect
import json
import logging
import os
import time
import ast
from contextlib import contextmanager
from lithops import Storage


def remove_dir_os(bucket: str, dir: str):
    keys_to_delete = [key for key in Storage().list_keys(bucket, prefix=dir)]
    Storage().delete_objects(bucket, keys_to_delete)


def initialize_timings():
    return {"read": 0, "compute": 0, "write": 0}


@contextmanager
def operation(op_type: str, timings: dict):
    start_time = time.time()
    yield
    end_time = time.time()
    timings[op_type] += end_time - start_time


# def get_timings(timings: dict):
#     return timings
#
#
# def reset_timings(timings: dict):
#     for key in timings:
#         timings[key] = 0


# TODO: review if this function should be here
def setup_logging(level):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    log_format = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s"
        if level == logging.DEBUG
        else "%(asctime)s [%(levelname)s] %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def save_profiling_results(file: str, profile_data: dict):
    with open(file, "w") as f:
        json.dump(profile_data, f, indent=4)


def load_profiling_results(file: str) -> dict:
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        try:
            data = json.load(f)
            # Convert string keys back to tuples for the outer dictionary
            data = {k: v for k, v in data.items()}
        except (json.JSONDecodeError, ValueError, SyntaxError):
            return {}
    return data


FLEXECUTOR_EXEC_PATH = "FLEXECUTOR_EXEC_PATH"


def get_my_exec_path():
    """
    Get the path where the flexorchestrator script is located
    @flexorchestrator() decorator is responsible for setting this path

    :return: the path where the flexorchestrator script is located
    """
    return os.environ.get(FLEXECUTOR_EXEC_PATH, None)


def flexorchestrator(bucket=""):
    """
    Decorator to initializations previous to the execution of user scripts.
    You must use it only in the main function of your script.
    Responsible for:
    - Set the path if where the flexorchestrator main script is located
    :param bucket:
    :return:
    """

    def function(func):
        def wrapper(*args, **kwargs):
            # Set the path of the flexorchestrator file
            key = FLEXECUTOR_EXEC_PATH
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_file = caller_frame.f_globals["__file__"]
            value = os.path.dirname(os.path.abspath(caller_file))
            os.environ[key] = value
            # Set the bucket
            key = "FLEX_BUCKET"
            os.environ[key] = bucket
            try:
                result = func(*args, **kwargs)
            finally:
                os.environ.pop(key, None)
            return result

        return wrapper

    return function
