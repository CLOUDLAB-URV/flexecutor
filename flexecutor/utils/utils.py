import inspect
import json
import logging
import os
import time
import ast
from contextlib import contextmanager


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


def load_profiling_results(file: str) -> dict:
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        try:
            data = json.load(f)
            # Convert string keys back to tuples
            data = {ast.literal_eval(k): v for k, v in data.items()}
        except (json.JSONDecodeError, ValueError, SyntaxError):
            raise ValueError(f"Error loading profiling results from {file}")
    return data


def save_profiling_results(file: str, profile_data: dict):
    serial_data = {str(k): v for k, v in profile_data.items()}
    with open(file, "w") as f:
        json.dump(serial_data, f, indent=4)


FLEXECUTOR_EXEC_PATH = "FLEXECUTOR_EXEC_PATH"
BUCKET_NAME = "FLEX_BUCKET"


def get_my_exec_path():
    """
    Get the path where the flexorchestrator script is located
    @flexorchestrator() decorator is responsible for setting this path

    :return: the path where the flexorchestrator script is located
    """
    return os.environ.get(FLEXECUTOR_EXEC_PATH) or os.getcwd()


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
            if "__file__" in caller_frame.f_globals:
                caller_file = caller_frame.f_globals["__file__"]
            elif "__session__" in caller_frame.f_globals:
                caller_file = caller_frame.f_globals["__session__"]
            else:
                caller_file = ""
            value = os.path.dirname(os.path.abspath(caller_file))
            os.environ[key] = value
            os.environ[BUCKET_NAME] = bucket
            try:
                result = func(*args, **kwargs)
            finally:
                os.environ.pop(key, None)
            return result

        return wrapper

    return function
