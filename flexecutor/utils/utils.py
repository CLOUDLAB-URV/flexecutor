import json
import logging
import os
import time
from contextlib import contextmanager


def initialize_timings():
    return {"read": 0, "compute": 0, "write": 0}


@contextmanager
def operation(op_type: str, timings: dict):
    start_time = time.time()
    yield
    end_time = time.time()
    timings[op_type] += end_time - start_time


def get_timings(timings: dict):
    return timings


def reset_timings(timings: dict):
    for key in timings:
        timings[key] = 0


# TODO: review if this function will alive here
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


def load_profiling_results(file: str, logger: logging.Logger) -> dict:
    if not os.path.exists(file):
        logger.info(
            f"No existing profiling results found at {file}. Initializing empty results."
        )
        return {}
    try:
        with open(file, "r") as f:
            data = json.load(f)
            results = {eval(k): v for k, v in data.items()}
            return results
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from {file}. Initializing empty results."
        )
        return {}


def save_profiling_results(file, profile_data):
    serial_data = {str(k): v for k, v in profile_data.items()}
    with open(file, "w") as f:
        json.dump(serial_data, f, indent=4)
