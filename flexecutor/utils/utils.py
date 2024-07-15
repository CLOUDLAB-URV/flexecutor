import inspect
import logging
import time
import os
from contextlib import contextmanager


def initialize_timings():
    return {"read": 0, "compute": 0, "write": 0}


@contextmanager
def operation(op_type: str, timings: dict):
    start_time = time.time()
    yield
    end_time = time.time()
    timings[op_type] += end_time - start_time


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
