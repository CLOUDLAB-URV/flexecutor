import time

from flexecutor.future import Future
from flexecutor.utils import initialize_timings, operation


def sleepy_func(input_data: Future, *args, **kwargs):
    timings = initialize_timings()

    with operation("read", timings):
        time.sleep(1)

    with operation("compute", timings):
        time.sleep(3)

    with operation("write", timings):
        time.sleep(0.7)

    return timings
