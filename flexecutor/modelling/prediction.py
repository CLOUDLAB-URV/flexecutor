from dataclasses import dataclass


@dataclass
class Prediction:
    read_time: float | None
    compute_time: float | None
    write_time: float | None
    cold_start_time: float | None
    total_time: float

    def __init__(self, total_time, read_time=None, compute_time=None, write_time=None, cold_start_time=None):
        self.read_time = read_time
        self.compute_time = compute_time
        self.write_time = write_time
        self.cold_start_time = cold_start_time
        self.total_time = total_time
