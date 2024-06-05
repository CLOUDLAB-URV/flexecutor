from dataclasses import dataclass


@dataclass
class ConfigSpace:
    """
    Configuration space for the task
    """
    cpu: float
    memory: float
    workers: int

    @property
    def key(self) -> tuple[float, float, int]:
        return self.cpu, self.memory, self.workers


@dataclass
class ConfigBounds:
    """
    Configuration space bounds for the task
    """
    cpu: tuple[float, float]
    memory: tuple[float, float]
    workers: tuple[int, int]

    def to_tuple_list(self) -> list[tuple]:
        return [self.cpu, self.memory, self.workers]


@dataclass
class FunctionProfiling:
    """
    Timing of the function
    """
    read: float
    compute: float
    write: float
    cold_start_time: float

    @classmethod
    def metrics(cls) -> list[str]:
        return ["read", "compute", "write", "cold_start_time"]


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
