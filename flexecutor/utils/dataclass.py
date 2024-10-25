from dataclasses import dataclass
from typing import Optional


@dataclass
class StageConfig:
    """
    Resource configuration for one stage.
    """

    cpu: float
    workers: int
    memory: float = 0

    @property
    def key(self) -> tuple[float, float, int]:
        return self.cpu, self.memory, self.workers

    def __array__(self):
        return [self.workers, self.cpu]


@dataclass
class ConfigBounds:
    """
    Configuration bounds for the stage
    """

    cpu: tuple[float, float]
    memory: tuple[float, float]
    workers: tuple[int, int]

    def to_tuple_list(self) -> list[tuple]:
        return [self.cpu, self.memory, self.workers]


@dataclass
class FunctionTimes:
    read: Optional[float] = None
    compute: Optional[float] = None
    write: Optional[float] = None
    cold_start: Optional[float] = None
    total: Optional[float] = None

    @classmethod
    def profile_keys(cls) -> list[str]:
        return ["read", "compute", "write", "cold_start"]

    def __lt__(self, other):
        return self.total < other.total
