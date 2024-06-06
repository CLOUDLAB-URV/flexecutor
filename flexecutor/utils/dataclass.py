from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceConfig:
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
class FunctionTimes:
    read: Optional[float] = None
    compute: Optional[float] = None
    write: Optional[float] = None
    cold_start: Optional[float] = None
    total: Optional[float] = None

    @classmethod
    def profile_keys(cls) -> list[str]:
        return ["read", "compute", "write", "cold_start"]
