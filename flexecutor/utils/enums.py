from enum import Enum


class StrategyEnum(Enum):
    SCATTER = 1
    BROADCAST = 2


class ChunkerTypeEnum(Enum):
    STATIC = 1
    DYNAMIC = 2