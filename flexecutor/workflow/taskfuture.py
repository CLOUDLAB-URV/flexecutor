from __future__ import annotations

from typing import Any, Optional

from lithops.utils import FuturesList


class TaskFuture:
    def __init__(self, future: Optional[FuturesList] = None):
        self.__future = future

    def result(self) -> Any:
        return self.__future.get_result()

    @property
    def stats(self):
        return [f.stats for f in self.__future]

    def error(self) -> bool:
        return any([f.error for f in self.__future])

    def __getattr__(self, item):
        if item in vars(self):
            return getattr(self, item)
        elif '__future' in vars(self) and item in vars(self.__future):
            return getattr(self.__future, item)
        raise AttributeError(f"Future object has no attribute {item}")


class InputFile(TaskFuture):
    def __init__(self, file: Any):
        super().__init__()
        self._file = file

    @property
    def file(self) -> Any:
        return self._file

    def result(self) -> Any:
        return self._file

    def error(self) -> bool:
        return False
