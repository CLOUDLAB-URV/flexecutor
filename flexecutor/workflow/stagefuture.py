from __future__ import annotations

from typing import Any, Optional, List

from lithops.utils import FuturesList

from flexecutor.utils.dataclass import FunctionTimes


class StageFuture:
    def __init__(self, stage_id: str, future: Optional[FuturesList] = None):
        self.__stage_id = stage_id
        self.__future = future

    def result(self) -> Any:
        return [i[0] for i in self.__future.get_result()]

    def _timings_list(self) -> list[FunctionTimes]:
        return [i[1] for i in self.__future.get_result()]

    @property
    def stats(self):
        return [f.stats for f in self.__future]

    def error(self) -> bool:
        return any([f.error for f in self.__future])

    def __getattr__(self, item):
        if item in vars(self):
            return getattr(self, item)
        elif "__future" in vars(self) and item in vars(self.__future):
            return getattr(self.__future, item)
        raise AttributeError(f"Future object has no attribute {item}")

    def get_timings(self) -> List[FunctionTimes]:
        """Get the timings of the future."""
        timings_list = []
        for r, s in zip(self._timings_list(), self.stats):
            host_submit_tstamp = s["host_submit_tstamp"]
            worker_start_tstamp = s["worker_start_tstamp"]
            r.cold_start = worker_start_tstamp - host_submit_tstamp
            timings_list.append(r)
        return timings_list
