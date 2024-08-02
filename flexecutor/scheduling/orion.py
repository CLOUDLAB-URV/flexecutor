from modelling.perfmodel import PerfModelEnum
from scheduling.scheduler import Scheduler


class Orion(Scheduler):
    def __init__(self, dag):
        super().__init__(dag, PerfModelEnum.DISTRIBUTION)

    def schedule(self):
        pass
