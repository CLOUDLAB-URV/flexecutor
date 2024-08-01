from modelling.perfmodel import PerfModelEnum
from scheduling.scheduler import Scheduler


class Caerus(Scheduler):
    def __init__(self, dag):
        super().__init__(dag, PerfModelEnum.ANALYTIC)

    def schedule(self):
        pass
