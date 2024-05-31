import numpy as np
from scipy.optimize import differential_evolution


class OptimizationProblemSolver:
    def __init__(self, workflow_stage):
        self.workflow_stage = workflow_stage

    def search_config(self, bounds):
        objective_func = self.workflow_stage.objective_func

        def integer_objective_func(x):
            x_int = np.round(x).astype(int)
            return objective_func(x_int)

        res = differential_evolution(
            integer_objective_func,
            bounds,
            strategy="best1bin",
            mutation=(0.5, 1),
            recombination=0.7,
            disp=True,
        )
        return res


class Scheduler:
    def __init__(self, workflow_stage):
        self.workflow_stage = workflow_stage
        self.optimization_problem_solver = OptimizationProblemSolver(
            self.workflow_stage
        )

    def search_config(self, bounds):
        res = self.optimization_problem_solver.search_config(bounds)
        return res
