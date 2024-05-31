import operator
import random
from typing import Dict

import numpy as np
from deap import algorithms, base, creator, gp, tools
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.modelling.prediction import Prediction


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


class GAPerfModel(PerfModel):
    def __init__(
        self,
        population_size=300,
        crossover_prob=0.7,
        mutation_prob=0.2,
        n_generations=40,
    ):
        super().__init__("genetic")
        self._population_size = population_size
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._n_generations = n_generations
        self._data = None
        self._best_individual = None
        self._toolbox = base.Toolbox()
        self._setup_genetic_algorithm()

    def _setup_genetic_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        pset = gp.PrimitiveSet("MAIN", 3)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
        pset.renameArguments(ARG0="cpus", ARG1="memory", ARG2="workers")

        self._toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self._toolbox.register(
            "individual", tools.initIterate, creator.Individual, self._toolbox.expr
        )
        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )
        self._toolbox.register("compile", gp.compile, pset=pset)

        self._toolbox.register("evaluate", self._evaluate)
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register(
            "mutate", gp.mutUniform, expr=self._toolbox.expr, pset=pset
        )
        self._toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self._toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

    def _evaluate(self, individual):
        func = self._toolbox.compile(expr=individual)
        errors = []
        for cpus, memory, workers, actual in self._data:
            predicted = func(cpus, memory, workers)
            errors.append((predicted - actual) ** 2)
        return (np.mean(errors),)

    def train(self, profiling_results: Dict) -> None:
        def preprocess_profiling_data(profiling_data):
            processed_data = []
            for config, executions in profiling_data.items():
                cpus, memory, workers = config
                latencies = [
                    sum(lats)
                    for breaks in zip(
                        executions["read"],
                        executions["compute"],
                        executions["write"],
                        executions["cold_start_time"],
                    )
                    for lats in zip(*breaks)
                ]
                # adapt to new profiling_data structure
                # latencies = [latency for run in executions for latency in run]

                # print(latencies)

                # Hay stagglers cuando hacemos profiling, la idea con esto es escoger percentiles no utilizar stagglers
                q1 = np.percentile(latencies, 25)
                q3 = np.percentile(latencies, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                filtered_latencies = [
                    latency for latency in latencies if lower_bound <= latency <= upper_bound
                ]

                for latency in filtered_latencies:
                    processed_data.append((cpus, memory, workers, latency))

            return processed_data

        self._data = preprocess_profiling_data(profiling_results)
        pop = self._toolbox.population(n=self._population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(
            pop,
            self._toolbox,
            cxpb=self._crossover_prob,
            mutpb=self._mutation_prob,
            ngen=self._n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
        self._best_individual = hof[0]
        self._objective_func = self._best_individual

    @property
    @overrides
    def parameters(self):
        return "Yet to be implemented"

    def predict(self, num_cpu, runtime_memory, num_workers, chunk_size=None) -> Prediction:
        func = self._toolbox.compile(expr=self._best_individual)
        return Prediction(func(num_cpu, runtime_memory, num_workers))

