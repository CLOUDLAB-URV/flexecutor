import operator
import random

import numpy as np
from deap import algorithms, base, creator, gp, tools


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


class GAPerfModel:
    def __init__(
        self,
        population_size=300,
        crossover_prob=0.7,
        mutation_prob=0.2,
        n_generations=40,
    ):
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_generations = n_generations
        self.data = None
        self.toolbox = base.Toolbox()
        self.setup_genetic_algorithm()

    def setup_genetic_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        pset = gp.PrimitiveSet("MAIN", 3)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
        pset.renameArguments(ARG0="cpus", ARG1="memory", ARG2="workers")

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=pset)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr, pset=pset
        )
        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

    def evaluate(self, individual):
        func = self.toolbox.compile(expr=individual)
        errors = []
        for cpus, memory, workers, actual in self.data:
            predicted = func(cpus, memory, workers)
            errors.append((predicted - actual) ** 2)
        return (np.mean(errors),)

    def train(self, data):
        processed_data = preprocess_profiling_data(data)
        self.data = processed_data
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
        self.best_individual = hof[0]

    def predict(self, cpus, memory, workers):
        func = self.toolbox.compile(expr=self.best_individual)
        return func(cpus, memory, workers)

    def generate_objective_function(self):
        return str(self.best_individual)


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


if __name__ == "__main__":
    profiling_data = {
        (2, 400, 5): {
            "read": [
                [
                    0.13607573509216309,
                    0.11152410507202148,
                    0.12084269523620605,
                    0.12781190872192383,
                    0.10123634338378906,
                ],
                [
                    0.12205648422241211,
                    0.0889730453491211,
                    0.09708809852600098,
                    0.09705781936645508,
                    0.08011627197265625,
                ],
            ],
            "write": [
                [
                    0.21655988693237305,
                    0.313230037689209,
                    0.37512636184692383,
                    0.44702625274658203,
                    0.3391242027282715,
                ],
                [
                    0.2964909076690674,
                    0.32325196266174316,
                    0.2654855251312256,
                    0.24427175521850586,
                    0.2672877311706543,
                ],
            ],
            "compute": [
                [
                    0.6712195873260498,
                    0.6449689865112305,
                    0.6611623764038086,
                    0.680020809173584,
                    0.7658612728118896,
                ],
                [
                    0.663583517074585,
                    0.649554967880249,
                    0.6580414772033691,
                    0.6754159927368164,
                    0.7627818584442139,
                ],
            ],
            "cold_start_time": [
                [
                    2.6361682415008545,
                    2.7001962661743164,
                    2.8119142055511475,
                    2.8189210891723633,
                    3.142885684967041,
                ],
                [
                    2.5462496280670166,
                    2.6483700275421143,
                    2.767951011657715,
                    2.821345567703247,
                    2.9759883880615234,
                ],
            ],
        }
    }

    model = GAPerfModel()
    model.train(profiling_data)
    print("Objective Function:", model.generate_objective_function())
    prediction = model.predict(2, 400, 5)
    print("Predicted Latency for (2 CPUs, 400 Memory, 5 Workers):", prediction)
