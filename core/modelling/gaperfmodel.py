import operator
import random
import numpy as np
import json
from deap import algorithms, base, creator, tools, gp


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
        errors = [
            (func(mem, cpus, workers) - actual) ** 2
            for mem, cpus, workers, actual in self.data
        ]
        return (np.mean(errors),)

    def train(self, data):
        self.data = data
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

    def predict(self, memory, cpus, workers):
        func = self.toolbox.compile(expr=self.best_individual)
        return func(memory, cpus, workers)

    def generate_objective_function(self):
        return str(self.best_individual)


if __name__ == "__main__":
    model = GAPerfModel()

    data = json.loads()
