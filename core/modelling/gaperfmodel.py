import numpy as np
import operator
import random
import pickle
import logging
from deap import creator, base, tools, gp, algorithms
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def protected_div(left, right):
    return left / right if right != 0 else 1e10


def rand101():
    return random.randint(-1, 1)


# TODO: Find a way to always return the same objective function
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
        pset.addEphemeralConstant("rand101", rand101)
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
            try:
                predicted = func(cpus, memory, workers)
                if not np.isfinite(predicted) or predicted > 100 or predicted <= 0:
                    penalty = 1e10
                else:
                    penalty = (predicted - actual) ** 2
                errors.append(penalty)
            except Exception as e:
                logger.error(f"Error evaluating individual: {e}")
                errors.append(1e10)
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

    def predict_latency(self, cpus, memory, workers):
        func = self.toolbox.compile(expr=self.best_individual)
        try:
            return func(cpus, memory, workers)
        except Exception as e:
            logger.error(f"Error predicting configuration: {e}")
            return np.nan

    def get_objective_function(self):
        compiled_func = self.toolbox.compile(expr=self.best_individual)

        def objective_func(x):
            cpus, memory, workers = np.round(x).astype(int)
            try:
                value = compiled_func(cpus, memory, workers)
                if not np.isfinite(value):
                    raise ValueError("Non-finite value")
                if value > 100 or value <= 0:
                    return 1e10
                return value
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 1e10

        return objective_func

    def save_model(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.best_individual, file)

    def load_model(self, filename):
        with open(filename, "rb") as file:
            self.best_individual = pickle.load(file)

    def get_objective_function(self):
        logger.info(f"Objective function: {self.best_individual}")
        compiled_func = self.toolbox.compile(expr=self.best_individual)

        def objective_func(x):
            cpus, memory, workers = np.round(x).astype(int)
            try:
                value = compiled_func(cpus, memory, workers)
                logger.info(
                    f"Objective function evaluated with (cpus={cpus}, memory={memory}, workers={workers}): value={value}"
                )
                if not np.isfinite(value):
                    raise ValueError("Non-finite value")
                if value > 100 or value <= 0:
                    penalty = 1e10
                else:
                    penalty = 0
                return value + penalty
            except Exception as e:
                logger.error(
                    f"Error in objective function with (cpus={cpus}, memory={memory}, workers={workers}): {e}"
                )
                return 1e10

        return objective_func


# def preprocess_profiling_data(profiling_data):
#     processed_data = []
#     for config, executions in profiling_data.items():
#         cpus, memory, workers = config
#         all_run_latencies = []
#         all_cold_starts = []

#         for run in executions:
#             run_latencies = [
#                 worker_data["read"] + worker_data["compute"] + worker_data["write"]
#                 for worker_data in run
#             ]
#             run_cold_starts = [worker_data["cold_start_time"] for worker_data in run]

#             # Considerare que la latencia del run es la latencia del worker con mayor latencia
#             max_latency = max(run_latencies)
#             # mediana de coldstarts, por alguna razon en el profiling el dato de coldstart varia demasiado
#             median_cold_start = np.median(run_cold_starts)

#             total_latency = max_latency + median_cold_start
#             all_run_latencies.append(total_latency)
#             all_cold_starts.extend(run_cold_starts)

#         # Hay stagglers cuando hacemos profiling, la idea con esto es escoger percentiles no utilizar stagglers
#         q1 = np.percentile(all_run_latencies, 25)
#         q3 = np.percentile(all_run_latencies, 75)
#         iqr = q3 - q1
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr

#         filtered_latencies = [
#             latency
#             for latency in all_run_latencies
#             if lower_bound <= latency <= upper_bound
#         ]

#         for latency in filtered_latencies:
#             processed_data.append((cpus, memory, workers, latency))

#     return processed_data


def preprocess_profiling_data(profiling_data):
    processed_data = []
    for config, executions in profiling_data.items():
        cpus, memory, workers = config
        for run in executions:
            run_latencies = [
                worker_data["read"] + worker_data["compute"] + worker_data["write"]
                for worker_data in run
            ]
            average_latency = np.mean(run_latencies)

            run_cold_starts = [worker_data["cold_start_time"] for worker_data in run]
            median_cold_start = np.median(run_cold_starts)

            total_latency = average_latency + median_cold_start
            processed_data.append((cpus, memory, workers, total_latency))

    return processed_data


if __name__ == "__main__":
    profiling_data = {
        (2, 400, 5): [
            [
                {
                    "read": 0.13607573509216309,
                    "compute": 0.6712195873260498,
                    "write": 0.21655988693237305,
                    "cold_start_time": 2.6361682415008545,
                },
                {
                    "read": 0.11152410507202148,
                    "compute": 0.6449689865112305,
                    "write": 0.313230037689209,
                    "cold_start_time": 2.7001962661743164,
                },
                {
                    "read": 0.12084269523620605,
                    "compute": 0.6611623764038086,
                    "write": 0.37512636184692383,
                    "cold_start_time": 2.8119142055511475,
                },
                {
                    "read": 0.12781190872192383,
                    "compute": 0.680020809173584,
                    "write": 0.44702625274658203,
                    "cold_start_time": 2.8189210891723633,
                },
                {
                    "read": 0.10123634338378906,
                    "compute": 0.7658612728118896,
                    "write": 0.3391242027282715,
                    "cold_start_time": 3.142885684967041,
                },
            ],
            [
                {
                    "read": 0.12205648422241211,
                    "compute": 0.663583517074585,
                    "write": 0.2964909076690674,
                    "cold_start_time": 2.5462496280670166,
                },
                {
                    "read": 0.0889730453491211,
                    "compute": 0.649554967880249,
                    "write": 0.32325196266174316,
                    "cold_start_time": 2.6483700275421143,
                },
                {
                    "read": 0.09708809852600098,
                    "compute": 0.6580414772033691,
                    "write": 0.2654855251312256,
                    "cold_start_time": 2.767951011657715,
                },
                {
                    "read": 0.09705781936645508,
                    "compute": 0.6754159927368164,
                    "write": 0.24427175521850586,
                    "cold_start_time": 2.821345567703247,
                },
                {
                    "read": 0.08011627197265625,
                    "compute": 0.7627818584442139,
                    "write": 0.2672877311706543,
                    "cold_start_time": 2.9759883880615234,
                },
            ],
        ]
    }

    model = GAPerfModel()
    model.train(profiling_data)
    print("Objective Function:", model.get_objective_function())
    prediction = model.predict(2, 400, 5)
    print("Predicted Latency for (2 CPUs, 400 Memory, 5 Workers):", prediction)
