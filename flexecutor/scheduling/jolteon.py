import types

import black
import numpy as np

from modelling.perfmodel import PerfModelEnum
from scheduling.orion import MyQueue
from scheduling.scheduler import Scheduler

workers_accessor = slice(0, None, 2)
cpu_accessor = slice(1, None, 2)


def _get_sampling_size(num_stages, risk, confidence_error):
    # Hoeffding's inequality
    # Is incongruently but maintained for compatibility with Jolteon
    # {0.5, 1, 1.5, 2, 3, 4} as the intra-function resource space, so the size is 8
    # {4, 8, 16, 32} as the parallelism space, so the size is 4
    search_space_size = (7 * 4) ** (num_stages // 2)  # num_X / 2 stages
    return int(
        np.ceil(1 / (2 * risk**2) * np.log(search_space_size / confidence_error))
    )


class Jolteon(Scheduler):

    def __init__(self, dag, total_parallelism: int, cpu_per_worker: float):
        super().__init__(dag, PerfModelEnum.MIXED)
        self.total_parallelism = total_parallelism
        self.cpu_per_worker = cpu_per_worker

        self.num_X = 2 * len(dag.stages)
        self.objective_fn = None
        self.constraint_fn = None

        # FIXME: please parametrise as needed

        self.bound = 40
        self.bound_type = "latency"  # "latency" or "cost"
        self.probe_depth = 4

        self.obj_params = None  # 2-dim array --> shape: (num_stages, coeffs)
        self.cons_params = None  # 3-dim array --> shape: (num_stages, coeffs, samples)

        # Used for probing
        self.cpu_search_space = np.array([0.6, 1, 1.5, 2, 2.5, 3, 4])
        self.workers_search_space = np.array([1, 4, 8, 16, 32])

        entry_point = [1, 3, 16, 3, 8, 3, 1, 3]
        self.x0 = entry_point if entry_point is not None else np.ones(self.num_X)

        x_bounds = [
            (1, 2),
            (0.5, 4.1),
            (4, 32),
            (0.5, 4.1),
            (4, 32),
            (0.5, 4.1),
            (1, 2),
            (0.5, 4.1),
        ]
        if isinstance(x_bounds, list):
            self.x_bounds = x_bounds
        else:
            x_tuple = x_bounds if isinstance(x_bounds, tuple) else (0.5, None)
            self.x_bounds = [x_tuple] * self.num_X

        # User-defined risk level (epsilon) for constraint satisfaction (e.g., 0.01 or 0.05)
        self.risk = 0.05

        # Confidence error (delta) for the lower bound property of the ground-truth optimal
        # objective value or the feasibility of the solution or both,
        # depending on the relationship between epsilon and alpha, default to 0.01
        self.confidence_error = 0.001

        # Function tolerance for the optimization solver
        self.fn_tol = self.risk * self.bound

    def schedule(self):
        # STEP 1: Use Hoeffding's inequality to determine the sampling size
        sample_size = _get_sampling_size(
            len(self._dag.stages), self.risk, self.confidence_error
        )

        # STEP 2: Extract the coefficients of the performance model
        self.obj_params = np.array(
            [stage.perf_model.parameters() for stage in self._dag.stages]
        )

        # STEP 3: Constraint Montecarlo sampling
        self.cons_params = np.array(
            [stage.perf_model.sample_offline(sample_size) for stage in self._dag.stages]
        ).transpose((0, 2, 1))

        # STEP 4: Generate the objective and constraint functions
        self._generate_func_code(
            # FIXME: allow parametrization
            self._dag.stages,
            None,
        )

        # STEP 5: Solve the optimization problem (gradient-based descent + probing)
        num_workers, num_cpu = self._iter_solve()
        num_workers, num_cpu = self._round_config(num_workers, num_cpu)
        num_workers, num_cpu = self._probe(num_workers, num_cpu)

        print(f"Num CPU: {num_cpu}")
        print(f"Num Func: {num_workers}")
        print("Jolteon PCPSolver finished as expected!")

    def _generate_func_code(self, critical_path, secondary_path=None):
        assert isinstance(critical_path, list)
        assert secondary_path is None or isinstance(secondary_path, list)
        obj_mode = "cost" if self.bound_type == "latency" else "latency"
        cons_mode = self.bound_type

        code = """
import numpy as np

def cpu(stage):
    return stage * 2 + 1

def workers(stage):
    return stage * 2
"""

        def _create_func_code(signature, objective, stages, bound=None) -> str:
            fn = signature + ":"
            fn += "\n   cold, x, kd_d, logx, x2, const = 0, 1, 2, 3, 4, 5"
            fn += "\n   return "
            for stage in stages:
                fn += stage.perf_model.generate_func_code(objective) + " + "
            fn = fn.removesuffix(" + ")
            fn += bound if bound is not None else ""
            fn += "\n\n"
            return fn

        # Generate objective function
        obj_stages = critical_path if obj_mode == "latency" else self._dag.stages
        obj_header = "def objective_func(config_list, coeffs_list)"
        code += _create_func_code(obj_header, obj_mode, obj_stages)

        # Generate constraint function(s)
        bound_value = " - b"
        cons_header = "def constraint_func(config_list, coeffs_list, b)"

        if cons_mode == "latency":
            cons2_header = "def constraint_func_2(config_list, coeffs_list, b)"
            code += _create_func_code(
                cons_header, cons_mode, critical_path, bound_value
            )
            if secondary_path is not None:
                code += _create_func_code(
                    cons2_header, cons_mode, secondary_path, bound_value
                )
        else:
            cons2_header = "def constraint_func_2(config_list, coeffs_list)"
            code += _create_func_code(
                cons_header, cons_mode, self._dag.stages, bound_value
            )
            # The time of the secondary path should be less than or equal to the time of the critical path
            if secondary_path is not None:
                c_s = set(critical_path) - set(secondary_path)
                s_c = set(secondary_path) - set(critical_path)
                assert len(c_s) > 0 and len(s_c) > 0
                bound_value = "("
                for s in s_c:
                    bound_value += f"{s.perf_model.generate_func_code('latency')} + "
                bound_value = bound_value.removesuffix(" + ") + ")"
                code += _create_func_code(cons2_header, cons_mode, c_s, bound_value)

        code = black.format_str(code, mode=black.FileMode())

        # code_dir = os.path.dirname(os.path.abspath(__file__))
        # code_path = os.path.join(code_dir, self._dag.dag_id + "_func.py")
        # with open(code_path, "w") as f:
        #     f.write(code)

        module_name = f"flexecutor.scheduling.{self._dag.dag_id}_func"
        module = types.ModuleType(module_name)
        exec(code, module.__dict__)

        self.objective_fn = getattr(module, "objective_func")
        self.constraint_fn = getattr(module, "constraint_func")

    def _round_config(self, num_workers, num_cpu):
        rounded_num_workers = []
        rounded_num_cpu = []
        for i, stage in enumerate(self._dag.stages):
            w = (
                next(
                    (p for p in self.workers_search_space if num_workers[i] <= p),
                    self.workers_search_space[-1],
                )
                if stage.perf_model.allow_parallel
                else 1
            )
            c = next(
                (v for v in self.cpu_search_space if num_cpu[i] < v),
                self.cpu_search_space[-1],
            )
            rounded_num_workers.append(w)
            rounded_num_cpu.append(c)
        return rounded_num_workers, rounded_num_cpu

    def _iter_solve(self) -> tuple[list, list]:
        bound = self.bound
        while True:
            import scipy.optimize as scipy_opt

            minimize_result = scipy_opt.minimize(
                lambda x: self.objective_fn(x, self.obj_params),
                self.x0,
                method="SLSQP",
                bounds=self.x_bounds,
                constraints={
                    "type": "ineq",
                    "fun": lambda x: -self.constraint_fn(x, self.cons_params, bound),
                },
                options={"ftol": self.fn_tol, "disp": False},
            )

            cons_val = np.array(
                self.constraint_fn(minimize_result.x, self.cons_params, bound)
            )
            ratio_not_satisfied = np.sum(cons_val > self.fn_tol) / len(cons_val)
            if ratio_not_satisfied < self.risk or minimize_result.success:
                break
            print("bound:", bound, "ratio:", ratio_not_satisfied)
            bound += self.fn_tol * 4

        return minimize_result.x[workers_accessor], minimize_result.x[cpu_accessor]

    def _probe(self, num_workers, num_cpu):
        def get_x_by_xpos(array):
            result = np.zeros(self.num_X)
            result[workers_accessor] = self.workers_search_space[
                array[workers_accessor]
            ]
            result[cpu_accessor] = self.cpu_search_space[array[cpu_accessor]]
            return result

        x_pos = np.zeros(self.num_X, dtype=int)
        x_pos[workers_accessor] = [
            np.where(self.workers_search_space == d)[0][0] for d in num_workers
        ]
        x_pos[cpu_accessor] = [
            np.where(self.cpu_search_space == k)[0][0] for k in num_cpu
        ]

        searched = set()

        def bfs(_x_pos, max_depth=4):
            queue = MyQueue()
            searched.add(tuple(_x_pos))
            queue.push((_x_pos, 0))

            _best_pos = _x_pos.copy()
            best_x = get_x_by_xpos(_best_pos)
            best_obj = self.objective_fn(best_x, self.obj_params)
            best_cons = self.constraint_fn(best_x, self.obj_params, self.bound)

            steps = [-1, 1]

            while len(queue) > 0:
                _x_pos, depth = queue.pop()

                _x = get_x_by_xpos(_x_pos)
                _cons = np.percentile(
                    self.constraint_fn(_x, self.cons_params, self.bound),
                    100 * (1 - self.risk),
                )
                obj = self.objective_fn(_x, self.obj_params)

                if (best_cons < 0 and 0 > _cons > best_cons and obj < best_obj) or (
                    best_cons > 0 and _cons < best_cons
                ):
                    _best_pos = _x.copy()
                    best_obj = obj
                    best_cons = _cons

                if depth < max_depth:
                    for t in range(self.num_X):
                        config = (
                            self.workers_search_space
                            if t % 2 == 0
                            else self.cpu_search_space
                        )
                        for s in steps:
                            new_x_pos = _x_pos.copy()
                            new_x_pos[t] += s
                            if (
                                new_x_pos[t] < 0
                                or new_x_pos[t] >= len(config)
                                or (t % 2 == 0 and new_x_pos[t] == 0)
                                or tuple(new_x_pos) in searched
                            ):
                                continue
                            searched.add(tuple(new_x_pos))
                            queue.push((new_x_pos, depth + 1))

            return _best_pos

        old_x_pos = x_pos.copy()
        while True:
            x = get_x_by_xpos(x_pos)
            cons = np.percentile(
                self.constraint_fn(x, self.cons_params, self.bound),
                100 * (1 - self.risk),
            )
            x_pos = bfs(x_pos, self.probe_depth)
            if np.all(x_pos == old_x_pos) or cons < 0:
                break
            old_x_pos = x_pos.copy()

        # Find the best solution
        best_pos = bfs(x_pos, self.probe_depth)
        best_workers = self.workers_search_space[best_pos[workers_accessor]].tolist()
        best_cpu = self.cpu_search_space[best_pos[cpu_accessor]].tolist()

        return best_workers, best_cpu
