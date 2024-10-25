import os
from typing import Callable, Optional, Union

import black
import numpy as np

from modelling.perfmodel import PerfModelEnum
from scheduling.orion import MyQueue
from scheduling.scheduler import Scheduler
from workflow.dag import DAG

workers_accessor = slice(0, None, 2)
cpu_accessor = slice(1, None, 2)


class Jolteon(Scheduler):

    def __init__(self, dag, total_parallelism: int, cpu_per_worker: float):
        super().__init__(dag, PerfModelEnum.MIXED)
        self.total_parallelism = total_parallelism
        self.cpu_per_worker = cpu_per_worker
        self.cpu_search_space = [0.6, 1, 1.5, 2, 2.5, 3, 4]
        self.workers_search_space = [1, 4, 8, 16, 32]

    def schedule(self):
        def get_sample_size(num_stages, risk, confidence_error):
            # Hoeffding's inequality
            # Is incongruently but maintained for compatibility with Jolteon
            # {0.5, 1, 1.5, 2, 3, 4} as the intra-function resource space, so the size is 8
            # {4, 8, 16, 32} as the parallelism space, so the size is 4
            search_space_size = (7 * 4) ** (num_stages // 2)  # num_X / 2 stages
            return int(
                np.ceil(
                    1 / (2 * risk**2) * np.log(search_space_size / confidence_error)
                )
            )

        sample_size = get_sample_size(len(self._dag.stages), 0.05, 0.001)
        print(f"Sample size: {sample_size}")

        # FIXME: allow parametrization
        bound_type = "latency"
        # noInspection PyUnresolvedReferences
        samples = np.array(
            [stage.perf_model.sample_offline(sample_size) for stage in self._dag.stages]
        )

        self._generate_func_code(
            # FIXME: allow parametrization
            self._dag.stages,
            None,
            bound_type,
        )

        from flexecutor.scheduling.machine_learning_func import (
            objective_func,
            constraint_func,
        )

        parameters = np.array(
            [stage.perf_model.parameters() for stage in self._dag.stages]
        )

        x_init = [1, 3, 16, 3, 8, 3, 1, 3]
        x_bound = [
            (1, 2),
            (0.5, 4.1),
            (4, 32),
            (0.5, 4.1),
            (4, 32),
            (0.5, 4.1),
            (1, 2),
            (0.5, 4.1),
        ]

        solver = PCPSolver(
            self._dag,
            objective_func,
            constraint_func,
            40,
            bound_type,
            parameters,
            samples,
            self.cpu_search_space,
            self.workers_search_space,
            x_init,
            x_bound,
        )

        num_workers, num_cpu = solver.iter_solve()
        num_workers, num_cpu = self._round_config(num_workers, num_cpu)
        num_workers, num_cpu = solver.probe(num_workers, num_cpu)

        print(f"Num CPU: {num_cpu}")
        print(f"Num Func: {num_workers}")
        print("Jolteon PCPSolver finished as expected!")

    def _generate_func_code(
        self,
        critical_path,
        secondary_path=None,
        cons_mode="latency",
    ):
        assert isinstance(critical_path, list)
        assert secondary_path is None or isinstance(secondary_path, list)
        assert cons_mode in ["latency", "cost"]
        code_dir = os.path.dirname(os.path.abspath(__file__))
        code_path = os.path.join(code_dir, self._dag.dag_id + "_func.py")
        obj_mode = "cost" if cons_mode == "latency" else "latency"

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
        with open(code_path, "w") as f:
            f.write(code)

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


class PCPSolver:
    def __init__(
        self,
        dag: DAG,
        objective_func: Callable,
        constraint_func: Callable,
        bound: float,
        bound_type: str,
        objective_params: np.ndarray,
        constraint_params: np.ndarray,
        cpu_search_space: list,
        workers_search_space: list,
        entry_point: Optional[list],
        x_bounds: Union[list | tuple | None],
    ):
        self.num_X = 2 * len(dag.stages)
        self.objective = objective_func
        self.constraint = constraint_func
        self.bound = bound
        self.obj_params = (
            objective_params  # 2-dim array --> shape: (num_stages, coeffs)
        )
        self.cons_params = np.transpose(
            constraint_params, (0, 2, 1)
        )  # 3-dim array --> shape: (num_stages, coeffs, samples)

        # used for probing
        self.cpu_search_space = cpu_search_space
        self.workers_search_space = workers_search_space

        self.x0 = entry_point if entry_point is not None else np.ones(self.num_X)
        if isinstance(x_bounds, list):
            self.x_bounds = x_bounds
        else:
            x_tuple = x_bounds if isinstance(x_bounds, tuple) else (0.5, None)
            self.x_bounds = [x_tuple] * self.num_X

        # FIXME: please rethink if we need to extract the values of next variables to user-defined parameters site

        # User-defined risk level (epsilon) for constraint satisfaction (e.g., 0.01 or 0.05)
        self.risk = 0.05
        # Confidence error (delta) for the lower bound property of the ground-truth optimal
        # objective value or the feasibility of the solution or both,
        # depending on the relationship between epsilon and alpha, default to 0.01
        self.confidence_error = 0.001

        # Used for solving
        self.ftol = self.risk * self.bound

        self.bound_type = bound_type
        self.probe_depth = 4

    def iter_solve(self) -> tuple[list, list]:
        bound = self.bound
        while True:
            import scipy.optimize as scipy_opt

            minimize_result = scipy_opt.minimize(
                lambda x: self.objective(x, self.obj_params),
                self.x0,
                method="SLSQP",
                bounds=self.x_bounds,
                constraints={
                    "type": "ineq",
                    "fun": lambda x: -self.constraint(x, self.cons_params, bound),
                },
                options={"ftol": self.ftol, "disp": False},
            )

            cons_val = np.array(
                self.constraint(minimize_result.x, self.cons_params, bound)
            )
            ratio_not_satisfied = np.sum(cons_val > self.ftol) / len(cons_val)
            if ratio_not_satisfied < self.risk or minimize_result.success:
                break
            print("bound:", bound, "ratio:", ratio_not_satisfied)
            bound += self.ftol * 4

        return minimize_result.x[workers_accessor], minimize_result.x[cpu_accessor]

    def probe(self, num_workers, num_cpu):
        def get_x_by_xpos(array):
            result = np.zeros(self.num_X)
            result[workers_accessor] = d_config[array[workers_accessor]]
            result[cpu_accessor] = k_config[array[cpu_accessor]]
            return result

        d_config = np.array(self.workers_search_space)
        k_config = np.array(self.cpu_search_space)
        x_pos = np.zeros(self.num_X, dtype=int)
        x_pos[workers_accessor] = [np.where(d_config == d)[0][0] for d in num_workers]
        x_pos[cpu_accessor] = [np.where(k_config == k)[0][0] for k in num_cpu]

        searched = set()

        def bfs(_x_pos, max_depth=4):
            queue = MyQueue()
            searched.add(tuple(_x_pos))
            queue.push((_x_pos, 0))

            _best_pos = _x_pos.copy()
            best_x = get_x_by_xpos(_best_pos)
            best_obj = self.objective(best_x, self.obj_params)
            best_cons = self.constraint(best_x, self.obj_params, self.bound)

            steps = [-1, 1]

            while len(queue) > 0:
                _x_pos, depth = queue.pop()

                _x = get_x_by_xpos(_x_pos)
                _cons = np.percentile(
                    self.constraint(_x, self.cons_params, self.bound),
                    100 * (1 - self.risk),
                )
                obj = self.objective(_x, self.obj_params)

                if (best_cons < 0 and 0 > _cons > best_cons and obj < best_obj) or (
                    best_cons > 0 and _cons < best_cons
                ):
                    _best_pos = _x.copy()
                    best_obj = obj
                    best_cons = _cons

                if depth < max_depth:
                    for t in range(self.num_X):
                        config = d_config if t % 2 == 0 else k_config
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
                self.constraint(x, self.cons_params, self.bound), 100 * (1 - self.risk)
            )
            x_pos = bfs(x_pos, self.probe_depth)
            if np.all(x_pos == old_x_pos) or cons < 0:
                break
            old_x_pos = x_pos.copy()

        # Find the best solution
        best_pos = bfs(x_pos, self.probe_depth)
        best_workers = d_config[best_pos[workers_accessor]].tolist()
        best_cpu = k_config[best_pos[cpu_accessor]].tolist()

        return best_workers, best_cpu
