import os
from typing import Callable

import numpy as np
from scipy.optimize import NonlinearConstraint

from modelling.perfmodel import PerfModelEnum
from scheduling.orion import MyQueue
from scheduling.scheduler import Scheduler
from workflow.dag import DAG


class Jolteon(Scheduler):

    def __init__(self, dag, total_parallelism: int, cpu_per_worker: float):
        super().__init__(dag, PerfModelEnum.MIXED)
        self.total_parallelism = total_parallelism
        self.cpu_per_worker = cpu_per_worker
        self.cpu_search_space = [0.6, 1, 1.5, 2, 2.5, 3, 4]
        self.workers_search_space = [1, 4, 8, 16, 32]

    def schedule(self):
        # FIXME: allow parametrization
        num_samples = 2715
        bound_type = "latency"
        # noInspection PyUnresolvedReferences
        samples = np.concatenate(
            [
                stage.perf_model.sample_offline(num_samples)
                for stage in self._dag.stages
            ],
            axis=1,
        )
        self._generate_func_code(
            "jolteon_func.py",
            # FIXME: allow parametrization
            list(self._dag.stages),
            None,
            bound_type,
            "scipy",
        )

        from flexecutor.scheduling.jolteon_func import objective_func, constraint_func

        parameters = np.concatenate(
            [stage.perf_model.parameters() for stage in self._dag.stages]
        )

        solver = PCPSolver(
            self._dag,
            objective_func,
            constraint_func,
            40,
            parameters.tolist(),
            samples.tolist(),
            self.cpu_search_space,
            self.workers_search_space,
        )

        # FIXME: these numbers are magic numbers in Jolteon (different per use case)
        # Review if we can destroy this shit
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

        res = solver.iter_solve(x_init, x_bound)

        num_workers = res["x"][0::2]
        num_cpu = res["x"][1::2]

        solver.bound = 40

        num_workers, num_cpu = self._round_config(num_workers, num_cpu)

        num_workers, num_cpu = solver.probe(num_workers, num_cpu)

        print(f"Num CPU: {num_cpu}")
        print(f"Num Func: {num_workers}")
        print("Jolteon PCPSolver finished as expected!")

    def _generate_func_code(
        self,
        file_name,
        critical_path,
        secondary_path=None,
        cons_mode="latency",
        solver_type="scipy",
    ):
        assert isinstance(file_name, str) and file_name.endswith(".py")
        assert isinstance(critical_path, list)
        assert secondary_path is None or isinstance(secondary_path, list)
        assert cons_mode in ["latency", "cost"]
        assert solver_type == "scipy"
        code_dir = os.path.dirname(os.path.abspath(__file__))
        code_path = os.path.join(code_dir, file_name)
        obj_mode = "cost" if cons_mode == "latency" else "latency"

        parent_ids = {}
        for stage in self._dag.stages:
            parent_ids[stage.stage_id] = -1

        for stage in self._dag.stages:
            if not stage.perf_model.allow_parallel:
                for parent in stage.parents:
                    if parent.perf_model.allow_parallel:
                        parent_ids[stage.stage_id] = parent.stage_id
                        break

        s = "import numpy as np\n\n"
        if solver_type == "pyomo":
            s += "import pyomo.environ as pyo\n"
            s += "from pyomo.environ import *\n\n"

        # Generate objective function
        var = "x"
        param = "p"
        if solver_type == "scipy":
            s += "def objective_func(x, p):\n" + "    return "
        else:
            s += "def objective_func(model):\n" + "    return "
            var = "model.x"
            param = "model.p"

        if obj_mode == "latency":
            for stage in critical_path:
                s += (
                    stage.perf_model.generate_func_code(
                        obj_mode, var, param, parent_ids[stage.stage_id], solver_type
                    )
                    + " + "
                )
        else:
            for stage in self._dag.stages:
                s += (
                    stage.perf_model.generate_func_code(
                        obj_mode, var, param, parent_ids[stage.stage_id], solver_type
                    )
                    + " + "
                )
        s = s[:-3]
        s += "\n\n"

        # Generate constraints
        bound = " - b"
        func2_def = "def constraint_func_2(x, p, b):\n" + "    return "
        if solver_type == "scipy":
            s += "def constraint_func(x, p, b):\n" + "    return "
            if cons_mode == "cost":
                func2_def = "def constraint_func_2(x, p):\n" + "    return "
        else:
            s += "def constraint_func(model):\n" + "    return "
            bound = " - model.b <= 0"
            func2_def = "def constraint_func_2(model):\n" + "    return "

        if cons_mode == "latency":
            for stage in critical_path:
                s += (
                    stage.perf_model.generate_func_code(
                        cons_mode, var, param, parent_ids[stage.stage_id], solver_type
                    )
                    + " + "
                )
            s = s[:-3]
            s += bound + "\n\n"
            if secondary_path is not None:
                s += func2_def
                for stage in secondary_path:
                    s += (
                        stage.perf_model.generate_func_code(
                            cons_mode,
                            var,
                            param,
                            parent_ids[stage.stage_id],
                            solver_type,
                        )
                        + " + "
                    )
                s = s[:-3]
                s += bound + "\n\n"
        else:
            for stage in self._dag.stages:
                s += (
                    stage.perf_model.generate_func_code(
                        cons_mode, var, param, parent_ids[stage.stage_id], solver_type
                    )
                    + " + "
                )
            s = s[:-3]
            s += bound + "\n\n"
            # The time of the secondary path should be less than or equal to the time of the critical path
            if secondary_path is not None:
                s += func2_def
                critical_set = set(critical_path)
                secondary_set = set(secondary_path)
                c_s = critical_set - secondary_set
                s_c = secondary_set - critical_set
                assert len(c_s) > 0 and len(s_c) > 0
                for stage in c_s:
                    s += (
                        stage.perf_model.generate_func_code(
                            "latency",
                            var,
                            param,
                            parent_ids[stage.stage_id],
                            solver_type,
                        )
                        + " + "
                    )
                s = s[:-3] + " - ("
                for stage in s_c:
                    s += (
                        stage.perf_model.generate_func_code(
                            "latency",
                            var,
                            param,
                            parent_ids[stage.stage_id],
                            solver_type,
                        )
                        + " + "
                    )
                s = s[:-3] + ")"
                if solver_type == "pyomo":
                    s += " <= 0"
                s += "\n\n"

        with open(code_path, "w") as f:
            f.write(s)

    def _round_config(self, num_workers, num_cpu):
        rounded_num_workers = []
        rounded_num_cpu = []
        # FIXME: check order in iteration
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
        objective_params: list,
        constraint_params: list,
        cpu_search_space: list,
        workers_search_space: list,
    ):

        self.num_X = 2 * len(dag.stages)
        self.objective = objective_func
        self.constraint = constraint_func
        self.bound = bound
        self.obj_params = objective_params
        self.cons_params = constraint_params

        # used for probing
        self.cpu_search_space = cpu_search_space
        self.workers_search_space = workers_search_space

        # FIXME: please rethink if we need to extract the values of next variables to user-defined parameters site

        # User-defined risk level (epsilon) for constraint satisfaction (e.g., 0.01 or 0.05)
        self.risk = 0.05
        # Approximated risk level (alpha), a parameter for the sample approximation problem
        self.approx_risk = 0
        # Confidence error (delta) for the lower bound property of the ground-truth optimal
        # objective value or the feasibility of the solution or both,
        # depending on the relationship between epsilon and alpha, default to 0.01
        self.confidence_error = 0.001

        # Used for solving
        self.ftol = self.risk * self.bound

        self.bound_type = "latency"
        self.need_probe = None
        self.probe_depth = 4

        # Solver information for the sample approximation problem
        self.solver_info = {"optlib": "scipy", "method": "SLSQP"}

    def iter_solve(self, init_vals=None, x_bound=None):
        while True:
            res = self._solve(init_vals=init_vals, x_bound=x_bound)
            if res["status"]:
                break
            else:
                cons_val = np.array(res["cons_val"])
                ratio_not_satisfied = np.sum(cons_val > self.ftol) / len(cons_val)
                if ratio_not_satisfied < self.risk:
                    break
                else:
                    print("bound:", self.bound, "ratio:", ratio_not_satisfied)
                    self.bound += self.ftol * 4

        return res

    def _solve(self, init_vals=None, x_bound=None) -> dict:
        import scipy.optimize as scipy_opt

        assert self.solver_info["optlib"] == "scipy"
        assert self.solver_info["method"] == "SLSQP"

        x0 = np.ones(self.num_X) * 2  # initial guess
        if init_vals is not None:
            if isinstance(init_vals, int) or isinstance(init_vals, float):
                x0 = np.ones(self.num_X) * init_vals
            elif isinstance(init_vals, list) and len(init_vals) == self.num_X:
                # ssert all(isinstaance(x, int) or isinstance(x, float) for x in init_vals)
                x0 = np.array(init_vals)
            elif isinstance(init_vals, np.ndarray) and init_vals.shape == (self.num_X,):
                x0 = init_vals

        X_bounds = [
            (0.5, None) for _ in range(self.num_X)
        ]  # optional bounds for each x
        if x_bound is not None:
            if isinstance(x_bound, tuple) and len(x_bound) == 2:
                X_bounds = [x_bound for _ in range(self.num_X)]
            elif isinstance(x_bound, list) and len(x_bound) == self.num_X:
                X_bounds = x_bound
            elif isinstance(x_bound, list) and len(x_bound) == 2:
                # [0] is for parallelism, [1] is for intra-function resource
                X_bounds = []
                for _ in range(self.num_X // 2):
                    X_bounds.append(x_bound[0])
                    X_bounds.append(x_bound[1])

        obj_params = np.array(self.obj_params)
        cons_params = np.array(self.cons_params).T
        nonlinear_constraints = NonlinearConstraint(
            lambda x: self.constraint(x, cons_params, self.bound), -np.inf, 0
        )

        res = scipy_opt.minimize(
            lambda x: self.objective(x, obj_params),
            x0,
            method=self.solver_info["method"],
            bounds=X_bounds,
            constraints=nonlinear_constraints,
            options={"ftol": self.ftol, "disp": False},
        )

        solve_res = {}
        solve_res["status"] = res.success
        solve_res["obj_val"] = res.fun
        solve_res["cons_val"] = self.constraint(res.x, cons_params, self.bound)
        solve_res["x"] = res.x

        return solve_res

    def probe(self, d_init, k_init):
        # assume init is within the feasible region
        d_pos = []
        k_pos = []
        d_config = np.array(self.workers_search_space)
        k_config = np.array(self.cpu_search_space)
        for d in d_init:
            mask = d_config == d
            if np.any(mask):
                j = np.where(mask)[0][0]
                d_pos.append(j)
        for k in k_init:
            mask = k_config == k
            if np.any(mask):
                j = np.where(mask)[0][0]
                k_pos.append(j)

        d_pos = np.array(d_pos)
        k_pos = np.array(k_pos)
        x_pos = np.zeros(self.num_X, dtype=int)
        x_pos[0::2] = d_pos
        x_pos[1::2] = k_pos

        cons_params = np.array(self.cons_params).T

        searched = set()

        need_pos = []
        if self.need_probe is not None:
            need_pos = [i for i in range(self.num_X) if self.need_probe[i]]

        def bfs(x_pos, max_depth=4):
            q = MyQueue()
            searched.add(tuple(x_pos))
            q.push([x_pos, 0])

            best_pos = x_pos.copy()
            best_x = np.zeros(self.num_X)
            best_x[0::2] = d_config[best_pos[0::2]]
            best_x[1::2] = k_config[best_pos[1::2]]
            best_obj = self.objective(best_x, self.obj_params)
            best_cons = self.constraint(best_x, self.obj_params, self.bound)

            steps = [-1, 1]

            while len(q) > 0:
                p = q.pop()

                x = np.zeros(self.num_X)
                x[0::2] = d_config[p[0][0::2]]
                x[1::2] = k_config[p[0][1::2]]
                cons = self.constraint(x, cons_params, self.bound)
                cons = np.percentile(cons, 100 * (1 - self.risk))
                obj = self.objective(x, self.obj_params)
                # print('x:', x, 'obj:', obj, 'cons:', cons)

                if best_cons < 0:  # tight bound
                    if cons < 0 and cons > best_cons and obj < best_obj:
                        best_pos = p[0].copy()
                        best_obj = obj
                        best_cons = cons
                else:  # find a feasible solution first
                    if cons < best_cons:
                        best_pos = p[0].copy()
                        best_obj = obj
                        best_cons = cons

                if p[1] < max_depth:
                    for t in range(self.num_X):
                        if len(need_pos) > 0 and t not in need_pos:
                            continue
                        if t % 2 == 0:  # d
                            config = d_config
                        else:  # k
                            config = k_config
                        for s in steps:
                            new_x_pos = p[0].copy()
                            new_x_pos[t] += s
                            if (
                                new_x_pos[t] < 0
                                or new_x_pos[t] >= len(config)
                                or (t % 2 == 0 and new_x_pos[t] == 0)
                            ):
                                continue
                            if tuple(new_x_pos) in searched:
                                continue
                            searched.add(tuple(new_x_pos))
                            q.push([new_x_pos, p[1] + 1])

            return best_pos

        x = np.zeros(self.num_X)
        x[0::2] = d_config[x_pos[0::2]]
        x[1::2] = k_config[x_pos[1::2]]
        cons = self.constraint(x, cons_params, self.bound)
        cons = np.percentile(cons, 100 * (1 - self.risk))
        feasible = cons < 0
        old_x_pos = x_pos.copy()
        while not feasible:  # find a feasible solution first
            x_pos = bfs(x_pos, self.probe_depth)
            if x_pos.tolist() == old_x_pos.tolist():  # no improvement
                break
            old_x_pos = x_pos.copy()
            x = np.zeros(self.num_X)
            x[0::2] = d_config[x_pos[0::2]]
            x[1::2] = k_config[x_pos[1::2]]
            cons = self.constraint(x, self.obj_params, self.bound)
            cons = self.constraint(x, cons_params, self.bound)
            cons = np.percentile(cons, 100 * (1 - self.risk))
            feasible = cons < 0

        # find the best solution
        best_pos = bfs(x_pos, self.probe_depth)
        best_d = d_config[best_pos[0::2]].tolist()
        best_k = k_config[best_pos[1::2]].tolist()

        return best_d, best_k
