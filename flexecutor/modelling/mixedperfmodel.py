from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from modelling.perfmodel import PerfModel
from utils.dataclass import StageConfig, FunctionTimes


class ModelStrategy(ABC):
    """
    The ModelStrategy is used to define the strategy to be used to model the performance
    """

    def __init__(self, mixed_model: "MixedPerfModel"):
        self._model = mixed_model

    @abstractmethod
    def run(self) -> None:
        pass


class GetAndSet:
    def get(self, attr: str):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(
                f"Attribute {attr} does not exist in the class MixedModelCoefficients"
            )

    def set(self, attr: str, value: float):
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            raise AttributeError(
                f"Attribute {attr} does not exist in the class MixedModelCoefficients"
            )


@dataclass
class ModelParams(GetAndSet):
    """
    The average parameters of the performance model
    - cold: array with the coefficients of the cold start phase
    - read: array with the coefficients of the read phase
    - compute: array with the coefficients of the compute phase
    - write: array with the coefficients of the write phase
    """

    read: np.array
    compute: np.array
    write: np.array
    cold: np.array

    def __init__(self):
        self.read = np.array([])
        self.compute = np.array([])
        self.write = np.array([])
        self.cold = np.array([])


class ModelCovariance(ModelParams):
    """
    The covariance of the average parameters of the performance model
    - cold: covariance of the cold start phase
    - read: covariance of the read phase
    - compute: covariance of the compute phase
    - write: covariance of the write phase
    """

    pass


@dataclass
class MixedModelCoefficients(GetAndSet):
    """
    The coefficients of the mixed performance model

    The dimension of the parameters is reduced from 8 to 5 (mixing common degree
    coeffs between phases), excluding cold start
    By merging the parameters of read, compute, and write as follows:
    - allow_parallel: a/d + b/(kd) + c*log(x)/x + e/x**2 + f, x can be d or kd
    - not allow_parallel: a/k + b*d + c*log(k)/k + e/k**2 + f
    """

    cold: float  # _ --> cold start time
    x: float  # a --> coefficient of 1/d or 1/k
    kd_d: float  # b --> coefficient of 1/(kd) or d
    logx: float  # c --> coefficient of log(x)/x, x can be d or kd
    x2: float  # e --> coefficient of 1/x**2, x can be d or kd
    const: float  # f --> constant coefficient

    def __init__(self):
        self.cold = 0
        self.x = 0
        self.kd_d = 0
        self.logx = 0
        self.x2 = 0
        self.const = 0

    def __array__(self):
        return np.array([self.cold, self.x, self.kd_d, self.logx, self.x2, self.const])


@dataclass
class CanIntraParallel(GetAndSet):
    """
    The can_intra_parallel is used to check if the phase can be parallelized
    """

    read: bool
    compute: bool
    write: bool

    def __init__(self):
        self.read = False
        self.compute = False
        self.write = False


def eq_vcpu_alloc(mem, num_func):
    """
    The eq_vcpu_alloc is used to convert the memory to vCPU (Lambda fix rate)
    Function inherited from Jolteon
    """
    return round((mem / 1792) * num_func, 1)


def io_func(x, a, b):
    """
    The io_func is used to model the read and write phases of the stage
    Note that the form is a*(1/x) + b
    @param x: array with the computational resource. Can be:
        - k: number of individual cpu units (per worker) --> Only for not allow_parallel
        - kd: number of total cpu units
        - d: number of workers
    @param a: variable coefficient for 1/x
    @param b: the constant coefficient
    @return: the time taken for the phase
    """
    return a / x + b


def io_func_pr(_input, a, b, c):
    """
    io_func2 is used to model the read parent_relevant phase
    Note that the form is a*(1/x) + b*y + c
    @param _input: two-dim array with (specific case):
        _input[0] (x): number of individual cpu units (per worker)
        _input[1] (y): number of workers
    @param a: variable coefficient for 1/x
    @param b: variable coefficient for y
    @param c: the constant coefficient
    @return: the time taken for the phase
    """
    x = _input[0]
    y = _input[1]
    return a / x + b * y + c


def comp_func(x, a, b, c, d):
    """
    The comp_func is used to model the compute phase of the stage
    Two different complexities are considered:
        - logarithmic complexity
        - quadratic complexity
    So, the curve is more adaptable to the real data, being aware of the different complexities
    Note that the form is a*(1/x) + b*log(x)/x + c/x**2 + d
    @param x: array with the computational resource. Can be:
        - k: number of individual cpu units (per worker) --> Only for not allow_parallel
        - kd: number of total cpu units
        - d: number of workers
    @param a: variable coefficient for 1/x
    @param b: variable coefficient for log(x)/x
    @param c: variable coefficient for 1/x**2
    @param d: the constant coefficient
    @return: the time taken for the phase
    """
    return a / x + b * np.log(x) / x + c / x**2 + d


def curve_fit(func, x, y, dims):
    return scipy_opt.curve_fit(func, x, y)[0:dims]


class MixedPerfModel(PerfModel, GetAndSet):
    """
    Mixed performance model that combines the white-box and black-box modelling
    Here, definitions of the notations used in the model:
    - y_s: array with cold start times
    - self.y_r: array with read times
    - y_c: array with compute times
    - y_w: array with write times
    - d: array with number of workers
    - kd: array with number of total cpu units
    - k: array with number of individual cpu units (per worker)
    - k_d: two-dim array with:
        k_d[0]: number of individual cpu units (per worker)
        k_d[1]: number of workers
    - can_intra_parallel:
        if True, we can parallelize the phase (read|compute|write) of the stage
        else otherwise
    - allow_parallel:
        if True, the stage can be parallelized
        else the stage cannot be parallelized (only 1 worker is allowed) for this stage
    - parent_relevant: this attribute deserves a better explanation:
        when profiling, not allow_parallel stages also are profiled with multiple workers
        but the result of the optimization will only output 1 worker
        so, parent_relevant is used to check if the read time of the stages depend on:
            only the number of cpu of the worker (parent_relevant = False)
            or the number of cpu of the worker and the number of workers (parent_relevant = True)
    """

    def __init__(self, stage):
        super().__init__("mixed", stage)

        self.can_intra_parallel = CanIntraParallel()
        self.parent_relevant = False

        self.params_avg = ModelParams()
        self.cov_avg = ModelCovariance()

        self.y_s = np.array([])
        self.y_r = np.array([])
        self.y_c = np.array([])
        self.y_w = np.array([])
        self.d = np.array([])
        self.kd = np.array([])
        self.k = np.array([])
        self.k_d = np.empty((0, 2))

        self.coeffs = MixedModelCoefficients()

    def _set_coeff_by_params(self, coeff, params, make_error=False):
        # IMPORTANT: Jolteon contains 1 huge error in its pkysus/Jolteon src repo
        # In not parallel: compute and write const times aren't considered in sampling
        # make_error is a trick to adapt to that
        # Please, remove it when the time arrives
        coeff.cold = params.cold
        coeff.logx += params.compute[1]
        coeff.x2 += params.compute[2]
        if self.allow_parallel:
            for phase in ["read", "compute", "write"]:
                if self.can_intra_parallel.get(phase):
                    coeff.kd_d += params.get(phase)[0]
                else:
                    coeff.x += params.get(phase)[0]
            coeff.const += params.read[1] + params.compute[3] + params.write[1]
        else:
            for phase in ["read", "compute", "write"]:
                coeff.x += params.get(phase)[0]
            if self.parent_relevant:
                coeff.kd_d += params.read[1]
                coeff.const += params.read[2]
            else:
                coeff.const += params.read[1]
            if not make_error:
                coeff.const += params.compute[3] + params.write[1]

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        # STEP 1: Populate the data
        self._populate_data(stage_profile_data)

        # STEP 2: Calculate the average parameters and covariance
        self._calc_params_and_covariances()

        # STEP 3: Compute the average coefficients
        self._set_coeff_by_params(self.coeffs, self.params_avg)

        # STEP 4: Print the accuracy of the model
        self._evaluate_model()

    def _evaluate_model(self):
        y_actual = self.y_r + self.y_c + self.y_w + self.y_s
        if self.allow_parallel:
            y_pred = (
                self.coeffs.x / self.d
                + self.coeffs.kd_d / self.kd
                + self.coeffs.const
                + np.mean(self.y_s)
            )
            if self.can_intra_parallel.compute:
                y_pred += self.coeffs.logx * np.log(self.kd) / self.kd
                y_pred += self.coeffs.x2 / self.kd**2
            else:
                y_pred += self.coeffs.logx * np.log(self.d) / self.d
                y_pred += self.coeffs.x2 / self.d**2
        else:
            y_pred = (
                self.coeffs.x / self.k
                + self.coeffs.kd_d * self.k_d[1]
                + self.coeffs.const
                + np.mean(self.y_s)
                + self.coeffs.logx * np.log(self.k) / self.k
                + self.coeffs.x2 / self.k**2
            )
        print(f"### {self._model_name} ### ")
        err = (y_pred - y_actual) / y_actual
        s_err = np.mean(np.abs(err))
        m_err = np.mean(err)
        print(
            "Stage {} mean abs error:".format(self._stage_name),
            "%.2f" % (s_err * 100),
            "%",
        )
        print(
            "Stage {} mean error:".format(self._stage_name),
            "%.2f" % (m_err * 100),
            "%",
        )
        print()

    def _calc_params_and_covariances(self):
        rw_parallel_choices = [
            {"var": "d", "func": io_func, "dims": 2},
            {"var": "kd", "func": io_func, "dims": 2},
        ]
        parallel_heuristic = {
            "read": {
                "data": self.y_r,
                "choices": rw_parallel_choices,
            },
            "compute": {
                "data": self.y_c,
                "choices": [
                    {"var": "d", "func": comp_func, "dims": 4},
                    {"var": "kd", "func": comp_func, "dims": 4},
                ],
            },
            "write": {
                "data": self.y_w,
                "choices": rw_parallel_choices,
            },
        }
        not_parallel_heuristic = {
            "read": {
                "data": self.y_r,
                "choices": [
                    {"var": "k", "func": io_func, "dims": 2},
                    {
                        "var": "k_d",
                        "func": io_func_pr,
                        "dims": 3,
                        "restriction": self.has_parent,
                    },
                ],
            },
            "compute": {
                "data": self.y_c,
                "choices": [
                    {"var": "k", "func": comp_func, "dims": 4},
                ],
            },
            "write": {
                "data": self.y_w,
                "choices": [
                    {"var": "k", "func": io_func, "dims": 2},
                ],
            },
        }

        heuristic = (
            parallel_heuristic if self.allow_parallel else not_parallel_heuristic
        )

        def choice_is_allowed(var):
            return "restriction" not in var or var["restriction"]

        for phase, items in heuristic.items():
            err_dict = {}
            for choice in items["choices"]:
                choice["params_avg"], choice["cov_avg"] = curve_fit(
                    choice["func"],
                    self.get(choice["var"]),
                    items["data"],
                    choice["dims"],
                )
                y_ = choice["func"](self.get(choice["var"]), *choice["params_avg"])
                err = (y_ - items["data"]) / items["data"]
                s_err = np.mean(np.abs(err))
                if choice_is_allowed(choice):
                    err_dict[choice["var"]] = s_err
            best_choice = min(err_dict, key=err_dict.get)
            avg_data = next(x for x in items["choices"] if (x["var"] == best_choice))
            self.params_avg.set(phase, avg_data["params_avg"])
            self.cov_avg.set(phase, avg_data["cov_avg"])
            # Set special attributes
            self.can_intra_parallel.set(phase, best_choice == "kd")
            if phase == "read":
                self.parent_relevant = best_choice == "k_d"

    def _populate_data(self, stage_profile_data):
        for config_tuple, data in stage_profile_data.items():
            _, memory, num_func = config_tuple

            # Jolteon's conversion from memory to vCPU (lambda fix rate)
            # FIXME: self system that does not use this conversion
            num_vcpu = eq_vcpu_alloc(memory, num_func if self.allow_parallel else 1)

            # Only taken the first item in each round & discarding the first exec (erase cold start effects)
            # FIXME: check if more data improve results
            skip_first_pos = 0
            number_items = len(data["cold_start"]) - skip_first_pos
            self.y_r = np.concatenate(
                [self.y_r, [item[0] for item in data["read"][skip_first_pos:]]]
            )
            self.y_c = np.concatenate(
                [self.y_c, [item[0] for item in data["compute"][skip_first_pos:]]]
            )
            self.y_w = np.concatenate(
                [self.y_w, [item[0] for item in data["write"][skip_first_pos:]]]
            )
            self.d = np.concatenate([self.d, [num_func] * number_items])
            self.k = np.concatenate([self.k, [num_vcpu] * number_items])
            self.kd = np.concatenate(
                [self.kd, [eq_vcpu_alloc(memory, num_func)] * number_items]
            )
            self.k_d = np.concatenate([self.k_d, [[num_vcpu, num_func]] * number_items])
            self.y_s = np.concatenate(
                [self.y_s, [item[0] for item in data["cold_start"][skip_first_pos:]]]
            )
        self.params_avg.cold = self.y_s
        self.k_d = self.k_d.reshape(-1, 2).T

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        pass

    @overrides
    def load_model(self):
        pass

    @overrides
    def save_model(self):
        pass

    @overrides
    def parameters(self):
        self.coeffs.cold = np.percentile(self.params_avg.cold, 60)
        return np.array(self.coeffs)

    def sample_offline(self, num_samples=10000) -> np.ndarray:
        # seed_val = int(time.time())
        seed_val = 0
        rng = np.random.default_rng(seed=seed_val)

        cold_samples = rng.choice(self.params_avg.cold, num_samples)
        read_samples = rng.multivariate_normal(
            self.params_avg.read, self.cov_avg.read, num_samples
        )
        compute_samples = rng.multivariate_normal(
            self.params_avg.compute, self.cov_avg.compute, num_samples
        )
        write_samples = rng.multivariate_normal(
            self.params_avg.write, self.cov_avg.write, num_samples
        )

        params_list = [ModelParams() for _ in range(num_samples)]
        coeffs_list = [MixedModelCoefficients() for _ in range(num_samples)]

        for i in range(num_samples):
            params_list[i].read = read_samples[i]
            params_list[i].compute = compute_samples[i]
            params_list[i].write = write_samples[i]
            params_list[i].cold = cold_samples[i]

            self._set_coeff_by_params(coeffs_list[i], params_list[i], make_error=True)

        return np.array([np.array(coeffs_list[i]) for i in range(num_samples)])

    def generate_func_code(self, mode) -> str:
        config_list = "config_list"
        coeffs_list = "coeffs_list"

        assert mode in ["latency", "cost"]

        stage_idx = int(self._stage_idx)
        cold_param = f"{coeffs_list}[{stage_idx}][cold]"
        x_param = f"{coeffs_list}[{stage_idx}][x]"
        kd_d_param = f"{coeffs_list}[{stage_idx}][kd_d]"
        logx_param = f"{coeffs_list}[{stage_idx}][logx]"
        x2_param = f"{coeffs_list}[{stage_idx}][x2]"
        const_param = f"{coeffs_list}[{stage_idx}][const]"

        var_d = f"{config_list}[workers({stage_idx})]" if self.allow_parallel else "1"
        var_k = f"{config_list}[cpu({stage_idx})]"
        var_x = f"({var_k} * {var_d})" if self.can_intra_parallel.compute else f"({var_d})"

        if self.allow_parallel:
            code = (
                f"{x_param}/{var_d} + {kd_d_param}/({var_k}*{var_d}) + {logx_param}*np.log({var_x})/{var_x} + "
                f"{x2_param}/{var_x}**2 + {const_param}"
            )
        else:
            code = f"{x_param}/{var_k} + "
            if self.parent_relevant:
                code += f"{kd_d_param}*{config_list}[workers({self.parent_idx})] + "
            code += f"{logx_param}*np.log({var_k})/{var_k} + {x2_param}/{var_k}**2 + {const_param}"
        if mode == "latency":
            code = f"{cold_param} + {code}"
        else:
            # 1792 / 1024 * 0.0000000167 * 1000 = 0.000029225
            # 1000 is to convert from ms to s
            # We multiply 1e5 to the cost to make it more readable
            # s = cold_param + ' / 2 + ' + s
            code = f"({code}) * {var_k} * {var_d} * 2.9225 + 0.02 * {var_d}"
        return code
