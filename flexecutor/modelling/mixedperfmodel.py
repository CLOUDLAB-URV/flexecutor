from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from modelling.perfmodel import PerfModel
from utils.dataclass import StageConfig, FunctionTimes


@dataclass
class MixedModelCoefficients:
    """
    The coefficients of the mixed performance model

    The dimension of the parameters is reduced from 8 to 5 (mixing common degree
    coeffs between phases), excluding cold start
    By merging the parameters of read, compute, and write as follows:
    - allow_parallel: a/d + b/(kd) + c*log(x)/x + e/x**2 + f, x can be d or kd
    - not allow_parallel: a/k + b*d + c*log(k)/k + e/k**2 + f
    """
    cold: float       # _ --> cold start time
    x: float          # a --> coefficient of 1/d or 1/k
    kd_d: float       # b --> coefficient of 1/(kd) or d
    logx: float       # c --> coefficient of log(x)/x, x can be d or kd
    x2: float         # e --> coefficient of 1/x**2, x can be d or kd
    const: float      # f --> constant coefficient


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


class MixedPerfModel(PerfModel):
    """
    Mixed performance model that combines the white-box and black-box modelling
    Here, definitions of the notations used in the model:
    - y_s: array with cold start times
    - y_r: array with read times
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

        self.can_intra_parallel = {
            "read": True,
            "compute": True,
            "write": True,
        }
        self.parent_relavent = (
            False  # only use for not allow parallel and related to parent stage
        )

        self.cold_params_avg = []  # random variable
        self.read_params_avg = []  # A/d + B, d is the equivalent vCPU allocation
        self.compute_params_avg = []  # A/d - B*log(d)/d + C/d**2 + D
        self.write_params_avg = []  # A/d + B
        self.read_cov_avg = []  # covariance matrix
        self.compute_cov_avg = []
        self.write_cov_avg = []

        self.coeffs = []


        self.x_coeff = 0  # the coefficient of 1/d or 1/k in the stage, x can be d or kd
        self.kd_d_coeff = 0  # the coefficient of 1/(kd) or d in the stage
        self.logx_coeff = (
            0  # the coefficient of log(x)/x in the stage, x can be d or kd
        )
        self.x2_coeff = 0  # the coefficient of 1/x**2 in the stage, x can be d or kd
        self.const_coeff = 0  # the constant coefficient in the stage

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        # STEP 1: Populate ndarray with data
        y_s = np.empty(0)  # cold start times
        y_r = np.empty(0)  # read times
        y_c = np.empty(0)  # compute times
        y_w = np.empty(0)  # write times
        d = np.empty(0)  # number of workers
        kd = []  # number of total cpu units
        k = np.empty(0)  # number of individual cpu units (per worker)
        k_d = []  # number of individual cpu units (per worker) and number of workers
        for config_tuple, data in stage_profile_data.items():
            _, memory, num_func = config_tuple

            # Jolteon's conversion from memory to vCPU (lambda fix rate)
            # FIXME: self system that does not use this conversion
            num_vcpu = eq_vcpu_alloc(memory, num_func if self.allow_parallel else 1)

            # Only taken the first item in each round & discarding the first exec (erase cold start effects)
            # FIXME: check if more data improve results
            number_items = len(data["cold_start"]) - 1
            y_r = np.append(y_r, [item[0] for item in data["read"][1:]])
            y_c = np.append(y_c, [item[0] for item in data["compute"][1:]])
            y_w = np.append(y_w, [item[0] for item in data["write"][1:]])
            d = np.append(d, [num_func] * number_items)
            k = np.append(k, [num_vcpu] * number_items)
            kd.append([[eq_vcpu_alloc(memory, num_func)] * number_items])
            k_d.append([[num_vcpu, num_func] * number_items])
            y_s = np.append(y_s, [item[0] for item in data["cold_start"][1:]])

        kd = np.array(kd).reshape(-1)
        k_d = np.array(k_d).reshape(-1, 2)
        self.cold_params_avg = y_s

        # STEP 2: Fit the data to the model & compute the coefficients
        if self.allow_parallel:
            phases_params = {
                "read": {
                    "data": y_r,
                    "params": self.read_params_avg,
                    "covariance": self.read_cov_avg,
                    "func": io_func,
                    "dims": 2,
                },
                "compute": {
                    "data": y_c,
                    "params": self.compute_params_avg,
                    "covariance": self.compute_cov_avg,
                    "func": comp_func,
                    "dims": 4,
                },
                "write": {
                    "data": y_w,
                    "params": self.write_params_avg,
                    "covariance": self.write_cov_avg,
                    "func": io_func,
                    "dims": 2,
                },
            }

            for phase, items in phases_params.items():
                popt1, pcov1 = curve_fit(items["func"], d, items["data"], items["dims"])
                y_ = items["func"](d, *popt1)
                err1 = (y_ - items["data"]) / items["data"]
                popt2, pcov2 = curve_fit(
                    items["func"], kd, items["data"], items["dims"]
                )
                y_ = items["func"](kd, *popt2)
                err2 = (y_ - items["data"]) / items["data"]
                s_err1 = np.mean(np.abs(err1))
                s_err2 = np.mean(np.abs(err2))
                m_err1 = np.mean(err1)
                m_err2 = np.mean(err2)
                if s_err1 < s_err2:
                    self.can_intra_parallel[phase] = False
                    items["params"] = popt1
                    items["covariance"] = pcov1
                else:
                    self.can_intra_parallel[phase] = True
                    items["params"] = popt2
                    items["covariance"] = pcov2
                if self.can_intra_parallel[phase]:
                    self.kd_d_coeff += items["params"][0]
                else:
                    self.x_coeff += items["params"][0]

            # FIXME: make this more elegant
            self.read_params_avg = phases_params["read"]["params"]
            self.compute_params_avg = phases_params["compute"]["params"]
            self.write_params_avg = phases_params["write"]["params"]
            self.read_cov_avg = phases_params["read"]["covariance"]
            self.compute_cov_avg = phases_params["compute"]["covariance"]
            self.write_cov_avg = phases_params["write"]["covariance"]

            self.logx_coeff += self.compute_params_avg[1]
            self.x2_coeff += self.compute_params_avg[2]
            self.const_coeff += (
                self.read_params_avg[1]
                + self.compute_params_avg[3]
                + self.write_params_avg[1]
            )

            y_actual = y_r + y_c + y_w + y_s
            y_pred = (
                self.x_coeff / d
                + self.kd_d_coeff / kd
                + self.const_coeff
                + np.mean(y_s)
            )
            if self.can_intra_parallel["compute"]:
                y_pred += self.logx_coeff * np.log(kd) / kd + self.x2_coeff / kd**2
            else:
                y_pred += self.logx_coeff * np.log(d) / d + self.x2_coeff / d**2
        else:
            popt1, pcov1 = curve_fit(io_func, k, y_r, 2)
            y_ = io_func(k, popt1[0], popt1[1])
            err1 = (y_ - y_r) / y_r
            popt2, pcov2 = curve_fit(io_func_pr, k_d.T, y_r, 3)
            y_ = io_func_pr(k_d.T, popt2[0], popt2[1], popt2[2])
            err2 = (y_ - y_r) / y_r
            s_err1 = np.mean(np.abs(err1))
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2 or self.has_parent is False:
                self.parent_relavent = False
                self.read_params_avg = popt1
                self.read_cov_avg = pcov1
            else:
                self.parent_relavent = True
                self.read_params_avg = popt2
                self.read_cov_avg = pcov2

            # Compute, directly use k to fit, comparison not needed
            self.compute_params_avg, self.compute_cov_avg = curve_fit(
                comp_func, k, y_c, 4
            )

            # Write, directly use k to fit, comparison not needed
            self.write_params_avg, self.write_cov_avg = curve_fit(io_func, k, y_w, 2)

            # Compute the coefficients
            self.x_coeff += (
                self.read_params_avg[0]
                + self.compute_params_avg[0]
                + self.write_params_avg[0]
            )
            if self.parent_relavent:
                self.kd_d_coeff += self.read_params_avg[1]
                self.const_coeff += self.read_params_avg[2]
            else:
                self.const_coeff += self.read_params_avg[1]
            self.logx_coeff += self.compute_params_avg[1]
            self.x2_coeff += self.compute_params_avg[2]
            self.const_coeff += self.compute_params_avg[3] + self.write_params_avg[1]

            # Compute the error for the stage
            y_actual = y_r + y_c + y_w + y_s
            y_pred = (
                self.x_coeff / k
                + self.kd_d_coeff * k_d.T[1]
                + self.const_coeff
                + np.mean(y_s)
                + self.logx_coeff * np.log(k) / k
                + self.x2_coeff / k**2
            )

        # STEP 3: Verify the accuracy of the model
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

    # @overrides
    # def predict_time(self, config: StageConfig) -> FunctionTimes:
    #     mode = "latency"
    #
    #     # FIXME: check parent_d and cold_percent meaning
    #     parent_d = 0
    #     cold_percent = 60
    #
    #     # FIXME: check conversion cpu-memory in this lines
    #     k = config.cpu
    #     kd = config.cpu * config.workers
    #     d = config.workers
    #     x = [1.0 / d, 1.0 / kd, np.log(d) / d, 1.0 / d**2, 1.0]
    #     if self.allow_parallel:
    #         if self.can_intra_parallel[1]:
    #             x[2] = np.log(kd) / kd
    #             x[3] = 1.0 / kd**2
    #     else:
    #         x = [1.0 / k, parent_d, np.log(k) / k, 1.0 / k**2, 1.0]
    #         if not self.parent_relavent:
    #             x[1] = 0
    #
    #     params = self.parameters
    #     pred = np.dot(params[1:], x)
    #     # if input_size != 1024:
    #     #     pred *= input_size / self.default_input_size
    #     if mode == "latency":
    #         pred += np.percentile(self.cold_params_avg, cold_percent)
    #         return pred
    #     else:
    #         # TODO: retrieve the meaning of this weird formula
    #         # 1792 / 1024 * 0.0000000167 * 1000
    #         return (
    #             pred * config.workers * config.cpu * 2.9225 + 0.02 * config.workers
    #         ) / 100000
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
        cold_percent = 60
        cold_coeff = np.percentile(self.cold_params_avg, cold_percent)

        return np.array(
            [
                cold_coeff,
                self.x_coeff,
                self.kd_d_coeff,
                self.logx_coeff,
                self.x2_coeff,
                self.const_coeff,
            ]
        )

    def sample_offline(self, num_samples=10000):
        # seed_val = int(time.time())
        seed_val = 0

        rng = np.random.default_rng(seed=seed_val)
        cold_samples = rng.choice(self.cold_params_avg, num_samples)
        read_samples = rng.multivariate_normal(
            self.read_params_avg, self.read_cov_avg, num_samples
        )
        compute_samples = rng.multivariate_normal(
            self.compute_params_avg, self.compute_cov_avg, num_samples
        )
        write_samples = rng.multivariate_normal(
            self.write_params_avg, self.write_cov_avg, num_samples
        )

        # Organize into coefficient form
        coeffs = np.zeros((num_samples, 6))
        coeffs[:, 0] = cold_samples
        if self.allow_parallel:
            if self.can_intra_parallel["read"]:
                coeffs[:, 2] += read_samples.T[0]  # 1/(kd)
            else:
                coeffs[:, 1] += read_samples.T[0]  # 1/d
            if self.can_intra_parallel["compute"]:
                coeffs[:, 2] += compute_samples.T[0]
            else:
                coeffs[:, 1] += compute_samples.T[0]
            if self.can_intra_parallel["write"]:
                coeffs[:, 2] += write_samples.T[0]
            else:
                coeffs[:, 1] += write_samples.T[0]
            coeffs[:, 3] += compute_samples.T[1]  # log(x)/x
            coeffs[:, 4] += compute_samples.T[2]  # 1/x**2
            coeffs[:, 5] += (
                read_samples.T[1] + compute_samples.T[3] + write_samples.T[1]
            )
        else:
            coeffs[:, 1] += (
                read_samples.T[0] + compute_samples.T[0] + write_samples.T[0]
            )
            if self.parent_relavent:
                coeffs[:, 2] += read_samples.T[1]
                coeffs[:, 5] += read_samples.T[2]
            else:
                coeffs[:, 5] += read_samples.T[1]
            coeffs[:, 3] += compute_samples.T[1]
            coeffs[:, 4] += compute_samples.T[2]

        return coeffs

    def generate_func_code(
        self, mode, var, param, parent_id=-1, solver_type="scipy"
    ) -> str:
        # assert isinstance(parent_id, int)
        parent_id = int(parent_id)
        assert mode in ["latency", "cost"]
        assert isinstance(var, str) and isinstance(param, str)
        assert solver_type == "scipy"

        # 6 param indices and 2 var indices for each stage
        # 0: cold, 1: x, 2: kd/d, 3: log(x)/x, 4: 1/x**2, 5: const
        # 0: var d, 1: var k

        s = ""
        offset = 0 if solver_type == "scipy" else 1
        stage_id = int(self._stage_id)
        cold_param = param + "[%d]" % (stage_id * 6 + offset)
        x_param = param + "[%d]" % (stage_id * 6 + 1 + offset)
        kd_d_param = param + "[%d]" % (stage_id * 6 + 2 + offset)
        logx_param = param + "[%d]" % (stage_id * 6 + 3 + offset)
        x2_param = param + "[%d]" % (stage_id * 6 + 4 + offset)
        const_param = param + "[%d]" % (stage_id * 6 + 5 + offset)

        var_d = var + "[%d]" % (stage_id * 2 + offset)
        if not self.allow_parallel:
            var_d = "1"
        var_k = var + "[%d]" % (stage_id * 2 + 1 + offset)
        var_x = ""
        if self.can_intra_parallel["compute"]:
            var_x = var_k + "*" + var_d
        else:
            var_x = var_d
        var_x = "(" + var_x + ")"

        log_method = "np.log"

        if self.allow_parallel:
            s += x_param + "/" + var_d + " + "
            s += kd_d_param + "/" + "(" + var_k + "*" + var_d + ")" + " + "
            s += logx_param + "*" + log_method + var_x + "/" + var_x + " + "
            s += x2_param + "/" + var_x + "**2" + " + "
            s += const_param
        else:
            s += x_param + "/" + var_k + " + "
            if self.parent_relavent and parent_id >= 0:
                var_pd = var + "[%d]" % (parent_id * 2)  # parent d
                s += kd_d_param + "*" + var_pd + " + "
            s += logx_param + "*" + log_method + "(" + var_k + ")" + "/" + var_k + " + "
            s += x2_param + "/" + var_k + "**2" + " + "
            s += const_param
        if mode == "latency":
            s = cold_param + " + " + s
        else:
            # 1792 / 1024 * 0.0000000167 * 1000 = 0.000029225
            # 1000 is to convert from ms to s
            # We multiply 1e5 to the cost to make it more readable
            # s = cold_param + ' / 2 + ' + s
            s = "(" + s + ") * " + var_k + " * " + var_d + " * 2.9225 + 0.02 * " + var_d
        return s
