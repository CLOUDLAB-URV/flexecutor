from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt

from modelling.perfmodel import PerfModel
from utils.dataclass import StageConfig, FunctionTimes


def io_func(x, a, b):
    return a / x + b


def io_func2(x, a, b, c):  # io_func2 is for parent relavent read
    return a / x[0] + b * x[1] + c


def comp_func(x, a, b, c, d) -> float:
    return a / x + b * np.log(x) / x + c / x**2 + d


def curve_fit(func, x, y):
    return scipy_opt.curve_fit(func, x, y)[0:2]


class MixedPerfModel(PerfModel):
    def __init__(self, model_type, stage):
        super().__init__(model_type, stage)

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

        # Reduce the dimension of the parameters from 8 to 5, excluding cold start
        # By merging the parameters of read, compute, and write as follows:
        # allow_parallel: a/d + b/(kd) + c*log(x)/x + e/x**2 + f, x can be d or kd
        # not allow_parallel: a/k + b*d + c*log(k)/k + e/k**2 + f,
        self.x_coeff = 0  # the coefficient of 1/d or 1/k in the stage, x can be d or kd
        self.kd_d_coeff = 0  # the coefficient of 1/(kd) or d in the stage
        self.logx_coeff = (
            0  # the coefficient of log(x)/x in the stage, x can be d or kd
        )
        self.x2_coeff = 0  # the coefficient of 1/x**2 in the stage, x can be d or kd
        self.const_coeff = 0  # the constant coefficient in the stage

    def train(self, stage_profile_data: Dict) -> None:
        # STEP 1: Populate ndarray with data
        y_s = np.empty(0)  # cold start times
        y_r = np.empty(0)  # read times
        y_c = np.empty(0)  # compute times
        y_w = np.empty(0)  # write times
        d = np.empty(0)  # number of workers
        kd = np.empty(0)  # number of total cpu units
        k = np.empty(0)  # number of individual cpu units (per worker)
        k_d = np.empty(0)  # read time may be related to the parent number of functions
        for config_tuple, data in stage_profile_data.items():
            num_vcpu, memory, num_func = config_tuple
            number_items = len(
                [item for sublist in data["cold_start"] for item in sublist]
            )
            np.append(y_s, [item for sublist in data["cold_start"] for item in sublist])
            np.append(y_r, [item for sublist in data["read"] for item in sublist])
            np.append(y_c, [item for sublist in data["compute"] for item in sublist])
            np.append(y_w, [item for sublist in data["write"] for item in sublist])
            np.append(d, [num_func] * number_items)
            np.append(k, [num_vcpu] * number_items)
            np.append(kd, [num_vcpu * num_func] * number_items)
            np.append(k_d, [num_vcpu, num_func] * number_items)

        if self.allow_parallel:
            phases_params = {
                "read": {
                    "data": y_r,
                    "params": self.read_params_avg,
                    "covariance": self.read_cov_avg,
                    "func": io_func,
                },
                "compute": {
                    "data": y_c,
                    "params": self.compute_params_avg,
                    "covariance": self.compute_cov_avg,
                    "func": comp_func,
                },
                "write": {
                    "data": y_w,
                    "params": self.write_params_avg,
                    "covariance": self.write_cov_avg,
                    "func": io_func,
                },
            }

            for phase, items in phases_params.items():
                popt1, pcov1 = curve_fit(items["func"], d, items["data"])
                y_ = items["func"](d, *popt1)
                err1 = (y_ - items["data"]) / items["data"]
                popt2, pcov2 = curve_fit(items["func"], kd, items["data"])
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
            popt1, pcov1 = curve_fit(io_func, k, y_r)
            y_ = io_func(k, popt1[0], popt1[1])
            err1 = (y_ - y_r) / y_r
            popt2, pcov2 = curve_fit(io_func2, k_d.T, y_r)
            y_ = io_func2(k_d.T, popt2[0], popt2[1], popt2[2])
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
            self.compute_params_avg, self.compute_cov_avg = curve_fit(comp_func, k, y_c)

            # Write, directly use k to fit, comparison not needed
            self.write_params_avg, self.write_cov_avg = curve_fit(io_func, k, y_w)

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

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def parameters(self):
        pass

    def init_from_dag(self, dag):
        pass
