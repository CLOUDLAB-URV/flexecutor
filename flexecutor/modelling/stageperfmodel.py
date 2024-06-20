import numpy as np
from overrides import overrides
import scipy.optimize as scipy_opt

from flexecutor.modelling.perfmodel import PerfModel


def eq_vcpu_alloc(mem, num_func):
    num_vcpu = mem / 1792
    return round(num_vcpu * num_func, 1)


def io_func(x, a, b):
    return a / x + b


# io_func2 is for parent relavent read
def io_func2(x, a, b, c):  
    return a / x[0] + b * x[1] + c


def comp_func(x, a, b, c, d):
    return a / x + b * np.log(x) / x + c / x**2 + d


class StagePerfModel(PerfModel):
    """
    StagePerfModel records the parameter distributions of a stage's performance model
    Adapted from https://github.com/pkusys/Jolteon/blob/main/workflow/perf_model.py

    """

    def __init__(
        self,
        model_name=None,
        model_dst=None,
        stage_id=None,
        stage_name=None,
        default_input_size=1024,
    ) -> None:
        assert isinstance(stage_name, str)
        assert isinstance(stage_id, int) and stage_id >= 0
        super().__init__("stage", model_name, model_dst)

        self._stage_name = stage_name
        self._stage_id = stage_id

        self._profiling_results = None

        self._allow_parallel = True
        self._has_parent = False

        assert isinstance(default_input_size, int) and default_input_size > 0
        self.default_input_size = default_input_size  # MB

        self._cold_params_avg = []  # random variable
        self._read_params_avg = []  # A/d + B, d is the equivalent vCPU allocation
        self._compute_params_avg = []  # A/d - B*log(d)/d + C/d**2 + D
        self._write_params_avg = []  # A/d + B
        self._read_cov_avg = []  # covariance matrix
        self._compute_cov_avg = []
        self._write_cov_avg = []

        self._can_intra_parallel = [True, True, True]  # stands for read, compute, write
        self._parent_relavent = False

        # Reduce the dimension of the parameters from 8 to 5, excluding cold start
        # By merging the parameters of read, compute, and write as follows:
        # allow_parallel: a/d + b/(kd) + c*log(x)/x + e/x**2 + f, x can be d or kd
        # not allow_parallel: a/k + b*d + c*log(k)/k + e/k**2 + f,
        self.x_coeff = 0  # the coefficient of 1/d or 1/k in the stage, x can be d or kd
        self.kd_d_coeff = 0  # the coefficient of 1/(kd) or d in the stage
        self.logx_coeff = 0  # the coefficient of log(x)/x in the stage, x can be d or kd
        self.x2_coeff = 0  # the coefficient of 1/x**2 in the stage, x can be d or kd
        self.const_coeff = 0  # the constant coefficient in the stage

    # TODO: review that and rethink
    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self._allow_parallel = allow_parallel

    def update_has_parent(self, has_parent) -> None:
        assert isinstance(has_parent, bool)
        self._has_parent = has_parent

    @overrides
    def save_model(self):
        pass

    @overrides
    def load_model(self):
        pass

    @overrides
    def parameters(self, cold_percent=60):
        cold_coeff = np.percentile(self._cold_params_avg, cold_percent)
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

    def compute_errors(self, func, x1, x2, y, params_avg, cov_avg, index):
        popt1, pcov1 = scipy_opt.curve_fit(func, x1, y)
        y_ = func(x1, *popt1)
        err1 = (y_ - y) / y

        popt2, pcov2 = scipy_opt.curve_fit(func, x2, y)
        y_ = func(x2, *popt2)
        err2 = (y_ - y) / y

        s_err1 = np.mean(np.abs(err1))
        s_err2 = np.mean(np.abs(err2))

        if s_err1 < s_err2:
            self._can_intra_parallel[index] = False
            params_avg[index] = popt1
            cov_avg[index] = pcov1
        else:
            self._can_intra_parallel[index] = True
            params_avg[index] = popt2
            cov_avg[index] = pcov2

        if self._can_intra_parallel[index]:
            self.kd_d_coeff += params_avg[index][0]
        else:
            self.x_coeff += params_avg[index][0]

    @overrides
    def train(self, stage_profile_data: dict) -> None:
        self._profiling_results = stage_profile_data
        assert isinstance(stage_profile_data, dict)
        
        # Validate profile_data
        keys = {"read", "compute", "write"}
        for profile_data in stage_profile_data.values():
                assert isinstance(profile_data, dict) and keys <= profile_data.keys()
        
        print(f"Training Stage performance model for {self._stage_name}")

        def get_metric(metric):
            return np.array([data[metric] for data in stage_profile_data.values()])[:, 1:, 0].reshape(-1)

        y_s = np.array(
            [data["cold_start"] for _, data in stage_profile_data.items()]
        )

        # Remove cold start
        num_epochs = y_s.shape[0]
        assert num_epochs >= 2
        num_epochs -= 1
        y_s = y_s[:, 1:, 0].reshape(-1)     
        self._cold_params_avg = y_s

        y_r, y_c, y_w = map(get_metric, ["read", "compute", "write"])
        print(f"y_s: {y_s}\ny_r: {y_r}\ny_c: {y_c}\ny_w: {y_w}\nnum_epochs: {num_epochs}")

        if self._allow_parallel:
            def get_vcpu_memory_func():
                return [(num_vcpu, memory, num_func) for (num_vcpu, memory, num_func), _ in stage_profile_data.items()] * num_epochs
            
            d = np.array([num_func for (_, _, num_func) in get_vcpu_memory_func()])
            kd = np.array([eq_vcpu_alloc(memory, num_func) for (_, memory, num_func) in get_vcpu_memory_func()])

            print(f"d: {d}")

            tasks = [
                {"func": io_func, "y": y_r},
                {"func": comp_func, "y": y_c},
                {"func": io_func, "y": y_w},
            ]
            params_avg = [self._read_params_avg, self._compute_params_avg, self._write_params_avg]
            cov_avg = [self._read_cov_avg, self._compute_cov_avg, self._write_cov_avg]

            for i, task in enumerate(tasks):
                self.compute_errors(task["func"], d, kd, task["y"], params_avg, cov_avg, i)

            self._read_params_avg, self._compute_params_avg, self._write_params_avg = params_avg
            self._read_cov_avg, self._compute_cov_avg, self._write_cov_avg = cov_avg

            self.logx_coeff += self._compute_params_avg[1]
            self.x2_coeff += self._compute_params_avg[2]
            self.const_coeff += self._read_params_avg[1] + self._compute_params_avg[3] + self._write_params_avg[1]

            y_actual = y_r + y_c + y_w + y_s
            y_pred = (self.x_coeff / d + self.kd_d_coeff / kd + self.const_coeff + np.mean(y_s) +
                    (self.logx_coeff * np.log(kd) / kd if self._can_intra_parallel[1] else self.logx_coeff * np.log(d) / d) +
                    self.x2_coeff / (kd if self._can_intra_parallel[1] else d)**2)
        else:
            k = np.array([eq_vcpu_alloc(memory, 1) for (_, memory, _) in get_vcpu_memory_func()])
            k_d = np.array([[eq_vcpu_alloc(memory, 1), num_func] for (_, memory, num_func) in get_vcpu_memory_func()])

            def fit_and_calc_error(func, x, y):
                popt, pcov = scipy_opt.curve_fit(func, x, y)
                y_ = func(x, *popt)
                return popt, pcov, (y_ - y) / y

            self._parent_relavent = False
            self._read_params_avg, self._read_cov_avg, err1 = fit_and_calc_error(io_func, k, y_r)
            _, _, err2 = fit_and_calc_error(io_func2, k_d.T, y_r)

            if np.mean(np.abs(err1)) >= np.mean(np.abs(err2)) and self.has_parent:
                self._parent_relavent = True
                self._read_params_avg, self._read_cov_avg, _ = fit_and_calc_error(io_func2, k_d.T, y_r)

            self._compute_params_avg, self._compute_cov_avg, _ = fit_and_calc_error(comp_func, k, y_c)
            self._write_params_avg, self._write_cov_avg, _ = fit_and_calc_error(io_func, k, y_w)

            self.x_coeff += sum(params[0] for params in [self._read_params_avg, self._compute_params_avg, self._write_params_avg])
            self.kd_d_coeff += self._read_params_avg[1] if self._parent_relavent else 0
            self.const_coeff += self._read_params_avg[2] if self._parent_relavent else self._read_params_avg[1]
            self.logx_coeff += self._compute_params_avg[1]
            self.x2_coeff += self._compute_params_avg[2]
            self.const_coeff += self._compute_params_avg[3] + self._write_params_avg[1]

            y_actual = y_r + y_c + y_w + y_s
            y_pred = (self.x_coeff / k + self.kd_d_coeff * k_d.T[1] + self.const_coeff + np.mean(y_s) +
                    self.logx_coeff * np.log(k) / k + self.x2_coeff / k**2)

        err = (y_pred - y_actual) / y_actual
        s_err = np.mean(np.abs(err))
        m_err = np.mean(err)
        
        print(f"Stage mean abs error: {s_err * 100:.2f} %")
        print(f"Stage mean error: {m_err * 100:.2f} %")


    @overrides
    def predict(
        self,
        num_cpu,
        runtime_memory,
        num_workers,
        chunk_size=1024,
        mode="latency",
        parent_d=0,
        cold_percent=60,
    ) -> float:
        assert num_workers > 0
        assert mode in ["latency", "cost"]

        k = eq_vcpu_alloc(num_cpu * runtime_memory, 1)
        kd = eq_vcpu_alloc(num_cpu * runtime_memory, num_workers)
        d = num_workers
        x = [1.0 / d, 1.0 / kd, np.log(d) / d, 1.0 / d**2, 1.0]
        if self._allow_parallel:
            if self._can_intra_parallel[1]:
                x[2] = np.log(kd) / kd
                x[3] = 1.0 / kd**2
        else:
            x = [1.0 / k, parent_d, np.log(k) / k, 1.0 / k**2, 1.0]
            if not self._parent_relavent:
                x[1] = 0

        params = self.parameters()
        pred = np.dot(params[1:], x)
        if chunk_size != 1024:
            pred *= chunk_size / self.default_input_size
        if mode == "latency":
            pred += np.percentile(self._cold_params_avg, cold_percent)
            return pred
        else:
            mem_price = runtime_memory / 1024 * 0.0000000167 * 1000 * 100000
            return (
                pred * num_workers * num_cpu * mem_price + 0.02 * num_workers
            ) / 100000
