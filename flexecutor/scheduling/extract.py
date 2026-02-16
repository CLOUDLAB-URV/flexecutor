from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.scheduling.scheduler import Scheduler
from flexecutor.utils.dataclass import StageConfig
from flexecutor.utils.utils import load_profiling_results, get_my_exec_path
from flexecutor.workflow.executor import AssetType, get_asset_path


class OptimizationStrategy:
    def optimize(self, profiling_data, target) -> StageConfig:
        pass


def mean_of_phase(samples):
    all_values = [v for sublist in samples for v in sublist]
    return sum(all_values) / len(all_values)


def op_of_phase(samples, fn):
    all_values = [v for sublist in samples for v in sublist]
    return fn(all_values)


class TimeOptimizationStrategy(OptimizationStrategy):
    def optimize(self, profiling_data, target) -> StageConfig:
        best_config = None
        min_total_time = float("inf")

        for config, metrics in profiling_data.items():
            read = op_of_phase(metrics["read"], max)
            compute = op_of_phase(metrics["compute"], max)
            write = op_of_phase(metrics["write"], max)
            cold_start = op_of_phase(metrics["cold_start"], max)

            total_time = read + compute + write + cold_start

            if total_time < min_total_time:
                min_total_time = total_time
                best_config = config

        return StageConfig(
            cpu=best_config[0], memory=best_config[1], workers=best_config[2]
        )


class UsageOptimizationStrategy(OptimizationStrategy):
    def optimize(self, profiling_data, target) -> StageConfig:
        best_config = None
        min_usage_score = float("inf")

        for config, metrics in profiling_data.items():
            read = mean_of_phase(metrics["read"])
            compute = mean_of_phase(metrics["compute"])
            write = mean_of_phase(metrics["write"])
            cold_start = mean_of_phase(metrics["cold_start"])

            total_time = read + compute + write + cold_start
            usage_score = total_time * (config[0] * config[1] * config[2])

            if usage_score < min_usage_score:
                min_usage_score = usage_score
                best_config = config

        return StageConfig(
            cpu=best_config[0], memory=best_config[1], workers=best_config[2]
        )


class PerformanceOptimizationStrategy(OptimizationStrategy):
    #  (1/time) * (1/usage)
    def optimize(self, profiling_data, target) -> StageConfig:
        best_config = None
        best_score = float("-inf")

        for config, metrics in profiling_data.items():
            read_avg = mean_of_phase(metrics["read"])
            compute_avg = mean_of_phase(metrics["compute"])
            write_avg = mean_of_phase(metrics["write"])
            cold_start_avg = mean_of_phase(metrics["cold_start"])

            total_time = read_avg + compute_avg + write_avg + cold_start_avg
            usage_score = total_time * (config[0] * config[1] * config[2])

            score = (1 / total_time) * (1 / usage_score)

            if score > best_score:
                best_score = score
                best_config = config

        return StageConfig(
            cpu=best_config[0], memory=best_config[1], workers=best_config[2]
        )


class Extract(Scheduler):
    def __init__(self, dag, target):
        if target not in ["time", "usage", "performance"]:
            raise ValueError(
                "Invalid target for Extract scheduler. Must be 'time', 'usage', or 'performance'."
            )

        super().__init__(dag, PerfModelEnum.NONE)

        self.strategies = {
            "time": TimeOptimizationStrategy(),
            "usage": UsageOptimizationStrategy(),
            "performance": PerformanceOptimizationStrategy(),
        }
        self.strategy = self.strategies.get(target)

    def schedule(self) -> list[StageConfig]:
        stages_list = self._dag.stages
        stage_configs = []
        for stage in stages_list:
            profile_data = load_profiling_results(
                get_asset_path(get_my_exec_path(), self._dag, stage, AssetType.PROFILE)
            )
            stage_configs.append(self.strategy.optimize(profile_data, stage))
        return stage_configs
