from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum
from flexecutor.utils.dataclass import StageConfig
from flexecutor.utils.utils import flexorchestrator, load_profiling_results

if __name__ == "__main__":

    @flexorchestrator()
    def main():
        profile_data = load_profiling_results("profiling/mocks/test1.json")

        model = PerfModel.instance(PerfModelEnum.GENETIC)
        model.train(profile_data)
        print("Objective Function:", model.objective_func)
        prediction = model.predict(StageConfig(cpu=2, memory=400, workers=5))
        print(
            "Predicted Latency for (2 CPUs, 400 Memory, 5 Workers):", prediction.total
        )

    main()
