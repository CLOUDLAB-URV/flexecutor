from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum
from flexecutor.utils.dataclass import ResourceConfig
from flexecutor.utils.utils import load_profiling_results

if __name__ == "__main__":
    profile_data = load_profiling_results("profiling/mocks/test1.json")

    perfmodel = PerfModel.instance(PerfModelEnum.ANALYTIC)
    perfmodel.update_allow_parallel(True)
    perfmodel.train(profile_data)

    print(perfmodel.parameters)
    print(perfmodel.predict(ResourceConfig(2, 400, 5)))
    # perfmodel.visualize(step="compute", degree=2)
    # perfmodel.visualize(step="read", degree=2)
    # perfmodel.visualize(step="write", degree=2)

    # Generate function code for latency
    # print(perfmodel.generate_func_code())
