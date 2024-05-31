from flexecutor.modelling.perfmodel import PerfModel

if __name__ == "__main__":
    profiling_results = {
        (1, 1024, 4, 64): {
            "read": [[0.45, 0.55]],
            "compute": [[1.0]],
            "write": [[0.3]],
            "cold_start_time": [[0.2]],
        },
        (2, 2048, 8, 128): {
            "read": [[0.4]],
            "compute": [[0.9]],
            "write": [[0.25]],
            "cold_start_time": [[0.15]],
        },
        (3, 3072, 16, 256): {
            "read": [[0.3]],
            "compute": [[0.8]],
            "write": [[0.2]],
            "cold_start_time": [[0.1]],
        }
    }

    perfmodel = PerfModel.instance("analytic")
    perfmodel.update_allow_parallel(True)
    perfmodel.train(profiling_results)

    print(perfmodel.parameters)

    # perfmodel.visualize(step="compute", degree=2)
    # perfmodel.visualize(step="read", degree=2)
    # perfmodel.visualize(step="write", degree=2)

    # Generate function code for latency
    # print(perfmodel.generate_func_code())