import numpy as np

from functions.word_occurrence import word_occurrence_count
from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.scheduling import Scheduler
from flexecutor.stage import WorkflowStage

config = {"log_level": "INFO"}

data_location = {
    "obj": "test-bucket/corpus.txt",
}

ws = WorkflowStage(
    name="word_count",
    model=PerfModel.instance("genetic"),
    function=word_occurrence_count,
    input_data=data_location,
    output_data="test-bucket/combined_file.txt",
    config=config,
)

config_space = [
    (3, 1024, 2),  # 1 vCPU, 512 MB per worker, 10 workers
    (1, 200, 10),  # 1 vCPU, 200 MB per worker, 10 workers
    (2, 2048, 7),  # 2 vCPUs, 2048 MB per worker, 7 workers
    (3, 3072, 5),  # 3 vCPUs, 3072 MB per worker, 5 workers
    (1, 512, 15),  # 1 vCPU, 512 MB per worker, 15 workers
    (1, 1024, 10),  # 1 vCPU, 1024 MB per worker, 10 workers
    (2, 2048, 5),  # 2 vCPUs, 2048 MB per worker, 5 workers
    (2, 1707, 6),  # 2 vCPUs, 1707 MB per worker, 6 workers
    (3, 3413, 3),  # 3 vCPUs, 3413 MB per worker, 3 workers
    (4, 4096, 4),  # 4 vCPUs, 4096 MB per worker, 4 workers
    (4, 5120, 2),  # 4 vCPUs, 5120 MB per worker, 2 workers
    (6, 10240, 1),  # 6 vCPUs, 10240 MB per worker, 1 worker
    (1, 2048, 10),  # 1 vCPU, 2048 MB per worker, 10 workers
    (2, 4096, 8),  # 2 vCPUs, 4096 MB per worker, 8 workers
    (3, 6144, 6),  # 3 vCPUs, 6144 MB per worker, 6 workers
    (4, 8192, 4),  # 4 vCPUs, 8192 MB per worker, 4 workers
    (5, 10240, 2),  # 5 vCPUs, 10240 MB per worker, 2 workers
    (6, 12288, 1),  # 6 vCPUs, 12288 MB per worker, 1 worker
]
ws.profile(config_space, num_iter=2)

# Once profiling is done, we can train the model we passed to the workflow step, it will save the model into a file
ws.train()

# After profiling, we can print the objective function from the performance model
objective_function = ws.objective_func
print(f"Objective function {objective_function}")

x_bound = [
    (1, 6),
    (512, 4096),
    (1, 3),
]

scheduler = Scheduler(ws)
result = scheduler.search_config(x_bound)
optimal_configuration = np.round(result.x).astype(int)
print("Integer Optimal Configuration:", optimal_configuration)
pred_latency = objective_function(optimal_configuration)


ws.update_config(
    cpu=optimal_configuration[0],
    memory=optimal_configuration[1],
    workers=optimal_configuration[2],
)


def calculate_actual_latency(timings):
    average_read = np.mean(timings["read"])
    average_compute = np.mean(timings["compute"])
    average_write = np.mean(timings["write"])
    cold_start_times = timings["cold_start_time"]
    median_cold_start = np.median(cold_start_times)
    total_latency = average_read + average_compute + average_write + median_cold_start
    return total_latency


exec_timings = ws.run()

exec_timings = calculate_actual_latency(exec_timings)
print(
    "Predicted latency with optimal configuration",
    pred_latency,
)
print("Actual latency:", exec_timings)

print(f"Accuracy {100 - (exec_timings - pred_latency) / pred_latency * 100} %")
