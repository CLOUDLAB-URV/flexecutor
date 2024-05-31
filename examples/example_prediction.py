import numpy as np
from example_training import ws
from flexexecutor.core.scheduling import Scheduler

# Finally, after all the training, we can predict optimal configurations that minimize latency

objective_function = ws.get_objective_function()
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


def calculate_actual_latency(act_latency):
    read_latencies = [item["read"] for item in act_latency]
    compute_latencies = [item["compute"] for item in act_latency]
    write_latencies = [item["write"] for item in act_latency]
    cold_start_times = [item["cold_start_time"] for item in act_latency]
    average_read = np.mean(read_latencies)
    average_compute = np.mean(compute_latencies)
    average_write = np.mean(write_latencies)

    median_cold_start = np.median(cold_start_times)

    total_latency = average_read + average_compute + average_write + median_cold_start

    return total_latency


act_latency = ws.run()

act_latency = calculate_actual_latency(act_latency)
print(
    "Predicted latency with optimal configuration",
    pred_latency,
)
print("Actual latency:", act_latency)

print(f"Accuracy {100 - (act_latency - pred_latency) / pred_latency * 100} %")
