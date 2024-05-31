from flexexecutor.core.stage import (
    WorkflowStage,
    initialize_timings,
    operation,
)
from flexexecutor.core.modelling import AnaPerfModel, GAPerfModel
from lithops.storage import Storage
from flexexecutor.core.scheduling import Scheduler
import collections
import numpy as np

config = {"log_level": "INFO"}


def word_occurrence_count(obj):
    timings = initialize_timings()
    storage = Storage()

    with operation("read", timings):
        data = obj.data_stream.read().decode("utf-8")

    with operation("compute", timings):
        words = data.split()
        word_count = collections.Counter(words)

    with operation("write", timings):
        result_key = f"results_{obj.data_byte_range[0]}-{obj.data_byte_range[1]}.txt"
        result_data = (
            f"Word Count: {len(word_count)}\nWord Frequencies: {dict(word_count)}\n"
        )

        storage.put_object(obj.bucket, result_key, result_data.encode("utf-8"))

    return timings


data_location = {
    "obj": "test-bucket/corpus.txt",
}

ws = WorkflowStage(
    name="word_count",
    model=GAPerfModel(),
    function=word_occurrence_count,
    input_data=data_location,
    output_data="test-bucket/combined_file.txt",
    config=config,
)

dataset_size = (
    int(
        Storage().head_object(bucket="test-bucket", key="tiny_shakespeare.txt")[
            "content-length"
        ]
    )
    / 1024**2
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
ws.profile(config_space)

ws.train()
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
print(
    "Predicted latency with optimal configuration",
    pred_latency,
)

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
print("Actual latency:", act_latency)

print(f"Accuracy {100 - (act_latency - pred_latency) / pred_latency * 100} %")
