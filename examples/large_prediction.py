from lithops.storage import Storage

from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.stage import (
    WorkflowStage,
)

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

# ws.run()

dataset_size = (
    int(
        Storage().head_object(bucket="test-bucket", key="tiny_shakespeare.txt")[
            "content-length"
        ]
    )
    / 1024**2
)

config_space = [
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
ws.profile(
    config_space=config_space,
    num_iter=3,
)

ws.train()

ws.plot_model_performance(config_space, path='images/generated.png')
print(f"Objective function {ws.objective_func}")

# scheduler = Scheduler(ws)
# scheduler.search_config()
