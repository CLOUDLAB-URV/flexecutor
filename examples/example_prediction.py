from flexexecutor.core.stage import (
    WorkflowStage,
    initialize_timings,
    operation,
)
from flexexecutor.core.modelling import AnaPerfModel

from lithops.storage import Storage

from flexexecutor.core.optimization import (
    OptimizationProblemSolver,
)
from flexexecutor.core.scheduling import Scheduler
import collections

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
    "obj": "test-bucket/tiny_shakespeare.txt",
}


ws = WorkflowStage(
    name="word_count",
    model=AnaPerfModel(1, "word_count"),
    function=word_occurrence_count,
    input_data=data_location,
    output_data="test-bucket/tiny_shakespeare.txt",
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

# Uncomment this section if profiling is needed
ws.profile(
    config_space=[(2, 400, 5)],  # Example configuration
    num_iter=2,
)

ws.train()
# ws.generate_objective_function()

# scheduler = Scheduler(ws)
# scheduler.search_config()
