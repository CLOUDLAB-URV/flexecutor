from examples.functions.word_occurrence import word_occurrence_count
from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.stage import WorkflowStage
from flexecutor.utils import setup_logging

if __name__ == "__main__":
    config = {"log_level": "INFO"}
    logger = setup_logging(config["log_level"])

    ws = WorkflowStage(
        name="word_count",
        model=PerfModel.instance("analytic"),
        function=word_occurrence_count,
        input_data="test-bucket/tiny_shakespeare.txt",
        output_data="test-bucket/tiny_shakespeare.txt",
        config=config,
    )

    ws.profile(
        config_space=[(2, 400, 5)],
        num_iter=2,
        # data_location="test-bucket/tiny_shakespeare.txt",
    )
