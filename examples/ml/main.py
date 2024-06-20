from typing import Any
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.stage import Stage


def dummy_func(obj: Any):
    pass


if __name__ == "__main__":

    @flexorchestrator
    def main():
        dag = DAG("machine-learning")

        stage0 = Stage(stage_id="stage0", func=dummy_func)
        stage1 = Stage(stage_id="stage1", func=dummy_func)
        stage2 = Stage(stage_id="stage2", func=dummy_func)
        stage3 = Stage(stage_id="stage3", func=dummy_func)

        stage0 >> [stage1, stage2, stage3]
        stage1 >> stage2
        stage2 >> stage3

        dag.add_stages([stage0, stage1, stage2, stage3])
        dag.draw()

    main()
