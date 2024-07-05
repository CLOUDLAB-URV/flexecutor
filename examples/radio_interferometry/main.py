from lithops import FunctionExecutor

from examples.radio_interferometry.functions import (
    rebinning,
    imaging,
    calibration,
)
from flexecutor.storage.storage import FlexInput, FlexOutput, StrategyEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("radio-interferometry")

        rebinning_stage = Stage(
            stage_id="rebinning",
            func=rebinning,
            inputs=[
                FlexInput(prefix="partitions"),
                FlexInput(
                    prefix="parameters/rebinning",
                    custom_input_id="lua",
                    strategy=StrategyEnum.BROADCAST,
                ),
            ],
            outputs=[
                FlexOutput(
                    prefix="rebinning_out/ms",
                    suffix=".ms.zip",
                ),
                FlexOutput(
                    prefix="rebinning_out/logs",
                    suffix=".log",
                ),
            ],
        )

        full_calibration_stage = Stage(
            stage_id="full_calibration",
            func=calibration,
            inputs=[
                FlexInput(
                    prefix="rebinning_out/ms",
                ),
                FlexInput(
                    prefix="parameters/calibration/step2a",
                    custom_input_id="step2a",
                    strategy=StrategyEnum.BROADCAST,
                ),
            ],
            outputs=[
                FlexOutput(
                    prefix="applycal_out/cal/h5",
                    custom_output_id="h5",
                    suffix=".h5",
                ),
                FlexOutput(prefix="applycal_out/apply/ms", suffix=".ms.zip"),
                FlexOutput(
                    prefix="applycal_out/cal/logs",
                    suffix=".log",
                ),
                FlexOutput(
                    prefix="applycal_out/sub/logs",
                    suffix=".log",
                ),
                FlexOutput(
                    prefix="applycal_out/apply/logs",
                    suffix=".log",
                ),
            ],
        )

        imaging_stage = Stage(
            stage_id="imaging",
            func=imaging,
            max_concurrency=1,
            inputs=[FlexInput(prefix="applycal_out/apply/ms")],
            outputs=[
                FlexOutput(prefix="image_out", suffix="-image.fits"),
                FlexOutput(
                    prefix="image_out/logs",
                    suffix=".log",
                ),
            ],
        )

        rebinning_stage >> full_calibration_stage >> imaging_stage

        dag.add_stages(
            [
                rebinning_stage,
                full_calibration_stage,
                imaging_stage,
            ]
        )
        executor = DAGExecutor(
            dag,
            executor=FunctionExecutor(
                log_level="INFO", **{"runtime_memory": 2048, "runtime_cpu": 4}
            ),
        )
        results = executor.execute()

        i = 1
        for result in results.values():
            print(f"STAGE #{i}: {result.get_timings()}")
            i += 1


    main()
