from lithops import FunctionExecutor

from examples.radio_interferometry.functions import (
    imaging,
    dp3,
)
from flexecutor.storage.storage import FlexInput, FlexOutput, StrategyEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        rebinning_parameters = {
            "steps": "[aoflag, avg, count]",
            "aoflag.type": "aoflagger",
            "avg.type": "averager",
            "avg.freqstep": 4,
            "avg.timestep": 8,
            "numthreads": 4,
        }

        calibration_parameters = {
            "msin.datacolumn": "DATA",
            "steps": "[cal]",
            "cal.type": "ddecal",
            "cal.mode": "diagonal",
            "cal.solint": 4,
            "cal.nchan": 4,
            "cal.maxiter": 50,
            "cal.uvlambdamin": 5,
            "cal.smoothnessconstraint": 2e6,
            "numthreads": 4,
        }

        subtraction_parameters = {
            "msin.datacolumn": "DATA",
            "msout.datacolumn": "SUBTRACTED_DATA",
            "steps": "[sub]",
            "sub.type": "h5parmpredict",
            "sub.directions": "[[CygA],[CasA]]",
            "sub.operation": "subtract",
            "sub.applycal.steps": "[sub_apply_amp,sub_apply_phase]",
            "sub.applycal.correction": "fulljones",
            "sub.applycal.sub_apply_amp.correction": "amplitude000",
            "sub.applycal.sub_apply_phase.correction": "phase000",
            "msout": ".",
        }

        apply_calibration_parameters = {
            "msin.datacolumn": "SUBTRACTED_DATA",
            "msout.datacolumn": "CORRECTED_DATA",
            "msout": ".",
            "steps": "[apply]",
            "apply.type": "applycal",
            "apply.steps": "[apply_amp,apply_phase]",
            "apply.apply_amp.correction": "amplitude000",
            "apply.apply_phase.correction": "phase000",
            "apply.direction": "[Main]",
        }

        imaging_parameters = [
            "-size",
            "1024",
            "1024",
            "-pol",
            "I",
            "-scale",
            "2arcmin",
            "-niter",
            "100000",
            "-gain",
            "0.1",
            "-mgain",
            "0.6",
            "-auto-mask",
            "5",
            "-local-rms",
            "-multiscale",
            "-no-update-model-required",
            "-make-psf",
            "-auto-threshold",
            "3",
            "-parallel-deconvolution",
            "4096",
            "-weight",
            "briggs",
            "0",
            "-data-column",
            "CORRECTED_DATA",
            "-nmiter",
            "0",
            "-j",
            str(5),
            "-name",
        ]

        dag = DAG("radio-interferometry")

        rebinning_stage = Stage(
            stage_id="rebinning",
            func=dp3,
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
            params={"parameters": rebinning_parameters, "dp3_types": "rebinning"},
        )

        full_calibration_stage = Stage(
            stage_id="full_calibration",
            func=dp3,
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
            params={
                "parameters": [
                    calibration_parameters,
                    subtraction_parameters,
                    apply_calibration_parameters,
                ],
                "dp3_types": ["calibration", "subtraction", "apply_calibration"],
            },
        )

        imaging_stage = Stage(
            stage_id="imaging",
            func=imaging,
            max_concurrency=1,
            inputs=[FlexInput(prefix="applycal_out/apply/ms")],
            outputs=[
                FlexOutput(prefix="image_out", suffix="-image.fits"),
                FlexOutput(prefix="image_out/logs", suffix=".log"),
            ],
            params={
                "parameters": imaging_parameters,
            },
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
