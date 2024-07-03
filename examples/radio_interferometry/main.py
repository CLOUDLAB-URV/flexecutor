from typing import Dict

from lithops import FunctionExecutor

from examples.radio_interferometry.dp3 import dp3
from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("radio-interferometry")

        rebinning_inputs: Dict[str, FlexInput]
        calibration_inputs: Dict[str, FlexInput]
        subtraction_inputs: Dict[str, FlexInput]
        apply_calibration_inputs: Dict[str, FlexInput]
        imaging_input: FlexInput
        rebinning_outputs: Dict[str, FlexOutput]
        calibration_outputs: Dict[str, FlexOutput]
        subtraction_outputs: Dict[str, FlexOutput]
        apply_calibration_outputs: Dict[str, FlexOutput]
        imaging_output: FlexOutput

        # TODO: set all custom_ids with dict iteration
        rebinning_inputs = {
            "partitions": FlexInput(prefix="partitions"),
            "lua": FlexInput(
                # prefix="parameters/rebinning/STEP1-NenuFAR64C1S.lua",
                prefix="parameters/rebinning",
                custom_input_id="lua",
                # TODO: check broadcast files in use-case
                # strategy=StrategyEnum.BROADCAST,
            ),
        }
        rebinning_outputs = {
            "result": FlexOutput(
                prefix="rebinning_out/ms",
                suffix=".ms",
            ),
            "logs": FlexOutput(
                prefix="rebinning_out/logs",
                suffix=".log",
            ),
        }

        rebinning_params = {
            # "msin": rebinning_inputs["partitions"],
            "msin": "/tmp/test-bucket/partitions/SB210.ms.zip",
            "steps": "[aoflag, avg, count]",
            "aoflag.type": "aoflagger",
            "aoflag.strategy": rebinning_inputs["lua"],
            "avg.type": "averager",
            "avg.freqstep": 4,
            "avg.timestep": 8,
            # "msout": rebinning_outputs["result"],
            "msout": "/tmp/test-bucket/rebinning_out/SB210.ms",
            "numthreads": 4,
            "log_output": rebinning_outputs["logs"],
        }
        #
        # calibration_inputs = {
        #     "msin": FlexInput(
        #         prefix="rebinning_out/ms",
        #     ),
        #     "sourcedb": FlexInput(
        #         prefix="parameters/calibration/STEP2A-apparent.sourcedb",
        #     ),
        # }
        #
        # calibration_outputs = {
        #     "h5": FlexOutput(
        #         prefix="applycal_out/cal/h5",
        #         suffix=".h5",
        #     ),
        #     "msout": FlexOutput(prefix="applycal_out/ms", suffix=".ms"),
        #     "logs": FlexOutput(
        #         prefix="applycal_out/cal/logs",
        #         suffix=".log",
        #     ),
        # }
        #
        # calibration_params = {
        #     "msin": calibration_inputs["msin"],
        #     "msin.datacolumn": "DATA",
        #     "steps": "[cal]",
        #     "cal.type": "ddecal",
        #     "cal.mode": "diagonal",
        #     "cal.sourcedb": calibration_inputs["sourcedb"],
        #     "cal.h5parm": calibration_outputs["h5"],
        #     "cal.solint": 4,
        #     "cal.nchan": 4,
        #     "cal.maxiter": 50,
        #     "cal.uvlambdamin": 5,
        #     "cal.smoothnessconstraint": 2e6,
        #     "numthreads": 4,
        #     "msout": calibration_outputs["result"],
        #     "log_output": calibration_outputs["logs"],
        # }
        #
        # subtraction_inputs = {
        #     "msin": FlexInput(
        #         prefix="applycal_out/ms",
        #     ),
        #     "sourcedb": FlexInput(
        #         prefix="parameters/calibration/STEP2A-apparent.sourcedb",
        #     ),
        #     "h5": FlexInput(
        #         prefix="applycal_out/cal/h5",
        #     ),
        # }
        #
        # subtraction_outputs = {
        #     "msout": FlexOutput(prefix="applycal_out/ms", suffix=".ms"),
        #     "logs": FlexOutput(
        #         prefix="applycal_out/substract/logs",
        #         suffix=".log",
        #     ),
        # }
        #
        # subtraction = {
        #     "msin": subtraction_inputs["msin"],
        #     "msin.datacolumn": "DATA",
        #     "msout.datacolumn": "SUBTRACTED_DATA",
        #     "steps": "[sub]",
        #     "sub.type": "h5parmpredict",
        #     "sub.sourcedb": subtraction_inputs["sourcedb"],
        #     "sub.directions": "[[CygA],[CasA]]",
        #     "sub.operation": "subtract",
        #     "sub.applycal.parmdb": subtraction_inputs["h5"],
        #     "sub.applycal.steps": "[sub_apply_amp,sub_apply_phase]",
        #     "sub.applycal.correction": "fulljones",
        #     "sub.applycal.sub_apply_amp.correction": "amplitude000",
        #     "sub.applycal.sub_apply_phase.correction": "phase000",
        #     "msout": subtraction_outputs["msout"],
        #     "log_output": subtraction_outputs["logs"],
        # }
        #
        # apply_calibration_inputs = {
        #     "msin": FlexInput(
        #         prefix="applycal_out/ms",
        #     ),
        #     "h5": FlexInput(
        #         prefix="applycal_out/cal/h5",
        #     ),
        # }
        #
        # apply_calibration_outputs = {
        #     "msout": FlexOutput(prefix="applycal_out/ms", suffix=".ms"),
        #     "logs": FlexOutput(
        #         prefix="applycal_out/apply/logs",
        #         suffix=".log",
        #     ),
        # }
        #
        # apply_calibration = {
        #     "msin": apply_calibration_inputs["msin"],
        #     "msin.datacolumn": "SUBTRACTED_DATA",
        #     "msout": apply_calibration_outputs["msout"],
        #     "msout.datacolumn": "CORRECTED_DATA",
        #     "steps": "[apply]",
        #     "apply.type": "applycal",
        #     "apply.steps": "[apply_amp,apply_phase]",
        #     "apply.apply_amp.correction": "amplitude000",
        #     "apply.apply_phase.correction": "phase000",
        #     "apply.direction": "[Main]",
        #     "apply.parmdb": apply_calibration_inputs["h5"],
        #     "log_output": apply_calibration_outputs["logs"],
        # }
        #
        # imaging_input = FlexInput(prefix="applycal_out/ms")
        # imaging_output = FlexOutput(prefix="image_out")
        #
        # # Imaging parameters with hash included in the key as a root directory
        # imaging_params = [
        #     "-size",
        #     "1024",
        #     "1024",
        #     "-pol",
        #     "I",
        #     "-scale",
        #     "2arcmin",
        #     "-niter",
        #     "100000",
        #     "-gain",
        #     "0.1",
        #     "-mgain",
        #     "0.6",
        #     "-auto-mask",
        #     "5",
        #     "-local-rms",
        #     "-multiscale",
        #     "-no-update-model-required",
        #     "-make-psf",
        #     "-auto-threshold",
        #     "3",
        #     "-parallel-deconvolution",
        #     "4096",
        #     "-weight",
        #     "briggs",
        #     "0",
        #     "-data-column",
        #     "CORRECTED_DATA",
        #     "-nmiter",
        #     "0",
        #     "-name",
        #     imaging_output,
        # ]

        stage1 = Stage(
            stage_id="rebinning",
            func=dp3,
            inputs=list(rebinning_inputs.values()),
            outputs=list(rebinning_outputs.values()),
            params=rebinning_params
        )

        # stage2 = Stage(
        #     stage_id="calibration",
        #     func=calibration,
        #     inputs=[],
        #     outputs=[],
        # )
        # stage3 = Stage(
        #     stage_id="imaging",
        #     func=imaging,
        #     inputs=[],
        #     outputs=[],
        # )

        # stage1 >> stage2 >> stage3

        dag.add_stages([stage1])
        executor = DAGExecutor(
            dag,
            executor=FunctionExecutor(
                log_level="DEBUG", **{"runtime_memory": 2048, "runtime_cpu": 4}
            ),
        )
        results = executor.execute()
        print(results["stage1"].get_timings())

    main()
