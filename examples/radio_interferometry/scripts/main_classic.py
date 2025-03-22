from flexecutor.storage.chunker import Chunker
from flexecutor.utils.enums import ChunkerTypeEnum
from lithops import FunctionExecutor

from dataplug.formats.astronomics.ms import MS, ms_partitioning_strategy

from examples.radio_interferometry.functions import (
    imaging,
    dp3,
)
from examples.radio_interferometry.utils import FlexInput, FlexOutput
from examples.radio_interferometry.utils import filter_io_params
from flexecutor.storage.storage import StrategyEnum
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        rebinning_parameters = {
            "msin": FlexInput(prefix="partitions-nozip", custom_data_id="partitions", chunker=Chunker(ChunkerTypeEnum.DYNAMIC, chunking_strategy=ms_partitioning_strategy, cloud_object_format=MS)),
            "steps": "[aoflag, avg, count]",
            "aoflag.type": "aoflagger",
            "aoflag.strategy": FlexInput(
                prefix="parameters/rebinning",
                custom_data_id="lua",
                read_strategy=StrategyEnum.BROADCAST,
            ),
            "avg.type": "averager",
            "avg.freqstep": 4,
            "avg.timestep": 8,
            # "avg.freqstep": 2,
            # "avg.timestep": 2,
            "numthreads": 4,
            "msout": FlexOutput(
                prefix="rebinning_out/ms",
                custom_data_id="rebinning_ms",
                suffix=".ms.zip",
            ),
            "log_output": FlexOutput(
                prefix="rebinning_out/logs",
                suffix=".log",
            ),
        }

        calibration_parameters = {
            "msin": FlexInput(
                prefix="rebinning_out/ms", custom_data_id="rebinning_ms"
            ),
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
            "cal.sourcedb": FlexInput(
                prefix="parameters/calibration/step2a",
                custom_data_id="step2a",
                read_strategy=StrategyEnum.BROADCAST,
            ),
            "log_output": FlexOutput(
                prefix="applycal_out/cal/logs",
                suffix=".log",
            ),
        }

        subtraction_parameters = {
            # "msin" is the output of the calibration stage
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
            "sub.sourcedb": FlexInput(
                prefix="parameters/calibration/step2a",
                custom_data_id="step2a",
                read_strategy=StrategyEnum.BROADCAST,
            ),
            "log_output": FlexOutput(
                prefix="applycal_out/sub/logs",
                suffix=".log",
            ),
        }

        apply_calibration_parameters = {
            # "msin" is the output of the subtraction stage
            "msin.datacolumn": "SUBTRACTED_DATA",
            "msout.datacolumn": "CORRECTED_DATA",
            "steps": "[apply]",
            "apply.type": "applycal",
            "apply.steps": "[apply_amp,apply_phase]",
            "apply.apply_amp.correction": "amplitude000",
            "apply.apply_phase.correction": "phase000",
            "apply.direction": "[Main]",
            "msout": FlexOutput(
                prefix="applycal_out/apply/ms",
                suffix=".ms.zip",
            ),
            "log_output": FlexOutput(
                prefix="applycal_out/apply/logs",
                suffix=".log",
            ),
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
            inputs=filter_io_params(rebinning_parameters, FlexInput),
            outputs=filter_io_params(rebinning_parameters, FlexOutput),
            params={"parameters": rebinning_parameters, "dp3_types": "rebinning"},
        )

        full_calibration_parameters = [
            calibration_parameters,
            subtraction_parameters,
            apply_calibration_parameters,
        ]
        full_calibration_stage = Stage(
            stage_id="full_calibration",
            func=dp3,
            inputs=filter_io_params(full_calibration_parameters, FlexInput),
            outputs=filter_io_params(full_calibration_parameters, FlexOutput),
            params={
                "parameters": full_calibration_parameters,
                "dp3_types": ["calibration", "subtraction", "apply_calibration"],
            },
        )

        imaging_stage = Stage(
            stage_id="imaging",
            func=imaging,
            max_concurrency=1,
            inputs=[
                FlexInput(
                    prefix="applycal_out/apply/ms", custom_data_id="imaging_input"
                )
            ],
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
                log_level="DEBUG", **{"runtime_memory": 2048, "runtime_cpu": 4}
            ),
        )
        results = executor.execute(num_workers=1) #num workers?

        i = 1
        for result in results.values():
            print(f"STAGE #{i}: {result.get_timings()}")
            i += 1

    main()
