from examples.radio_interferometry.functions import dp3, imaging
from examples.radio_interferometry.utils import filter_io_params
from flexecutor.storage.storage import FlexInput, FlexOutput, StrategyEnum
from flexecutor.workflow.stage import Stage


cal_rebinning_parameters = {
    "msin": FlexInput(prefix="partitions-CAL"),
    "steps": "[aoflag, avg, count]",
    "aoflag.type": "aoflagger",
    "aoflag.strategy": FlexInput(
        prefix="parameters/rebinning",
        custom_data_id="lua",
        read_strategy=StrategyEnum.BROADCAST,
    ),
    "avg.type": "averager",
    "avg.freqstep": 5,
    "avg.timestep": 2,
    "msout": FlexOutput(
        prefix="CAL/rebinning_out/ms",
        suffix=".ms.zip",
    ),
    "numthreads": 4,
    "log_output": FlexOutput(
        prefix="CAL/rebinning_out/logs",
        suffix=".log",
    ),
}

cal_calibration_params = {
    "msin": FlexInput(prefix="CAL/rebinning_out/ms"),
    "msin.datacolumn": "DATA",
    "msout": ".",
    "steps": "[cal]",
    "cal.type": "gaincal",
    "cal.caltype": "diagonal",
    "cal.sourcedb": FlexInput(
        prefix="parameters/calibration/step2a",
        custom_data_id="step2a",
        read_strategy=StrategyEnum.BROADCAST,
    ),
    "cal.parmdb": FlexOutput(
        prefix="CAL/calibration_out/h5",
        suffix=".h5",
    ),
    "cal.solint": 0,  # means 1 solution for all time steps
    "cal.nchan": 1,  # means 1 solution per channel
    "cal.maxiter": 50,
    "cal.uvlambdamin": 5,
    "cal.smoothnessconstraint": 2e6,
    "numthreads": 4,
    "log_output": FlexOutput(
        prefix="CAL/calibration_out/logs",
        suffix=".log",
    ),
}

target_rebinning_params = {
    "msin": FlexInput(prefix="partitions-TAR"),
    "steps": "[aoflag, avg, count]",
    "aoflag.type": "aoflagger",
    "aoflag.strategy": FlexInput(
        prefix="parameters/rebinning",
        custom_data_id="lua",
        read_strategy=StrategyEnum.BROADCAST,
    ),
    "avg.type": "averager",
    "avg.freqstep": 5,  # averaging 5 channels
    "avg.timestep": 2,  # averaging 2 times samples
    "msout": FlexOutput(
        prefix="TAR/rebinning_out/ms",
        suffix=".ms.zip",
    ),
    "numthreads": 4,
    "log_output": FlexOutput(
        prefix="TAR/rebinning_out/logs",
        suffix=".log",
    ),
}

target_apply_calibration_params = {
    "msin": FlexInput(prefix="TAR/rebinning_out/ms"),
    "msin.datacolumn": "DATA",
    "msout": FlexOutput(
        prefix="TAR/applycal_out/ms",
        suffix=".ms.zip",
    ),
    "msout.datacolumn": "CORRECTED_DATA",
    "steps": "[apply]",
    "apply.type": "applycal",
    "apply.steps": "[apply_amp,apply_phase]",
    "apply.apply_amp.correction": "amplitude000",
    "apply.apply_phase.correction": "phase000",
    "apply.direction": "[Main]",
    "apply.parmdb": FlexInput(prefix="CAL/calibration_out/h5"),
    "log_output": FlexOutput(
        prefix="TAR/applycal_out/logs",
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
    "5arcmin",
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

cal_rebinning_stage = Stage(
    stage_id="CAL-rebinning",
    func=dp3,
    inputs=filter_io_params(cal_rebinning_parameters, FlexInput),
    outputs=filter_io_params(cal_rebinning_parameters, FlexOutput),
    params={"parameters": cal_rebinning_parameters, "dp3_types": "rebinning"},
)

cal_calibration_stage = Stage(
    stage_id="CAL-calibration",
    func=dp3,
    inputs=filter_io_params(cal_calibration_params, FlexInput),
    outputs=filter_io_params(cal_calibration_params, FlexOutput),
    params={
        "parameters": cal_calibration_params,
        "dp3_types": "calibration",
    },
)

target_rebinning_stage = Stage(
    stage_id="TARGET-rebinning",
    func=dp3,
    inputs=filter_io_params(target_rebinning_params, FlexInput),
    outputs=filter_io_params(target_rebinning_params, FlexOutput),
    params={
        "parameters": target_rebinning_params,
        "dp3_types": "rebinning",
    },
)

target_apply_calibration_stage = Stage(
    stage_id="TARGET-apply-calibration",
    func=dp3,
    inputs=filter_io_params(target_apply_calibration_params, FlexInput),
    outputs=filter_io_params(target_apply_calibration_params, FlexOutput),
    params={
        "parameters": target_apply_calibration_params,
        "dp3_types": "apply_calibration",
    },
)

imaging_stage = Stage(
    stage_id="imaging",
    func=imaging,
    max_concurrency=1,
    inputs=[FlexInput(prefix="TAR/applycal_out/ms", custom_data_id="imaging_input")],
    outputs=[
        FlexOutput(prefix="image_out", suffix="-image.fits"),
        FlexOutput(prefix="image_out/logs", suffix=".log"),
    ],
    params={
        "parameters": imaging_parameters,
    },
)


@task(returns=int)
def cal_rebinning():
    print(cal_rebinning_stage.execute().get_timings())
    return 1


@task(returns=int)
def cal_calibration(cal_rebinning_output):
    print(cal_calibration_stage.execute().get_timings())
    return 1


@task(returns=int)
def target_rebinning():
    print(target_rebinning_stage.execute().get_timings())
    return 1


@task(returns=int)
def target_calibration(target_rebinning_output, calibrator_calibration_output):
    print(target_apply_calibration_stage.execute().get_timings())
    return 1


@task(returns=int)
def target_imaging(target_calibration_output):
    print(imaging_stage.execute().get_timings())
    return 1


if __name__ == "__main__":
    print("Starting workflow")

    calibrator_rebinning_output = cal_rebinning()
    calibrator_calibration_output = cal_calibration(calibrator_rebinning_output)
    target_rebinning_output = target_rebinning()
    target_calibration_output = target_calibration(
        target_rebinning_output, calibrator_calibration_output
    )
    target_imaging_output = target_imaging(target_calibration_output)
