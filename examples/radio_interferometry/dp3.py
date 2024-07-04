import logging
import os
import subprocess as sp
import uuid
from pathlib import Path

from examples.radio_interferometry.utils import (
    unzip,
    dict_to_parset,
    my_zip,
)
from flexecutor.utils.iomanager import IOManager

logger = logging.getLogger(__name__)


def rebinning(io: IOManager):
    parameters = {
        "steps": "[aoflag, avg, count]",
        "aoflag.type": "aoflagger",
        "avg.type": "averager",
        "avg.freqstep": 4,
        "avg.timestep": 8,
        "numthreads": 4,
    }

    msout_path = Path(f"/tmp/{str(uuid.uuid4())[0:8]}-msout.ms")
    parameters["msout"] = msout_path
    ms_zip = io.get_input_paths("partitions")[0]
    parameters["msin"] = unzip(Path(ms_zip), logger)
    [parameters["aoflag.strategy"]] = io.get_input_paths("lua")
    parameters["log_output"] = io.next_output_path("rebinning_out/logs")

    execute_dp3(parameters)

    zip_path = io.next_output_path("rebinning_out/ms")
    zip_name = zip_path[:-4]
    os.rename(msout_path, zip_name)
    my_zip(Path(zip_name), Path(zip_path))


def calibration(io: IOManager):
    parameters = {
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

    msout_path = Path(f"/tmp/{str(uuid.uuid4())[0:8]}-msout.ms")
    parameters["msout"] = msout_path
    ms_zip = io.get_input_paths("rebinning_out/ms")[0]
    print(ms_zip)
    parameters["msin"] = unzip(Path(ms_zip), logger)
    # check if parameters["msin"] directory is created
    print(parameters["msin"].is_dir())
    [step2a_zip] = io.get_input_paths("step2a")
    parameters["cal.sourcedb"] = unzip(Path(step2a_zip), logger)
    parameters["log_output"] = io.next_output_path("applycal_out/cal/logs")
    parameters["cal.h5parm"] = io.next_output_path("h5")

    execute_dp3(parameters)

    # STEP4: zip the output
    zip_path = io.next_output_path("applycal_out/cal/ms")
    zip_name = zip_path[:-4]
    os.rename(msout_path, zip_name)
    my_zip(Path(zip_name), Path(zip_path))


def subtraction(io: IOManager):
    parameters = {
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

    ms_zip = io.get_input_paths("applycal_out/cal/ms")[0]
    print(ms_zip)
    parameters["msin"] = unzip(Path(ms_zip), logger)
    [step2a_zip] = io.get_input_paths("step2a")
    parameters["sub.sourcedb"] = unzip(Path(step2a_zip), logger)
    [parameters["sub.applycal.parmdb"]] = io.get_input_paths("h5")
    parameters["log_output"] = io.next_output_path("applycal_out/sub/logs")

    execute_dp3(parameters)

    zip_path = io.next_output_path("applycal_out/sub/ms")
    zip_name = zip_path[:-4]
    os.rename(parameters["msin"], zip_name)
    my_zip(Path(zip_name), Path(zip_path))


def apply_calibration(io: IOManager):
    parameters = {
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

    ms_zip = io.get_input_paths("applycal_out/sub/ms")[0]
    print(ms_zip)
    parameters["msin"] = unzip(Path(ms_zip), logger)
    [parameters["apply.parmdb"]] = io.get_input_paths("h5")
    parameters["log_output"] = io.next_output_path("applycal_out/apply/logs")

    execute_dp3(parameters)

    zip_path = io.next_output_path("applycal_out/apply/ms")
    zip_name = zip_path[:-4]
    os.rename(parameters["msin"], zip_name)
    my_zip(Path(zip_name), Path(zip_path))


def imaging(io: IOManager):
    cpus = 5
    imaging_params = [
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
        str(cpus),
        "-name",
    ]

    dst = io.next_output_path("image_out")
    dst = dst[: -len("-image.fits")]
    imaging_params.append(dst)

    zip_paths = io.get_input_paths("applycal_out/apply/ms")
    for zip_path in zip_paths:
        ms_path = unzip(Path(zip_path), logger)
        imaging_params.append(ms_path)

    with open(io.next_output_path("image_out/logs"), "w") as log_file:
        proc = sp.Popen(
            ["wsclean"] + imaging_params, stdout=sp.PIPE, stderr=sp.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def execute_dp3(parameters):
    params_path = dict_to_parset(parameters)
    cmd = ["DP3", str(params_path)]
    print("Executing command: ", cmd)
    os.makedirs(os.path.dirname(parameters["log_output"]), exist_ok=True)
    with open(parameters["log_output"], "w") as log_file:
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        stdout, stderr = proc.communicate()
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")
