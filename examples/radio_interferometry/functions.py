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


def dp3(io: IOManager):
    parameters = io.get_param("parameters")
    if type(parameters) is not list:
        parameters = [parameters]
    dp3_types = io.get_param("dp3_types")
    if type(dp3_types) is not list:
        dp3_types = [dp3_types]
    msout_path = Path(f"/tmp/{str(uuid.uuid4())[0:8]}-msout.ms")

    for params, dp3_type in zip(parameters, dp3_types):
        before_exec_dp3(params, msout_path, dp3_type, io)
        exec_dp3(params)
        after_exec_dp3(params, msout_path, dp3_type, io)


def imaging(io: IOManager):
    imaging_params = io.get_param("parameters")
    dst = io.next_output_path("image_out")
    dst = dst.removesuffix("-image.fits")
    imaging_params.append(dst)

    zip_paths = io.get_input_paths("applycal_out/apply/ms")
    for zip_path in zip_paths:
        ms_path = unzip(Path(zip_path))
        imaging_params.append(ms_path)

    with open(io.next_output_path("image_out/logs"), "w") as log_file:
        proc = sp.Popen(
            ["wsclean"] + imaging_params, stdout=sp.PIPE, stderr=sp.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def before_exec_dp3(parameters, msout_path: Path, dp3_type: str, io: IOManager):
    """
    This function prepares the parameters for the DP3 execution.
    Depending on the DP3 type, it will unzip the input files and set the parameters accordingly.
    @param parameters: The parameters for the DP3 execution.
    @param msout_path: The path to the output MS file.
    @param dp3_type: rebinning | calibration | subtraction | apply_calibration
    @param io: the IOManager instance
    """
    if dp3_type == "rebinning":
        msout_path = Path(f"/tmp/{str(uuid.uuid4())[0:8]}-msout.ms")
        parameters["msout"] = msout_path
        ms_zip = io.get_input_paths("partitions")[0]
        parameters["msin"] = unzip(Path(ms_zip))
        [parameters["aoflag.strategy"]] = io.get_input_paths("lua")
        parameters["log_output"] = io.next_output_path("rebinning_out/logs")

    elif dp3_type == "calibration":
        ms_zip = io.get_input_paths("rebinning_out/ms")[0]
        msin_path = unzip(Path(ms_zip))
        [step2a_zip] = io.get_input_paths("step2a")
        step2a_path = unzip(Path(step2a_zip))
        h5_path = "/tmp/cal.h5"
        parameters["msin"] = msin_path
        parameters["msout"] = msout_path
        parameters["cal.sourcedb"] = step2a_path
        parameters["log_output"] = io.next_output_path("applycal_out/cal/logs")
        parameters["cal.h5parm"] = h5_path

    elif dp3_type == "subtraction":
        [step2a_zip] = io.get_input_paths("step2a")
        step2a_path = step2a_zip.removesuffix(".zip")
        h5_path = "/tmp/cal.h5"
        parameters["msin"] = msout_path
        parameters["msout"] = "."
        parameters["sub.sourcedb"] = step2a_path
        parameters["sub.applycal.parmdb"] = h5_path
        parameters["log_output"] = io.next_output_path("applycal_out/sub/logs")

    elif dp3_type == "apply_calibration":
        h5_path = "/tmp/cal.h5"
        parameters["msin"] = msout_path
        parameters["msout"] = "."
        parameters["apply.parmdb"] = h5_path
        parameters["log_output"] = io.next_output_path("applycal_out/apply/logs")


def exec_dp3(parameters):
    """
    This function executes the DP3 command with the given parameters.
    @param parameters: The parameters for the DP3 execution.
    """
    params_path = dict_to_parset(parameters)
    cmd = ["DP3", str(params_path)]
    print("Executing command: ", cmd)
    os.makedirs(os.path.dirname(parameters["log_output"]), exist_ok=True)
    with open(parameters["log_output"], "w") as log_file:
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        stdout, stderr = proc.communicate()
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def after_exec_dp3(params, msout_path: Path, dp3_type: str, io: IOManager):
    """
    This function prepares the output files after the DP3 execution.
    @param params: The parameters of the DP3 execution.
    @param msout_path: The path to the output MS file.
    @param dp3_type: rebinning | calibration | subtraction | apply_calibration
    @param io: the IOManager instance
    """
    if dp3_type == "rebinning":
        zip_path = io.next_output_path("rebinning_out/ms")
        zip_name = zip_path.removesuffix(".zip")
        os.rename(params["msout"], zip_name)
        my_zip(Path(zip_name), Path(zip_path))

    elif dp3_type == "calibration" or dp3_type == "subtraction":
        pass

    elif dp3_type == "apply_calibration":
        zip_path = io.next_output_path("applycal_out/apply/ms")
        zip_name = zip_path.removesuffix(".zip")
        os.rename(msout_path, zip_name)
        my_zip(Path(zip_name), Path(zip_path))
