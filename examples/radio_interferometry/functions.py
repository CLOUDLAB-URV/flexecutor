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
from flexecutor import StageContext

logger = logging.getLogger(__name__)


def dp3(ctx: StageContext):
    parameters = ctx.get_param("parameters")
    if type(parameters) is not list:
        parameters = [parameters]
    dp3_types = ctx.get_param("dp3_types")
    if type(dp3_types) is not list:
        dp3_types = [dp3_types]
    msout_path = Path(f"/tmp/{str(uuid.uuid4())[0:8]}-msout.ms")

    for params, dp3_type in zip(parameters, dp3_types):
        original_params = params.copy()
        msout_path = before_exec_dp3(params, msout_path, dp3_type, ctx)
        exec_dp3(params)
        after_exec_dp3(original_params, msout_path, dp3_type, ctx)


def imaging(ctx: StageContext):
    imaging_params = ctx.get_param("parameters")
    dst = ctx.next_output_path("image_out")
    dst = dst.removesuffix("-image.fits")
    imaging_params.append(dst)

    zip_paths = ctx.get_input_paths("imaging_input")
    for zip_path in zip_paths:
        ms_path = unzip(Path(zip_path))
        imaging_params.append(ms_path)

    define_home_if_unset()

    with open(ctx.next_output_path("image_out/logs"), "w") as log_file:
        proc = sp.Popen(
            ["wsclean"] + imaging_params, stdout=sp.PIPE, stderr=sp.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def before_exec_dp3(
    parameters, msout_path: Path, dp3_type: str, ctx: StageContext
) -> Path:
    """
    This function prepares the parameters for the DP3 execution.
    Depending on the DP3 type, it will unzip the input files and set the parameters accordingly.
    @param parameters: The parameters for the DP3 execution.
    @param msout_path: The path to the output MS file.
    @param dp3_type: rebinning | calibration | subtraction | apply_calibration
    @param ctx: the StageContext instance
    @return: The path to the output MS file.
    """
    parameters["log_output"] = ctx.next_output_path(parameters["log_output"].id)

    if dp3_type == "rebinning":
        parameters["msout"] = msout_path
        if ctx.is_dynamic_chunker(parameters["msin"].id):
            file = ctx.get_input_paths(parameters["msin"].id)[0]
            path = Path(file).read_text().strip()
            print("MS path: ", path)
            parameters["msin"] = path
        else:
            ms_zip = ctx.get_input_paths(parameters["msin"].id)[0]
            parameters["msin"] = unzip(Path(ms_zip))
        [parameters["aoflag.strategy"]] = ctx.get_input_paths(
            parameters["aoflag.strategy"].id
        )

    elif dp3_type == "calibration":
        ms_zip = ctx.get_input_paths(parameters["msin"].id)[0]
        msin_path = unzip(Path(ms_zip))
        
        input_paths = ctx.get_input_paths(parameters["cal.sourcedb"].id)
        
        if not input_paths:
            raise FileNotFoundError(f"No se encontró el archivo: {parameters['cal.sourcedb'].id} en {input_paths}")
        
        if len(input_paths) > 1:
            raise ValueError(f"Se encontraron múltiples archivos para {parameters['cal.sourcedb'].id}: {input_paths}")
        
        [step2a_zip] = input_paths
        
        #[step2a_zip] = ctx.get_input_paths(parameters["cal.sourcedb"].id)
        
        step2a_path = unzip(Path(step2a_zip))
        h5_path = "/tmp/cal.h5"
        parameters["msin"] = msin_path
        parameters["msout"] = msout_path
        parameters["cal.sourcedb"] = step2a_path
        if "cal.parmdb" in parameters:
            parameters["cal.parmdb"] = ctx.next_output_path(parameters["cal.parmdb"].id)
        else:
            parameters["cal.h5parm"] = h5_path

    elif dp3_type == "subtraction":
        [step2a_zip] = ctx.get_input_paths(parameters["sub.sourcedb"].id)
        step2a_path = step2a_zip.removesuffix(".zip")
        h5_path = "/tmp/cal.h5"
        parameters["msin"] = msout_path
        parameters["msout"] = "."
        parameters["sub.sourcedb"] = step2a_path
        parameters["sub.applycal.parmdb"] = h5_path

    elif dp3_type == "apply_calibration":
        h5_path = "/tmp/cal.h5"
        if "msin" in parameters:
            ms_zip = ctx.get_input_paths(parameters["msin"].id)[0]
            msin_path = unzip(Path(ms_zip))
            parameters["msin"] = msin_path
            msout_path = Path(msin_path)
        else:
            parameters["msin"] = msout_path
        parameters["msout"] = "."
        if "apply.parmdb" in parameters:
            parameters["apply.parmdb"] = ctx.get_input_paths(
                parameters["apply.parmdb"].id
            )[0]
        else:
            parameters["apply.parmdb"] = h5_path

    return msout_path


def exec_dp3(parameters):
    """
    This function executes the DP3 command with the given parameters.
    @param parameters: The parameters for the DP3 execution.
    """
    params_path = dict_to_parset(parameters)
    cmd = ["DP3", str(params_path)]
    print("Executing command: ", cmd)
    os.makedirs(os.path.dirname(parameters["log_output"]), exist_ok=True)

    define_home_if_unset()

    with open(parameters["log_output"], "w") as log_file:
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        stdout, stderr = proc.communicate()
        print(stdout)
        print(stderr)
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def define_home_if_unset():
    if "HOME" not in os.environ:
        os.environ["HOME"] = "/tmp"


def print_cmd_output(stderr, stdout):
    # print the output of the command
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)


def after_exec_dp3(params, msout_path: Path, dp3_type: str, ctx: StageContext):
    """
    This function prepares the output files after the DP3 execution.
    @param params: The parameters of the DP3 execution.
    @param msout_path: The path to the output MS file.
    @param dp3_type: rebinning | calibration | subtraction | apply_calibration
    @param ctx: the StageContext instance
    """

    if dp3_type in ["rebinning", "apply_calibration"]:
        zip_path = ctx.next_output_path(params["msout"].id)
        zip_name = zip_path.removesuffix(".zip")
        os.rename(msout_path, zip_name)
        my_zip(Path(zip_name), Path(zip_path))

    elif dp3_type == "calibration" or dp3_type == "subtraction":
        pass
