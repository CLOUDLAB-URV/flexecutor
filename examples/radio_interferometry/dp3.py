import logging
from pathlib import Path
from typing import Dict, List

from examples.radio_interferometry.utils import unzip, dict_to_parser, my_zip
from flexecutor.utils.iomanager import IOManager

import subprocess as sp


def dp3(io: IOManager):
    logger = logging.Logger

    [partition_path] = io.get_input_paths("partitions")
    partition_path = unzip(Path(partition_path), logger)
    [lua_path] = io.get_input_paths("lua")
    parameters: Dict = io.get_param("parameters")
    params_path = dict_to_parser(parameters)

    cmd = ["DP3", str(params_path)]

    with open(parameters["log_output"], "w") as log_file:
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        stdout, stderr = proc.communicate()
        log_file.write(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    msout_path = parameters["msout"]
    output_path = Path(io.next_output_path("results"))
    my_zip(msout_path, output_path, logger)

