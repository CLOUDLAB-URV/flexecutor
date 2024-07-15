import inspect
import os
from typing import TYPE_CHECKING
import json
from enum import Enum
from typing import List
from flexecutor.utils.dataclass import FunctionTimes, StageConfig

if TYPE_CHECKING:
    from flexecutor.workflow.stage import Stage


class AssetType(Enum):
    MODEL = ("model", ".pkl")
    PROFILE = ("profiling", ".json")
    IMAGE = ("image", ".png")


FLEXECUTOR_EXEC_PATH = "FLEXECUTOR_EXEC_PATH"


def get_my_exec_path():
    return os.environ.get(FLEXECUTOR_EXEC_PATH, None)


# FIXME: This is a bit messy, we should have a class to handle profiling
def get_asset_path(stage: "Stage", asset_type: AssetType):
    dir_name, file_extension = asset_type.value
    os.makedirs(f"{get_my_exec_path()}/{dir_name}/{stage.dag_id}", exist_ok=True)
    return f"{get_my_exec_path()}/{dir_name}/{stage.dag_id}/{stage.stage_id}{file_extension}"


def store_profiling(
    file: str, new_profile_data: List[FunctionTimes], resource_config: StageConfig
):
    profile_data = load_profiling_results(file)
    config_key = str(resource_config.key)
    if config_key not in profile_data:
        profile_data[config_key] = {}
    for key in FunctionTimes.profile_keys():
        if key not in profile_data[config_key]:
            profile_data[config_key][key] = []
        profile_data[config_key][key].append([])
    for profiling in new_profile_data:
        for key in FunctionTimes.profile_keys():
            profile_data[config_key][key][-1].append(getattr(profiling, key))
    save_profiling_results(file, profile_data)


def load_profiling_results(file: str) -> dict:
    file = os.path.join(get_my_exec_path(), file)
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_profiling_results(file: str, profile_data: dict):
    serial_data = {str(k): v for k, v in profile_data.items()}
    with open(file, "w") as f:
        json.dump(serial_data, f, indent=4)


def flexorchestrator(bucket=""):
    """
    Decorator to initializations previous to the execution of user scripts.
    You must use it only in the main function of your script.
    Responsible for:
    - Set the path if where the flexorchestrator main script is located
    :param bucket:
    :return:
    """

    def function(func):
        def wrapper(*args, **kwargs):
            # Set the path of the flexorchestrator file
            key = FLEXECUTOR_EXEC_PATH
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_file = caller_frame.f_globals["__file__"]
            value = os.path.dirname(os.path.abspath(caller_file))
            os.environ[key] = value
            # Set the bucket
            key = "FLEX_BUCKET"
            os.environ[key] = bucket
            try:
                result = func(*args, **kwargs)
            finally:
                os.environ.pop(key, None)
            return result

        return wrapper

    return function
