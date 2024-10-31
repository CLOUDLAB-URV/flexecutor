import os
import zipfile
from pathlib import Path, PosixPath
from typing import Union, List, Dict, Type

from flexecutor.storage.storage import FlexData


def unzip(ms: Path) -> Path:
    print(f"Extracting zip file at {ms}")
    if ms.suffix != ".zip":
        print(f"Expected a .zip file, got {ms}")
        raise ValueError(f"Expected a .zip file, got {ms}")

    extract_path = ms.parent
    print(f"Extracting to directory: {extract_path}")

    with zipfile.ZipFile(ms, "r") as zipf:
        zipf.extractall(extract_path)
        extracted_dir = ms.parent / ms.stem

    ms.unlink()
    print(f"Deleted zip file at {ms}")

    print(f"Extracted to directory: {extracted_dir}")
    return extracted_dir


def my_zip(ms: Path, dst: Path) -> Path:
    print(f"Starting zipping process for: {ms}")

    if dst.exists() and dst.is_dir():
        print(
            f"Cannot create a zip file as a directory with the name {dst} exists."
        )
        raise IsADirectoryError(
            f"Cannot create a zip file as a directory with the name {dst} exists."
        )

    if ms.is_dir():
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zipf:
            for root, dirs, files in os.walk(ms):
                for file in files:
                    file_path = Path(root) / file
                    arcname = ms.name / file_path.relative_to(ms)
                    zipf.write(file_path, arcname)
    else:
        print(f"{ms} is not a directory.")
        raise NotADirectoryError(f"Expected a directory, got {ms}")

    print(f"Created zip file at {dst}")
    return dst


def get_dir_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def dict_to_parset(
    data, output_dir=PosixPath("/tmp"), filename="output.parser"
) -> PosixPath:
    lines = []

    for key, value in data.items():
        # Check if the value is another dictionary
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):  # Check for nested dictionaries
                    lines.append(f"[{sub_key}]")
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        lines.append(f"{sub_sub_key} = {sub_sub_value}")
                else:
                    lines.append(f"{sub_key} = {sub_value}")
        else:
            lines.append(f"{key} = {value}")

    # Convert the list of lines to a single string
    parset_content = "\n".join(lines)

    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with output_path.open("w") as file:
        file.write(parset_content)

    return output_path


def filter_io_params(
    parameters: Union[List[Dict] | Dict],
    flex_type: Type[Union["FlexInput", "FlexOutput"]],
):
    if type(parameters) is not list:
        parameters = [parameters]
    all_values = [value for parameter in parameters for value in parameter.values()]
    flex_values = [value for value in all_values if type(value) is flex_type]
    return list(set(flex_values))


# FlexInput and FlexOutput classes are a way for distinguishing between
# input and output data in the radio-interferometry workflow
class FlexInput(FlexData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FlexOutput(FlexData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

