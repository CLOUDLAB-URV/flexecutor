import os
import zipfile
from pathlib import Path, PosixPath


def unzip(ms: Path, logger) -> Path:
    logger.info(f"Extracting zip file at {ms}")
    if ms.suffix != ".zip":
        logger.error(f"Expected a .zip file, got {ms}")
        raise ValueError(f"Expected a .zip file, got {ms}")

    extract_path = ms.parent
    logger.info(f"Extracting to directory: {extract_path}")

    with zipfile.ZipFile(ms, "r") as zipf:
        zip_contents = zipf.namelist()
        logger.debug(f"Zip contents: {zip_contents}")

        zipf.extractall(extract_path)

        extracted_dir = extract_path / ms.stem

    ms.unlink()
    logger.info(f"Deleted zip file at {ms}")

    logger.info(f"Extracted to directory: {extracted_dir}")
    return extracted_dir


def my_zip(ms: Path, dst: Path, logger) -> Path:
    logger.info(f"Starting zipping process for: {ms}")
    zip_filepath = dst

    if zip_filepath.exists() and zip_filepath.is_dir():
        logger.error(
            f"Cannot create a zip file as a directory with the name {zip_filepath} exists."
        )
        raise IsADirectoryError(
            f"Cannot create a zip file as a directory with the name {zip_filepath} exists."
        )

    if ms.is_dir():
        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_STORED) as zipf:
            for root, dirs, files in os.walk(ms):
                for file in files:
                    file_path = Path(root) / file
                    arcname = ms.name / file_path.relative_to(ms)
                    zipf.write(file_path, arcname)
    else:
        logger.error(f"{ms} is not a directory.")
        raise NotADirectoryError(f"Expected a directory, got {ms}")

    logger.info(f"Created zip file at {zip_filepath}")
    return zip_filepath


def dict_to_parser(
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
    parser_content = "\n".join(lines)

    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with output_path.open("w") as file:
        file.write(parser_content)

    return output_path
