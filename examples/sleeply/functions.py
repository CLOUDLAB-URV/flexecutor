import os
import pickle
import uuid
import time
from flexecutor import StageContext

def prepare_sleep(ctx: StageContext):
    execution_start = time.time()

    input_paths = ctx.get_input_paths("sleep_meta")
    total_files = 0
    total_bytes = 0

    for path in input_paths:
        with open(path, "rb") as f:
            metadata = pickle.load(f)

        stage_num = metadata["stage"]
        size = metadata["size"]

        dummy_data = os.urandom(size)
        total_bytes += size
        total_files += 1

        output_id = f"stage_{stage_num}_file"
        out_path = ctx.next_output_path(output_id)

        with open(out_path, "wb") as f:
            f.write(dummy_data)

    execution_end = time.time()

    log_content = (
        f"uuid: {str(uuid.uuid4())}\n"
        f"stage: 0\n"
        f"generated_files: {total_files}\n"
        f"total_size: {total_bytes} bytes\n"
        f"execution_time: {execution_end - execution_start:.3f}s\n"
    )

    log_path = ctx.next_output_path("log")
    with open(log_path, "w") as f:
        f.write(log_content)

def benchmark_stage(ctx: StageContext):

    execution_start = time.time()

    meta_paths = ctx.get_input_paths("sleep_meta")

    input_ids = [k for k in ctx._context.inputs.keys() if k != "sleep_meta"]
    if len(input_ids) != 1:
        raise FileNotFoundError(f"Expected 2 inputs (sleep_meta + 1 datablock), got: {ctx._context.inputs.keys()}")

    dummy_input_id = input_ids[0]
    dummy_paths = ctx.get_input_paths(dummy_input_id)

    for meta_path, dummy_path in zip(meta_paths, dummy_paths):
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        duration = metadata["duration"]
        stage = metadata["stage"]

        with open(dummy_path, "rb") as f:
            data = f.read()

        file_size = len(data)

        start = time.time()
        time.sleep(duration)
        end = time.time()

        log_content = (
            f"uuid: {str(uuid.uuid4())}\n"          #just in case, maybe not necessary
            f"stage: {stage}\n"
            f"duration: {duration}s\n"
            f"execution_time: {end - start:.3f}s\n"
            f"file_size: {file_size} bytes\n"
            f"total_time: {end-execution_start:.3f}\n"
        )

        log_path = ctx.next_output_path("log")
        with open(log_path, "w") as f:
            f.write(log_content)
