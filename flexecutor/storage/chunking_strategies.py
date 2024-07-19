import os

import pandas as pd

from flexecutor.storage.chunker import ChunkerContext


def preprocess_static_csv(ctx: ChunkerContext) -> None:
    # TODO: Manage the case when there are multiple files
    file = ctx.get_input_paths()[0]
    df = pd.read_csv(file)
    chunk_size = len(df) // ctx.get_num_workers()
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    for worker_id, chunk in enumerate(chunks):
        chunk.to_csv(ctx.next_chunk_path(), index=False)


def preprocess_static_txt(ctx: ChunkerContext) -> None:
    # TODO: Manage the case when there are multiple files
    file_path = ctx.get_input_paths()[0]
    file = open(file_path, "r")
    file_size = os.path.getsize(file_path)
    text = file.read()
    start = 0
    for ctx.worker_id in range(ctx.get_num_workers()):
        end = ((ctx.worker_id + 1) * file_size) // ctx.get_num_workers()
        end = min(text.rfind(" ", start, end), end)
        with open(ctx.next_chunk_path(), "w") as f:
            f.write(text[start:end])
        start = end + 1
