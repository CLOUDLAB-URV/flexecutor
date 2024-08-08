import os

import pandas as pd

from flexecutor.storage.chunker import ChunkerContext


def chunking_static_csv(ctx: ChunkerContext) -> None:
    # TODO: Manage the case when there are multiple files
    file = ctx.get_input_paths()[0]
    df = pd.read_csv(file)
    chunk_sizes = [len(df) // ctx.get_num_workers()] * ctx.get_num_workers()
    remaining = len(df) % ctx.get_num_workers()
    for i in range(remaining):
        chunk_sizes[i % ctx.get_num_workers()] += 1
    chunks = [df.iloc[sum(chunk_sizes[:i]):sum(chunk_sizes[:i + 1])] for i in range(ctx.get_num_workers())]
    for worker_id, chunk in enumerate(chunks):
        chunk.to_csv(ctx.next_chunk_path(), index=False)


def chunking_static_txt(ctx: ChunkerContext) -> None:
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
