import io

import pandas as pd
from lithops import Storage


def preprocess_static_csv(prefix, flex_input, num_workers) -> None:
    storage = Storage()
    filename = "titanic.csv"
    key = f"{prefix}{filename}"
    df = pd.read_csv(io.BytesIO(storage.get_object(flex_input.bucket, key)))
    chunk_size = len(df) // num_workers
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    for worker_id, chunk in enumerate(chunks):
        storage.put_object(
            flex_input.bucket,
            f"{flex_input.prefix}{filename}.part{worker_id}",
            chunk.to_csv(index=False).encode("utf-8"),
        )


def preprocess_static_txt(prefix, flex_input, num_workers) -> None:
    storage = Storage()
    filename = "tiny-shakespeare.txt"
    key = f"{prefix}{filename}"
    file_size = int(storage.head_object(flex_input.bucket, key)["content-length"])
    file = storage.get_object(flex_input.bucket, key)
    text = file.decode("utf-8")
    start = 0
    for worker_id in range(num_workers):
        end = ((worker_id + 1) * file_size) // num_workers
        end = min(text.rfind(" ", start, end), end)
        storage.put_object(
            flex_input.bucket,
            f"{flex_input.prefix}{filename}.part{worker_id}",
            text[start:end].encode("utf-8"),
        )
        start = end + 1
