from flexecutor.storage.chunker import FileChunker
from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.iomanager import IOManager


def word_count(io: IOManager):
    txt_paths = io.input_paths("txt")
    for txt_path in txt_paths:
        with open(txt_path, "r") as f:
            content = f.read()

        count = len(content.split())

        count_path = io.next_output_path("count")
        with open(count_path, "w") as f:
            f.write(str(count))


word_count_input = FlexInput(
    "txt",
    bucket="test-bucket",
    prefix="dir",
    chunker=FileChunker(),
)
word_count_output = FlexOutput(bucket="test-bucket", prefix="count", suffix=".count")
