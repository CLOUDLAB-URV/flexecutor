from flexecutor.storage.chunker import CarelessFileChunker, WordCounterChunker
from flexecutor.storage.storage import FlexInput, FlexOutput
from flexecutor.utils.iomanager import IOManager


def word_count(io: IOManager):
    txt_paths = io.get_input_paths("txt")
    for txt_path in txt_paths:
        with open(txt_path, "r") as f:
            content = f.read()

        count = len(content.split())

        count_path = io.next_output_path("count")
        with open(count_path, "w") as f:
            f.write(str(count))


# word_count_input = FlexInput(
#     prefix="dir-chunks",
#     custom_input_id="txt",
#     bucket="test-bucket",
#     chunker=WordCounterChunker("dir"),
# )

word_count_input = FlexInput(
    prefix="dir",
    custom_input_id="txt",
    bucket="test-bucket",
    chunker=CarelessFileChunker(),
)

word_count_output = FlexOutput(prefix="count", bucket="test-bucket", suffix=".count")
