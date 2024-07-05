from flexecutor.storage.chunker import FileChunker
from flexecutor.storage.storage import FlexInput, FlexOutput, StrategyEnum
from flexecutor.utils.storagecontext import StorageContext


def word_count(st_context: StorageContext):
    txt_paths = st_context.get_input_paths("txt")
    for txt_path in txt_paths:
        with open(txt_path, "r") as f:
            content = f.read()

        count = len(content.split())

        count_path = st_context.next_output_path("count")
        with open(count_path, "w") as f:
            f.write(str(count))


def sum_counts(st_context: StorageContext):
    count_paths = st_context.get_input_paths("count")
    total = 0
    for count_path in count_paths:
        with open(count_path, "r") as f:
            count = int(f.read())
        total += count

    total_path = st_context.next_output_path("total")
    with open(total_path, "w") as f:
        f.write(str(total))


word_count_input = FlexInput(
    prefix="dir", custom_input_id="txt", bucket="test-bucket", chunker=FileChunker()
)
word_count_output = FlexOutput(prefix="count", bucket="test-bucket", suffix=".count")

reduce_input = FlexInput(
    prefix="count",
    custom_input_id="count",
    bucket="test-bucket",
    strategy=StrategyEnum.BROADCAST,
)

reduce_output = FlexOutput(prefix="total", bucket="test-bucket", suffix=".total")
