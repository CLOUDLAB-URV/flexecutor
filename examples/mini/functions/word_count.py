
from flexecutor.storage.chunker import Chunker, ChunkerTypeEnum
from flexecutor.storage.chunking_strategies import chunking_static_txt
from flexecutor.storage.storage import FlexData, StrategyEnum
from flexecutor import StageContext


def word_count(ctx: StageContext):
    txt_paths = ctx.get_input_paths("txt")
    for txt_path in txt_paths:
        with open(txt_path, "r") as f:
            content = f.read()

        count = len(content.split())

        count_path = ctx.next_output_path("count")
        with open(count_path, "w") as f:
            f.write(str(count))


def sum_counts(ctx: StageContext):
    count_paths = ctx.get_input_paths("count")
    total = 0
    for count_path in count_paths:
        with open(count_path, "r") as f:
            count = int(f.read())
        total += count

    total_path = ctx.next_output_path("total")
    with open(total_path, "w") as f:
        f.write(str(total))


flex_data_txt = FlexData(
    prefix="dir",
    custom_data_id="txt",
    bucket="test-bucket",
    chunker=Chunker(chunker_type=ChunkerTypeEnum.STATIC, chunking_strategy=chunking_static_txt),
)

# word_count_input = FlexInput(
#     prefix="dir",
#     custom_data_id="txt",
#     bucket="test-bucket",
#     chunker=CarelessFileChunker(),
# )

flex_data_word_count = FlexData(prefix="count", bucket="test-bucket", read_strategy=StrategyEnum.BROADCAST, suffix=".count")

flex_data_reduce_count = FlexData(prefix="total", bucket="test-bucket", suffix=".total")
