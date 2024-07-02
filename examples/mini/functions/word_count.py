from flexecutor.storage.chunker import FileChunker
from flexecutor.storage.storage import FlexInput, FlexOutput
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


word_count_input = FlexInput(
    prefix="lsls", custom_input_id="txt", bucket="test-bucket", chunker=FileChunker()
)
word_count_output = FlexOutput(prefix="count", bucket="test-bucket", suffix=".count")
