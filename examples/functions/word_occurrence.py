import collections

from lithops import Storage

from flexecutor.stage import operation
from flexecutor.utils.utils import initialize_timings


def word_occurrence_count(obj):
    timings = initialize_timings()
    storage = Storage()

    with operation("read", timings):
        data = obj.data_stream.read().decode("utf-8")

    with operation("compute", timings):
        words = data.split()
        word_count = collections.Counter(words)

    with operation("write", timings):
        result_key = f"results_{obj.data_byte_range[0]}-{obj.data_byte_range[1]}.txt"
        result_data = (
            f"Word Count: {len(word_count)}\nWord Frequencies: {dict(word_count)}\n"
        )

        storage.put_object(obj.bucket, result_key, result_data.encode("utf-8"))

    return timings
