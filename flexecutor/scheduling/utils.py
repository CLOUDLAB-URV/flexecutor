from lithops import Storage


def get_size_by_s3_prefix(bucket, prefix):
    storage = Storage()
    s3_objects = storage.list_objects(bucket, prefix)
    return sum([item["Size"] for item in s3_objects])
