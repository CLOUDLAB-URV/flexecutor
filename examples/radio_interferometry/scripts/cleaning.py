from lithops import Storage


def clean_radio_data():
    storage = Storage()
    bucket_name = "flexecutor-demo"
    prefixes = ["rebinning_out/", "applycal_out/", "image_out/"]

    for prefix in prefixes:
        object_keys = storage.list_keys(bucket_name, prefix=prefix)
        if object_keys:
            storage.delete_objects(bucket_name, object_keys)

    print("Clean up (radio interferometry) completed.")


if __name__ == "__main__":
    clean_radio_data()
