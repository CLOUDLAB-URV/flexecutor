import unittest
from unittest.mock import patch, Mock
from flexecutor.storage import FlexData, StrategyEnum


class TestFlexInput(unittest.TestCase):

    def setUp(self):
        print("Setting up Test Environment for FlexInput")
        self.prefix = "test_prefix/"
        self.bucket = "test_bucket"
        self.custom_data_id = "input123"
        self.read_strategy = StrategyEnum.SCATTER
        self.local_base_path = "/tmp"

        self.mock_storage_handler = Mock()
        self.mock_storage_handler.list_objects.return_value = [
            {"Key": "file1.txt"},
            {"Key": "file2.txt"},
        ]

        self.patcher = patch("importlib.import_module")
        self.mock_import_module = self.patcher.start()
        self.mock_module = Mock()
        self.mock_module.StorageBackend.return_value = self.mock_storage_handler
        self.mock_import_module.return_value = self.mock_module

        self.flex_data = FlexData(
            prefix=self.prefix,
            bucket=self.bucket,
            custom_data_id=self.custom_data_id,
            read_strategy=self.read_strategy,
            local_base_path=self.local_base_path,
        )
        print("Test Environment Set Up Complete")

    def test_scan_objects(self):
        print(
            "Testing scan_objects: Expecting keys and paths to be set based on storage list_objects output."
        )
        self.flex_data.scan_keys()
        self.flex_data.set_local_paths()
        self.flex_data.set_file_indexes(worker_id=0, num_workers=1)

        expected_keys = ["file1.txt", "file2.txt"]
        expected_local_paths = [
            "/tmp/test_prefix/file1.txt",
            "/tmp/test_prefix/file2.txt",
        ]
        self.assertEqual(self.flex_data.keys, expected_keys)
        self.assertEqual(self.flex_data.local_paths, expected_local_paths)
        self.assertEqual(self.flex_data.file_indexes, (0, 2))
        print(f"Expected keys: {expected_keys}, Actual keys: {self.flex_data.keys}")
        print(
            f"Expected paths: {expected_local_paths}, Actual paths: {self.flex_data.local_paths}"
        )

    def test_broadcast_strategy(self):
        print("Testing scan_objects with BROADCAST strategy")
        self.flex_data.read_strategy = StrategyEnum.BROADCAST
        self.flex_data.scan_keys()
        self.flex_data.set_local_paths()
        self.flex_data.set_file_indexes(worker_id=0, num_workers=2)
        expected_file_index = (0, 2)
        self.assertEqual(self.flex_data.file_indexes, expected_file_index)
        print("scan_objects with BROADCAST strategy passed")

    def test_scatter_strategy(self):
        print("Testing scan_objects with SCATTER strategy")
        num_workers = 2
        results = []
        for worker_id in range(num_workers):
            self.flex_data.scan_keys()
            self.flex_data.set_local_paths()
            self.flex_data.set_file_indexes(worker_id, num_workers)
            start, end = self.flex_data.file_indexes
            handled_files = self.flex_data.local_paths[start:end]
            results.append((start, end, handled_files))
            print(
                f"Worker {worker_id} handles files from index {start} to {end - 1}: {handled_files}"
            )
        expected_results = [
            (0, 1, ["/tmp/test_prefix/file1.txt"]),
            (1, 2, ["/tmp/test_prefix/file2.txt"]),
        ]
        self.assertEqual(results, expected_results)
        print("scan_objects with SCATTER strategy passed")

    def test_scatter_strategy_with_chunker(self):
        pass

    def test_scatter_more_files_than_workers(self):
        # TODO: Automatic partitioning? or a worker handles more files than the others?
        pass

    def test_scatter_more_workers_than_files(self):
        # TODO: This must  partition the file into chunks
        pass

    def tearDown(self):
        print("Tearing down Test Environment")
        self.patcher.stop()


if __name__ == "__main__":
    unittest.main()
