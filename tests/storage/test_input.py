import unittest
from unittest.mock import MagicMock, patch
from flexecutor.storage import FlexInput, StrategyEnum


class TestFlexInput(unittest.TestCase):

    def setUp(self):
        # Create a mock storage object to mock lithops storage, this way we don't need objects in OS.
        self.mock_storage = MagicMock()
        self.mock_storage.list_objects.return_value = [
            {"Key": "file1.txt"},
            {"Key": "file2.txt"},
        ]
        self.flex_input = FlexInput(
            prefix="test_prefix/",
            bucket="test_bucket",
            storage=self.mock_storage,
        )

    def test_scan_objects(self):
        self.flex_input.scan_objects(worker_id=0, num_workers=1)

        expected_keys = ["file1.txt", "file2.txt"]
        expected_local_paths = [
            "/tmp/test_prefix/file1.txt",
            "/tmp/test_prefix/file2.txt",
        ]
        self.assertEqual(self.flex_input.keys, expected_keys)
        self.assertEqual(self.flex_input.local_paths, expected_local_paths)
        self.assertEqual(self.flex_input.chunk_indexes, (0, 2))

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
