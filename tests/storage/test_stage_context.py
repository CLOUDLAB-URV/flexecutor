import os
import unittest

from flexecutor.workflow.stagecontext import StageContext, InternalStageContext
from flexecutor.storage.storage import StrategyEnum, FlexData


class TestStageContext(unittest.TestCase):
    def setUp(self):
        print("Setting up Test Environment for StageContext")

        self.flex_data_input = FlexData(
            prefix="test_prefix/",
            bucket="test_bucket",
            custom_data_id="input123",
            read_strategy=StrategyEnum.SCATTER,
            local_base_path="/tmp",
        )
        self.flex_data_output = FlexData(
            prefix="test_output/",
            bucket="test_bucket",
            custom_data_id="output123",
            suffix=".file",
            local_base_path="/tmp",
        )
        self.params = {"param1": "value1", "param2": "value2"}

        self.internal_context = InternalStageContext(
            worker_id=0,
            num_workers=1,
            inputs=[self.flex_data_input],
            outputs=[self.flex_data_output],
            params=self.params,
        )
        self.stage_context = StageContext(self.internal_context)
        print("Test Environment Set Up Complete")

    def test_initialization(self):
        print("Testing Initialization of InternalStageContext and StageContext")
        self.assertEqual(self.internal_context.worker_id, 0)
        self.assertEqual(self.internal_context.num_workers, 1)
        self.assertEqual(self.internal_context.inputs["input123"], self.q)
        self.assertEqual(self.internal_context.outputs["output123"], self.flex_data_output)
        self.assertEqual(self.internal_context._params, self.params)
        print("Initialization test passed")

    def test_input_paths(self):
        print("Testing input_paths method")
        self.flex_data_input.local_paths = [
            "/tmp/test_prefix/file1.txt",
            "/tmp/test_prefix/file2.txt",
        ]
        self.flex_data_input.file_index = (0, 2)
        paths = self.stage_context.get_input_paths("input123")
        expected_paths = ["/tmp/test_prefix/file1.txt", "/tmp/test_prefix/file2.txt"]
        self.assertEqual(paths, expected_paths)
        print(f"Expected paths: {expected_paths}, Actual paths: {paths}")
        print("input_paths method test passed")

    def test_get_param(self):
        print("Testing get_param method")
        param_value = self.stage_context.get_param("param1")
        self.assertEqual(param_value, "value1")
        print(f"Expected param value: 'value1', Actual param value: {param_value}")
        print("get_param method test passed")

    def test_next_output_path(self):
        print("Testing next_output_path method")
        output_path = self.stage_context.next_output_path("output123")
        self.assertTrue(output_path.startswith("/tmp/test_output/"))
        self.assertTrue(output_path.endswith(".file"))
        self.assertIn(output_path, self.flex_data_output.local_paths)
        self.assertIn(
            f"{self.flex_data_output.prefix}/{os.path.basename(output_path)}",
            self.flex_data_output.keys,
        )
        print(f"Generated output path: {output_path}")
        print("next_output_path method test passed")

    def tearDown(self):
        print("Tearing down Test Environment")


if __name__ == "__main__":
    unittest.main()
