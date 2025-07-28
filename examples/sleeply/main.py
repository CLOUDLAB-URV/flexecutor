from lithops import FunctionExecutor
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.storage.storage import FlexData

from functions import prepare_sleep, benchmark_stage

if __name__ == "__main__":

    @flexorchestrator(bucket="your-bucket-name")        
    def main():
        dag = DAG("benchmark-sleep-dag")

        sleep_meta = FlexData(prefix="sleeply/", custom_data_id="sleep_meta")

        datablock_1 = FlexData(prefix="datablock/stage1/", custom_data_id="stage_1_file")
        datablock_2 = FlexData(prefix="datablock/stage2/", custom_data_id="stage_2_file")
        datablock_3 = FlexData(prefix="datablock/stage3/", custom_data_id="stage_3_file")

        log_output = FlexData(prefix="logs/", custom_data_id="log", suffix=".log")

        stage_datablock = Stage(
            stage_id="generate-datablocks",
            func=prepare_sleep,
            inputs=[sleep_meta],
            outputs=[datablock_1, datablock_2, datablock_3, log_output])

        stage_bmark1 = Stage(
            stage_id="benchmark-1",
            func=benchmark_stage,
            inputs=[sleep_meta, datablock_1],
            outputs=[log_output]
        )

        stage_bmark2 = Stage(
            stage_id="benchmark-2",
            func=benchmark_stage,
            inputs=[sleep_meta, datablock_2],
            outputs=[log_output]
        )

        stage_bmark3 = Stage(
            stage_id="benchmark-3",
            func=benchmark_stage,
            inputs=[sleep_meta, datablock_3],
            outputs=[log_output]
        )

        stage_datablock >> stage_bmark1 >> stage_bmark2 >> stage_bmark3        #sequential execution

        dag.add_stages([stage_datablock, stage_bmark1, stage_bmark2, stage_bmark3])

        executor = DAGExecutor(
            dag,
            executor=FunctionExecutor(log_level="INFO"),
        )
        executor.execute(num_workers=1)
        executor.shutdown()
    
    main()
