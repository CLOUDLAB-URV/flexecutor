import hashlib
import logging
import concurrent.futures
from pathlib import PosixPath, Path

import numpy as np
from casacore.tables import table
from lithops import FunctionExecutor

from examples.radio_interferometry.utils import get_dir_size, my_zip, unzip
from examples.radio_interferometry.utils import FlexInput, FlexOutput
from flexecutor import StageContext
from flexecutor.utils.utils import flexorchestrator, setup_logging
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage

MB = 1024 * 1024


def generate_concatenated_identifier(ms_tables, num_partitions):
    hash_md5 = hashlib.md5()

    total_rows = 0
    total_cols = 0
    ms_names = []

    for ms in ms_tables:
        total_rows += ms.nrows()
        total_cols = max(total_cols, ms.ncols())
        ms_names.append(ms.name())

    metadata = f"{total_rows}_{total_cols}_{num_partitions}_{'_'.join(ms_names)}"
    hash_md5.update(metadata.encode("utf-8"))
    identifier = hash_md5.hexdigest()
    identifier = identifier.strip("/")
    return identifier


def create_partition(i, start_row, end_row, ms, msout, logger):
    logger.debug(
        f"Creating partition {i} with rows from {start_row} to {end_row - 1}..."
    )
    partition = ms.selectrows(list(range(start_row, end_row)))
    partition_path = PosixPath(msout.removesuffix('.zip'))
    partition.copy(str(partition_path), deep=True)
    partition.close()

    partition_size = get_dir_size(partition_path)
    logger.debug(f"Partition {i} created. Size before zip: {partition_size} bytes")

    my_zip(partition_path, Path(msout))
    return partition_size


def partition_ms(ctx: StageContext):
    logger = setup_logging(logging.DEBUG)

    num_partitions = 5
    full_file_paths = [Path(file) for file in ctx.get_input_paths("huge-ms")]

    logger.debug(f"Downloaded files to: {full_file_paths}")

    mss = []
    for f_path in full_file_paths:
        logger.debug(f"Processing file: {f_path}")
        unzipped_ms = unzip(f_path)
        logger.debug(f"Unzipped contents at: {unzipped_ms}")

        ms_table = table(str(unzipped_ms), ack=False)
        mss.append(ms_table)
        logger.info(f"Number of rows in the measurement set: {ms_table.nrows()}")
    ms = table(mss)
    identifier = generate_concatenated_identifier(mss, num_partitions)
    logger.info(f"Unique identifier for concatenated measurement sets: {identifier}")

    logger.info(f"#rows: {ms.nrows()}; #cols: {ms.ncols()}")

    ms_sorted = ms.sort("TIME")
    total_rows = ms_sorted.nrows()
    logger.info(f"Total rows in the measurement set: {total_rows}")
    times = np.array(ms_sorted.getcol("TIME"))
    total_duration = times[-1] - times[0]
    logger.debug(f"Total duration in the measurement set: {total_duration}")

    chunk_duration = total_duration / num_partitions
    partitions_info = []
    start_time = times[0]
    partition_count = 0

    end_time = 0
    for i in range(num_partitions):
        if i < num_partitions - 1:
            end_time = start_time + chunk_duration
            end_index = np.searchsorted(times, end_time, side="left")
        else:
            end_index = total_rows
        start_index = np.searchsorted(times, start_time, side="left")
        partitions_info.append((partition_count, start_index, end_index))
        start_time = end_time
        partition_count += 1

    partition_sizes = []
    msout_list = [ctx.next_output_path("chunks") for _ in range(num_partitions)]
    print(msout_list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                create_partition, info[0], info[1], info[2], ms_sorted, msout, logger
            )
            for info, msout in zip(partitions_info, msout_list)
        ]
        for future in concurrent.futures.as_completed(futures):
            partition_size = future.result()
            partition_sizes.append(partition_size)

    for partition_file, partition_size in zip(msout_list, partition_sizes):
        logger.debug(
            f"Partition file {partition_file} with size {partition_size / MB:.2f} MB created and ready for upload."
        )

    total_partition_size = sum(partition_sizes)
    logger.debug(f"Total size of all partitions: {total_partition_size / MB:.2f} MB")

    ms_sorted.close()
    logger.info("Partitioning completed.")


if __name__ == "__main__":

    @flexorchestrator(bucket="test-bucket")
    def main():
        dag = DAG("chunking")

        chunking_stage = Stage(
            stage_id="chunking",
            func=partition_ms,
            inputs=[FlexInput(prefix="huge-ms")],
            outputs=[FlexOutput(prefix="chunks", suffix=".ms.zip")],
        )

        dag.add_stage(chunking_stage)
        executor = DAGExecutor(
            dag,
            executor=FunctionExecutor(
                log_level="INFO", **{"runtime_memory": 2048, "runtime_cpu": 4}
            ),
        )
        results = executor.execute()
        print(results["chunking"].get_timings())

    main()
