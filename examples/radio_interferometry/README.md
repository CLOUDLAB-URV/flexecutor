# Radio interferometry pipeline

## Folder structure

- `compss/`:  Contains the scripts to run the pipeline with COMPSs.
- `notebooks/`: Contains the Jupyter notebooks to run the pipeline interactively.
- `scripts/`: Contains the scripts to run the pipeline with Flexecutor DAG engine.

## Pipeline types

- `classic`: Sequential pipeline (rebibbing + calibration + imaging).
- `julien`: Parallel pipeline (TARGET + CAL branches).

> **Note**: Zip files pushed to S3 mustn't contain the root folder. The files must be at the root level of the zip file.