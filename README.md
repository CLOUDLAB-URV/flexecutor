# Flexecutor

Flexecutor is a Python framework for building and optimizing serverless workflows. It extends Lithops by providing a high-level workflow abstraction (DAG-based) and integrating multiple state-of-the-art smart provisioning strategies. This allows users to deploy parallel workloads on serverless platforms without manually tuning resource configurations.

## Installation

```bash
git clone https://github.com/CLOUDLAB-URV/flexecutor.git
cd flexecutor
pip install .
```

## Key Features

- **Cloud-agnostic**: Runs on any Lithops-supported backend (AWS Lambda, Azure Functions, GCP Functions, IBM Code Engine, Kubernetes, etc.).
- **DAG Workflow Model**: Define multi-stage workflows declaratively.
- **Automatic Data Handling**: Input and output data are managed through object storage, with customizable distribution strategies.
- **Smart Provisioning**: Choose resource configurations based on time, cost, or a trade-off.
- **Extensible Design**: Integrate new provisioning strategies without modifying workflow logic.

## Key entities

| Concept | Description |
| --- | --- |
| `FlexData` | Defines input/output data and how it is distributed to workers. |
| `Stage` | Unit of computation that processes inputs and produces outputs. |
| `DAG` | Workflow composed of dependent stages. |
| `DAGExecutor` | Executes, profiles, trains, optimizes and runs workflows. |
| `Scheduler` | Algorithm that selects resource configurations (workers, CPU, memory). |

---

# How to Code Your Flexecutor App

A Flexecutor application consists of three main components:

1. **User functions** - Python callables that perform your computation
2. **Data definitions** - `FlexData` objects specifying input/output locations
3. **Workflow graph** - `DAG` and `Stage` objects defining execution order

## Step-by-Step Guide

### 1. Define your user functions

Create Python functions that accept a `StageContext` parameter.

This context provides access to input files, output paths, and parameters (`get_input_paths()`, `next_output_path()`, `get_param()`).

### 2. Set up the `@flexorchestrator` decorator

Wrap your main function with the `@flexorchestrator` decorator.

This initializes the execution environment and configures the bucket and execution path.

### 3. Create `FlexData` objects

Define your input and output data using `FlexData`.

- `StrategyEnum.SCATTER`: split files across workers
- `StrategyEnum.BROADCAST`: distribute all files to every worker

### 4. Create stages

Build `Stage` objects that bind user functions to input/output `FlexData`.

Each stage needs:

- A `stage_id`
- A function
- Lists of input and output `FlexData`
- Optional parameters

### 5. Build the DAG

Create a `DAG` container and add your stages.

Use the `>>` operator to define dependencies.

### 6. Define your scheduler

Choose a smart provisioning strategy depending on your optimization goal.

You will pass the scheduler instance to the `DAGExecutor`.

See `Caerus`, `Ditto`, `Orion` and `Jolteon` classes and review the input data of each scheduler.

### 7. Execute the smart provisioning loop

Flexecutor provides a provisioning loop that evaluates execution configurations, learns performance models, and selects the optimal configuration automatically. The loop consists of:

1. **Profile**: Run the workflow under different configurations and collect metrics.
2. **Train**: Learn a performance model from the profiling data.
3. **Optimize**: Compute the best configuration according to the scheduler objective.
4. **Execute**: Run the workflow using the optimal configuration.

Example:

```python
config_space = [
    {"process": {"workers": 4, "cpu": 1, "memory": 1024}},
    {"process": {"workers": 16, "cpu": 1, "memory": 1024}},
]

executor.profile(config_space)
executor.train()
executor.optimize()
executor.execute()
executor.shutdown()
```

### 8. Clean up

Call `shutdown()` on the executor before end.

---

## Reference this work

Flexecutor is part of the accepted paper: *“Flexecutor: Out-of-the-Box Smart Provisioning for Serverless Workflows”*

```
@inproceedings{molina2025flexecutor,
  title = {Flexecutor: Out-of-the-Box Smart Provisioning for Serverless Workflows},
  author = {Molina-Gim{\'e}nez, Enrique and Barcelona-Pons, Daniel and Iacoponelli, Octavio H. and Garc{\'i}a-L{\'o}pez, Pedro},
  booktitle = {Proceedings of the 11th Workshop of Serverless Computing (WoSC'25)},
  year = {2025},
  location = {Nashville, TN, USA},
  publisher = {ACM},
  address = {New York, NY, USA},
  pages = {1--6},
  doi = {10.1145/3774899.3775013},
  url = {https://github.com/CLOUDLAB-URV/flexecutor}
}
```
