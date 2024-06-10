# Flexecutor examples repository
This folder contains examples of how to use `flexecutor`.

## Examples structure

We can find different folders at the top level of this folder:
- In general, each folder contains a different use case with its related staff. One use case corresponds to one DAG, formed by a set of stages. Several examples of use cases are:
  - Video processing
  - ...
- The `mini` folder contains little examples whose purpose is to validate different features of `flexecutor`. The scripts in this folder are not real applications; mainly, they are used for testing software components during development.

## Generated files
When running the examples, some files are generated in its use-case folder:
- `profiling/`: This folder contains the times of execution of the stages in the app.
- `images/`: This folder contains generated images.
- `models/`: This folder contains the models generated after the training of the app.

## How to write an example
> ⚠️ IMPORTANT: decorate your main script with the `@flexorchestrator` decorator for correct behavior.