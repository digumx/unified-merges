This repository contains the code, benchmarks and experimental data for the
paper "Unifying Syntactic and Semantic Abstractions for Deep Neural Networks".

# Installation Instructions:

Clone the repository and update all submodules. Then, create a new conda
environment via:

```
conda env remove --name unified-merge
conda env create -f env.yaml
```
Then follow instructions from the NeuralSat repository to install NeuralSat

Create two directories `outs` and `stats` for storing logs and statistics files.

# Running:

## Running All Experiments:

Use the script `batch_run_verif.py`. Pass in `-m ours` to run experiments using
our method, and `-m baseline` for baseline experiments. There are several other
options available, see `--help`.

Once the run is complete, the `write_res_to_excel.py` script can be used to
generate an excel spreadsheet summarizing the data from the run.

## Running a Single Experiment:

The code can handle networks in either `.nnet` or `.onnx` formats.

The code expects properties to be spefied using _.prop files_. The
format of these files are the string representation of the dicts specified in
`encode_property` in `property.py`.

# Results:

The results presented in the paper are found in `results/acasxu.xlsx`.
