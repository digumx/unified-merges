This repository contains the code, benchmarks and experimental data for the
paper "Unifying Syntactic and Semantic Abstractions for Deep Neural Networks".

# Installation Instructions:

Clone the repository and update all submodules. Then, create a new conda
environment via:

```
conda env remove --name unified-merge
conda env create -f env.yaml
```



# Running:

## Running All Experiments:

Use the script `batch_run_verif.py`. 

## Running a Single Experiment:

The code can handle networks in either `.nnet` or `.onnx` formats.

The code expects properties to be spefied using _.prop files_. The
format of these files are the string representation of the dicts specified in
`encode_property` in `property.py`.

# Results:
