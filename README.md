NnpCG
==============================
[//]: # (Badges)
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/)
[![GitHub Actions Build Status](https://github.com/ProLint/prolint2/workflows/CI/badge.svg)](https://github.com/tieleman-lab/nnpcg/actions?query=workflow%3ACI)


Python interface to set up coarse-grained simulations with state-of-the-art neural network potentials.

Installation
============
To install **nnpcg** we recommend creating a new conda environment as follows:

``` bash
   git clone https://github.com/tieleman-lab/nnpcg
   cd nnpcg
   conda env create -f nnpcg_train_run.yml
   conda env create -f nnpcg_analysis.yml
   conda activate nnpcg_dev
   git clone https://github.com/torchmd/torchmd-net.git
   cd torchmd-net
   pip install -e .
```

Then you can install **nnpcg** via pip in both environments:

``` bash
   cd ..
   pip install -e .
   conda activate nnpcg_ana
   pip install -e .
```
How to contribute?
==================
If you find a bug in the source code, you can help us by submitting an issue to our [GitHub repo](https://github.com/tieleman-lab/nnpcg). Even better, you can submit a Pull Request with a fix. 

We really appreciate your feedback!

License 
=======

Source code included in this project is available under the [MIT License](https://opensource.org/licenses/MIT).

### Copyright

Copyright (c) 2023, Daniel P. Ramirez


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
