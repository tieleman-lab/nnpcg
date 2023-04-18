NnpCG
==============================
[//]: # (Badges)
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/)
[![GitHub Actions Build Status](https://github.com/ProLint/prolint2/workflows/CI/badge.svg)](https://github.com/tieleman-lab/nnpcg/actions?query=workflow%3ACI)


Python interface to set up coarse-grained simulations with state-of-the-art neural network potentials.

Installation
============
To install **nnpcg** we recommend creating two different conda environments as follows:

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

Curated Lipids DB
=================
| NMRLipids Score | NMRLpids Benchmark | Trajectory |
| :-------------: | :----------------: | :--------: |
| 0.83 | [link](http://databank.nmrlipids.fi/trayectorias/1) | [link](https://zenodo.org/record/6582985#.ZDnYauzMI-Q) |
| 0.76 | [link](http://databank.nmrlipids.fi/trayectorias/617) | [link](https://zenodo.org/record/166034#.ZDnZ6ezMI-Q) |
| 0.73 | [link](http://databank.nmrlipids.fi/trayectorias/696) | [link](https://zenodo.org/record/7022749#.ZDnaO-zMI-Q) |
| 0.69 | [link](http://databank.nmrlipids.fi/trayectorias/457) | [link](https://zenodo.org/record/3741793#.ZDnb2OzMI-Q) |
| 0.61 | [link](http://databank.nmrlipids.fi/trayectorias/63) | [link](https://zenodo.org/record/13498#.ZDndQuzMI-Q) |

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
