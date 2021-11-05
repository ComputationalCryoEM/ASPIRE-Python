![Logo](http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg)

[![Azure Build Status](https://dev.azure.com/ComputationalCryoEM/Aspire-Python/_apis/build/status/ComputationalCryoEM.ASPIRE-Python?branchName=master)](https://dev.azure.com/ComputationalCryoEM/Aspire-Python/_build/latest?definitionId=3&branchName=master)
[![Github Actions Status](https://github.com/ComputationalCryoEM/ASPIRE-Python/actions/workflows/workflow.yml/badge.svg)](https://github.com/ComputationalCryoEM/ASPIRE-Python/actions/workflows/workflow.yml)
[![codecov](https://codecov.io/gh/ComputationalCryoEM/ASPIRE-Python/branch/master/graph/badge.svg?token=3XFC4VONX0)](https://codecov.io/gh/ComputationalCryoEM/ASPIRE-Python)

# ASPIRE - Algorithms for Single Particle Reconstruction - v0.8.1

This is the Python version to supersede the [Matlab ASPIRE](https://github.com/PrincetonUniversity/aspire).

ASPIRE is an open-source software package for processing single-particle cryo-EM data to determine three-dimensional structures of biological macromolecules. The package includes advanced algorithms based on rigorous mathematics and recent developments in
statistics and machine learning. It provides unique and improved solutions to important computational challenges of the cryo-EM
processing pipeline, including 3-D *ab-initio* modeling, 2-D class averaging, automatic particle picking, and 3-D heterogeneity analysis.

For more information about the project, algorithms, and related publications please refer to the [ASPIRE Project website](http://spr.math.princeton.edu/).

**For full documentation see [the docs](https://computationalcryoem.github.io/ASPIRE-Python).**

## Installation Instructions

For end-users
-------------

ASPIRE is a pip-installable package that works on Linux/Mac/Windows, and requires Python 3.6. The simplest option is to use Anaconda 64-bit for your platform with a minimum of Python 3.6 and `pip`, and then use `pip` to install `aspire` in that environment.

```
conda create -n aspire_env python=3.6 pip
conda activate aspire_env
pip install aspire
```

The final step above should install any dependent packages from `pip` automatically. To see what packages are required, browse `setup.py`.

Note that this step installs the base `aspire` package for you to work with, but not the unit tests/documentation. If you need to install ASPIRE for development purposes, read on.

For developers
--------------

After cloning this repo, the simplest option is to use Anaconda 64-bit for your platform, and use the provided `environment.yml` file to build a Conda environment to run ASPIRE. This is very similar to above except you will be based off of your local checkout, and you are free to rename `aspire_dev` used in the commands below. The `pip` line will install aspire in a locally editable mode, and is equivalent to `python setup.py develop`:

```
cd /path/to/git/clone/folder

# Creates the conda environment and installs base dependencies.
conda env create -f environment.yml --name aspire_dev

# Enable the environment
conda activate aspire_dev

# Install the aspire package in a locally editable way,
# and additionally installs the developer tools extras:
pip install -e ".[dev]"

```

If you prefer not to use Anaconda, or want to manage environments yourself, you should be able to use `pip` with Python >= 3.6.
Please see the full documentation for details.

You may optionally install additional packages for GPU extensions:

```
# Additional GPU packages (requires CUDA)
pip install -e ".[gpu]"
```

### Make sure everything works

Once ASPIRE is installed, make sure the unit tests run correctly on your platform by doing:

```
cd /path/to/git/clone/folder
pytest
```
