![Logo](http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg)

[![Github Actions Status](https://github.com/ComputationalCryoEM/ASPIRE-Python/actions/workflows/workflow.yml/badge.svg)](https://github.com/ComputationalCryoEM/ASPIRE-Python/actions/workflows/workflow.yml)
[![codecov](https://codecov.io/gh/ComputationalCryoEM/ASPIRE-Python/branch/main/graph/badge.svg?token=3XFC4VONX0)](https://codecov.io/gh/ComputationalCryoEM/ASPIRE-Python)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5657281.svg)](https://doi.org/10.5281/zenodo.5657281)
[![Downloads](https://static.pepy.tech/badge/aspire/month)](https://pepy.tech/project/aspire)

# ASPIRE - Algorithms for Single Particle Reconstruction - v0.13.0

The ASPIRE-Python project supersedes [Matlab ASPIRE](https://github.com/PrincetonUniversity/aspire).

ASPIRE is an open-source software package for processing single-particle cryo-EM data to determine three-dimensional structures of biological macromolecules. The package includes advanced algorithms based on rigorous mathematics and recent developments in
statistics and machine learning. It provides unique and improved solutions to important computational challenges of the cryo-EM
processing pipeline, including 3-D *ab-initio* modeling, 2-D class averaging, automatic particle picking, and 3-D heterogeneity analysis.

For more information about the project, algorithms, and related publications please refer to the [ASPIRE Project website](http://spr.math.princeton.edu/).

**For full documentation and tutorials see [the docs](https://computationalcryoem.github.io/ASPIRE-Python).**

Please cite using the following DOI. This DOI represents all versions, and will always resolve to the latest one.

```
ComputationalCryoEM/ASPIRE-Python: v0.13.0 https://doi.org/10.5281/zenodo.5657281

```

## Installation Instructions

Getting Started - Installation
------------------------------

ASPIRE is a pip-installable package for Linux/Mac/Windows, and
requires Python 3.8-3.11. The recommended method of installation for
getting started is to use Anaconda (64-bit) for your platform to
install Python. Python's package manager `pip` can then be used to
install `aspire` safely in that environment.

If you are unfamiliar with `conda`, the
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)
distribution for `x86_64` is recommended.

Assuming you have `conda` and a compatible system, the following steps
will checkout current code release, create an environment, and install
ASPIRE.

```
# Clone the code
git clone https://github.com/ComputationalCryoEM/ASPIRE-Python.git
cd ASPIRE-Python

# Create a fresh environment
conda create --name aspire python=3.8 pip

# Enable the environment
conda activate aspire

# Install the `aspire` package from the checked out code
# with the additional `dev` extension.
pip install -e ".[dev]"
```

If you prefer not to use Anaconda, or have other preferences for managing environments, you should be able to directly use `pip` with Python >= 3.8 from the local checkout or via PyPI.
Please see the full documentation for details and advanced instructions.

### Installation Testing

To check the installation, a unit test suite is provided,
taking approximate 15 minutes on an average machine.

```
pytest
```
