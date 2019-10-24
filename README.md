![Logo](http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg)

[![Azure Build Status](https://dev.azure.com/ComputationalCryoEM/Aspire-Python/_apis/build/status/ComputationalCryoEM.ASPIRE-Python?branchName=master)](https://dev.azure.com/ComputationalCryoEM/Aspire-Python/_build/latest?definitionId=3&branchName=master)
[![Travis Build Status](https://travis-ci.org/ComputationalCryoEM/ASPIRE-Python.svg?branch=master)](https://travis-ci.org/ComputationalCryoEM/ASPIRE-Python)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/ywgud2vu9ot330bq/branch/master?svg=true)](https://ci.appveyor.com/project/vineetbansal/aspire-python/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/ComputationalCryoEM/ASPIRE-Python/badge.svg?branch=master)](https://coveralls.io/github/ComputationalCryoEM/ASPIRE-Python?branch=master)
[![Documentation Status](https://readthedocs.org/projects/aspire/badge/?version=latest)](https://aspire.readthedocs.io/en/latest/?badge=latest)

# ASPIRE - Algorithms for Single Particle Reconstruction

This is the Python version to supersede the [Matlab ASPIRE](https://github.com/PrincetonUniversity/aspire). 

ASPIRE is an open-source software package for processing single-particle cryo-EM data to determine three-dimensional structures of biological macromolecules. The package includes advanced algorithms based on rigorous mathematics and recent developments in
statistics and machine learning. It provides unique and improved solutions to important computational challenges of the cryo-EM
processing pipeline, including 3-D *ab-initio* modeling, 2-D class averaging, automatic particle picking, and 3-D heterogeneity analysis.

For more information about the project, algorithms, and related publications please refer to the [ASPIRE Project website](http://spr.math.princeton.edu/).


## Installation Instructions

For end-users
-------------

ASPIRE is a pip-installable package that works on Linux/Mac/Windows, and requires Python 3.6. The simplest option is to use Anaconda 64-bit for your platform with a minimum of Python 3.6 and pip, and then use `pip` to install `aspire` in that environment.

```
conda create -n aspire_env python=3.6 pip
conda activate aspire_env
pip install aspire
```

The final step above should install any dependent packages from `pip` automatically.

Note that this step installs the base `aspire` package for you to work with, but not the unit tests/documentation. If you need to install ASPIRE for development purposes, read on.

For developers
--------------

After cloning this repo, the simplest option is to use Anaconda 64-bit for your platform, and use the provided `environment.yml` file to build a Conda environment to run ASPIRE.

```
cd /path/to/git/clone/folder
conda env create -f environment.yml
conda activate aspire
```

### Make sure everything works

Once ASPIRE is installed, make sure the unit tests run correctly on your platform by doing:
```
cd /path/to/git/clone/folder
python setup.py test
```

Tests currently take around 2 minutes to run. If some tests fail, you may realize that `python setup.py test` produces too much information.
You may want to re-run tests using:
```
cd /path/to/git/clone/folder
PYTHONPATH=./src pytest tests
```
This provides a cleaner output to analyze.

### Install

If the tests pass, install the ASPIRE package for the currently active Conda environment:
```
cd /path/to/git/clone/folder
python setup.py install
```
