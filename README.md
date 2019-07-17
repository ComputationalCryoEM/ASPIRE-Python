![Logo](http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg)

[![Azure Build Status](https://dev.azure.com/vineetbansal0645/Aspire-Python/_apis/build/status/ComputationalCryoEM.ASPIRE-Python?branchName=master)](https://dev.azure.com/vineetbansal0645/Aspire-Python/_build/latest?definitionId=3&branchName=master)
[![Travis Build Status](https://travis-ci.org/ComputationalCryoEM/ASPIRE-Python.svg?branch=master)](https://travis-ci.org/ComputationalCryoEM/ASPIRE-Python)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/ywgud2vu9ot330bq/branch/master?svg=true)](https://ci.appveyor.com/project/vineetbansal/aspire-python/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/ComputationalCryoEM/ASPIRE-Python/badge.svg?branch=master)](https://coveralls.io/github/ComputationalCryoEM/ASPIRE-Python?branch=master)
[![Documentation Status](https://readthedocs.org/projects/aspire/badge/?version=latest)](https://aspire.readthedocs.io/en/latest/?badge=latest)

# ASPIRE

Algorithms for Single Particle Reconstruction

## Installation Instructions

### Linux/Mac OS X/Windows

The simplest option is to use Anaconda 64-bit for your platform, and use the provided `environment.yml` file to build a Conda environment to run ASPIRE.

```
cd /path/to/git/clone/folder
conda env create -f environment.yml
conda activate aspire
```

## Make sure everything works

Once ASPIRE is installed, make sure the unit tests run correctly on your platform by doing:
```
cd /path/to/git/clone/folder
python setup.py test
```

Tests currently take around 5 minutes to run. If some tests fail, you may realize that `python setup.py test` produces too much information.
You may want to re-run tests using:
```
cd /path/to/git/clone/folder
PYTHONPATH=./src pytest tests
```
This provides a cleaner output to analyze.

## Install

If the tests pass, install the ASPIRE package for the currently active Conda environment:
```
cd /path/to/git/clone/folder
python setup.py install
```

## Development Guidelines

ASPIRE follows [Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
Please submit any PRs against the `develop` branch.

![Gitflow Diagram](https://wac-cdn.atlassian.com/dam/jcr:61ccc620-5249-4338-be66-94d563f2843c/05%20(2).svg?cdnVersion=357)
