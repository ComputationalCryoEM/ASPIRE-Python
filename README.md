[![Azure Build Status](https://dev.azure.com/vineetbansal0645/ASPyRE/_apis/build/status/computationalcryoem.aspyre?branchName=master)](https://dev.azure.com/vineetbansal0645/ASPyRE/_build/latest?definitionId=2&branchName=master)
[![Travis Build Status](https://travis-ci.org/computationalcryoem/aspyre.svg?branch=master)](https://travis-ci.org/computationalcryoem/aspyre)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/5yn93qobpptnw2dw/branch/master?svg=true)](https://ci.appveyor.com/project/vineetbansal/aspyre-mu8e3/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/computationalcryoem/aspyre/badge.svg?branch=master)](https://coveralls.io/github/computationalcryoem/aspyre?branch=master)
[![Documentation Status](https://readthedocs.org/projects/aspyre/badge/?version=latest)](https://aspyre.readthedocs.io/en/latest/?badge=latest)

# ASPyRE

Algorithms for Single Particle Reconstruction

## Installation Instructions

### Linux/Mac OS X/Windows

The simplest option is to use Anaconda 64-bit for your platform, and use the provided `environment.yml` file to build a Conda environment to run ASPyRE.

```
cd /path/to/git/clone/folder
conda env create -f environment.yml
conda activate aspyre
```

## Make sure everything works

Once ASPyRE is installed, make sure the unit tests run correctly on your platform by doing:
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

If the tests pass, install the ASPyRE package for the currently active Conda environment:
```
cd /path/to/git/clone/folder
python setup.py install
```

## Development Guidelines

ASPyRE follows [Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
Please submit any PRs against the `develop` branch.

![Gitflow Diagram](https://wac-cdn.atlassian.com/dam/jcr:61ccc620-5249-4338-be66-94d563f2843c/05%20(2).svg?cdnVersion=357)
