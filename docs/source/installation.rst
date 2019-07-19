Installation
============

ASPIRE runs on Python 3.6, and comes with an ``environment.yml`` to reconstruct a working Conda environment to run the package.
The package is tested on Linux/Windows/Mac OS X. Pre-built binaries are available for all platform-specific components. No manual
compilation should be needed.

Install Conda
*************

You will need to install Conda for **Python3**, either
`Anaconda <https://www.anaconda.com/download/#linux>`__ or
`Miniconda <https://conda.io/miniconda.html>`__, click on the right
distribution to view Conda's installation instructions.

.. note::
   If you're not sure which distribution is right for you, go with `Miniconda <https://conda.io/miniconda.html>`__

Install and Activate the environment
************************************

Once `conda` is installed and available on the path, create and activate the environment using:

::

    cd /path/to/git/clone/folder
    conda env create -f environment.yml
    conda activate aspire

Test the package
****************

Make sure all unit tests run correctly by doing:

::

    cd /path/to/git/clone/folder
    python setup.py test

Tests currently take around 5 minutes to run. If some tests fail, you may realize that ``python setup.py test`` produces too much information. Re-running tests using ``py.test tests`` in ``/path/to/git/clone/folder`` may provide a cleaner output to analyze.

Install the package
*******************

If the tests pass, install the ASPIRE package for the currently active Conda environment:

::

    cd /path/to/git/clone/folder
    python setup.py install

.. note::
    Installing the package installs ASPIRE to the ``site-packages`` folder of your active environment.
    This is only desirable if you are not going to be doing any development on ASPIRE,
    but simply want to run scripts that depend on the ASPIRE package.

Generating Documentation
************************

Sphinx Documentation of the source (a local copy of what you're looking at right now) can be generated using:

::

    cd /path/to/git/clone/folder/docs
    sphinx-apidoc -f -o ./source ../src -H Modules
    make clean
    make html

The built html files can be found at ``/path/to/git/clone/folder/docs/build/html``