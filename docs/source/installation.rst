Installation
============

This package is tested on Linux/Windows/Mac OS X. Pre-built binaries
should be available for platform-specific dependencies. No manual
compilation should be needed.

For end users who simply want to use or run scripts depending on
ASPIRE, installing the ``aspire`` package from PyPI is sufficient.

.. note:: Installing the package installs ASPIRE to the
    ``site-packages`` folder of your active environment.  This is only
    desirable if you are not going to be doing any development on
    ASPIRE, and only intend to run scripts that depend on the ASPIRE
    package.

For those who wish to develop, we recommend starting with the
instructions on our README (copied below). Additionally some more
advanced instructions are provided here for installing with software
and hardware optimizations.  Although not explicitly required, For
developers and users not confident in software management the use of
``conda`` is strongly encouraged.


Install Conda
*************

To follow the suggested installation, you will need to install Conda
for **Python3**, either `Anaconda
<https://www.anaconda.com/download/#linux>`__ or `Miniconda
<https://conda.io/miniconda.html>`__, click on the right distribution
to view Conda's installation instructions.

.. note:: If you're not sure which distribution is right for you, go
   with `Miniconda <https://conda.io/miniconda.html>`__

.. note:: For Apple silicon to use the osx-arm platform, patching and
   building some dependencies from source is currently required.  The
   Intel ``osx-64`` install is still preferred even for Apple silicon
   users, otherwise `notes are
   provided. <https://github.com/ComputationalCryoEM/ASPIRE-Python/discussions/969>`_

Getting Started - Installation
************************************

Python 3.8 is used as an example, but the same procedure should work
for any of our supported Python versions 3.8-3.11. Below we pip install
the ``aspire`` package using the ``-e`` flag to install the project in
editable mode. The ``".[dev]"`` command installs ``aspire`` from the local
path with additional development tools such as pytest and Jupyter Notebook.
See the `pip documentation <https://pip.pypa.io/en/stable/cli/pip_install/#options>`__
for more details on using pip install.


::

   # Clone the code
   git clone https://github.com/ComputationalCryoEM/ASPIRE-Python.git
   cd ASPIRE-Python

   # Create a fresh environment
   conda create --name aspire python=3.8 pip

   # Enable the environment
   conda activate aspire

   # Install the ``aspire`` package from the checked out code
   # with the additional ``dev`` extension.
   pip install -e ".[dev]"

.. note:: Required dependent packages supporting Python 3.11 are currently only supported on Linux.


Test the package
****************

Make sure all unit tests run correctly by doing:

::

    pytest

Tests currently take around 15 minutes to run, but this depends on
your specific machine's resources and configuration.

Optimized Numerical Backends
****************************

For advanced users, ``conda`` provides optimized numerical backends
that offer significant performance improvements on appropriate
machines.  The backends accelerate the performance of ``numpy``,
``scipy``, and ``scikit`` packages.  ASPIRE ships several
``environment*.yml`` files which define tested package versions along
with these optimized numerical installations.

The default ``environment-default.yml`` does not force a specific
backend, instead relying on ``conda`` to select something reasonable.
In the case of an Intel machine, the default ``conda`` install will
automatically install some optimizations for you.  However, these
files can be used to specify a specific setup or as the basis for your
own customized ``conda`` environment.

.. list-table:: Suggested Conda Environments
   :widths: 25 25
   :header-rows: 1

   * - Architecture
     - Recommended Environment File
   * - Default
     - environment-default.yml
   * - Intel x86_64
     - environment-intel.yml
   * - AMD x86_64
     - environment-openblas.yml
   * - Apple M1
     - environment-accelerate.yml

Using any of these environments follows the same pattern outlined
below.  As an example to specify using the ``accelerate`` backend on
an M1 laptop:

::

   cd ASPIRE-Python
   conda env create -f environment-accelerate.yml --name aspire_acc
   conda activate aspire_acc
   pip install -e ".[dev]"

Installing GPU Extensions
*************************

ASPIRE does support using a GPU, depending on several external
packages.  The collection of GPU extensions can be installed using
``pip``.  Extensions are grouped based on CUDA versions.  To find the
CUDA driver version, run ``nvidia-smi`` on the intended system.

.. list-table:: CUDA GPU Extension Versions
   :widths: 25 25
   :header-rows: 1

   * - CUDA Version
     - ASPIRE Extension
   * - >=12
     - gpu-12x

For example, if you have CUDA 12.3 installed on your system,
the command below would install GPU packages required for ASPIRE.

::

    # From a local git repo
    pip install -e ".[gpu-12x]"

    # From PyPI
    pip install "aspire[gpu-12x]"

    
By default if the required GPU extensions are correctly installed,
ASPIRE should automatically begin using the GPU calls to our ``nufft`` module.

Using GPU in other areas of the code is still an experimental feature
and requires a minor configuration setting to enable ``cupy``. See the
:ref:`sphx_glr_auto_tutorials_configuration.py` for details. Because
GPU extensions depend on several third party softwares and machines
vary wildly, we can only offer limited support if one of the packages
has a problem on your system.  We are currently expanding GPU code
coverage.

Generating Documentation
************************

Sphinx Documentation of the source (a local copy of what you're
looking at right now) can be generated by using the following commands
from the root of the code repository.

ASPIRE has both traditional documentation and a gallery of tutorial
scripts.  To make only the documentation run ``make html-noplot``.
The ``make html`` command makes the traditonal documentation then runs
and renders the ``gallery/tutorials`` examples, which takes several
minutes.

::

    cd docs

    # Parse the code in ``src``
    sphinx-apidoc -f -o ./source ../src -H Modules

    make html-noplot  # Generate only documentation
    # or
    make html         # Generate documentation and gallery examples

    # To remove any documentation build artifacts
    make distclean

The resulting html files can be found at ``docs/build/html``.
