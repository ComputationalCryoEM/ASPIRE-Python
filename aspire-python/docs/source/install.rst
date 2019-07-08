.. include:: header.rst

Requirements (Linux)
^^^^^^^^^^^^^^^^^^^^

1. Conda
""""""""
Conda for **Python3**, either
`Anaconda <https://www.anaconda.com/download/#linux>`__ or
`Miniconda <https://conda.io/miniconda.html>`__, click on the right
distribution to view Conda's installation instructions.

.. note::
   If you're not sure which distribution is right for you, go with `Miniconda <https://conda.io/miniconda.html>`__


2. Linux packages
"""""""""""""""""

In order to install Python package ``Finufftpy`` you need to preinstall these. Run::

   sudo apt install -y --upgrade gcc g++ libfftw3-bin libfftw3-dev

This command was tested on Ubuntu 16TLS.
For other Linux distros please use the substitute command for ``apt`` (yum, apt-get, brew, etc.)

Conda environment
^^^^^^^^^^^^^^^^^

Creating environment for ASPIRE
"""""""""""""""""""""""""""""""
Run::

   conda env create -f environment.yml

**Common errors:**


- ``"Prefix already exists .../envs/aspire"``
   - Please delete the directory shown and try again.

- ``"Can't process without a name"``
   - You're not in the right folder. Please run under root folder of ASPIRE-Python.

Activating ASPIRE environment
"""""""""""""""""""""""""""""

Run::

   source activate aspire

.. attention::

   When you're done working with ASPIRE. It is highly recommended that you **deactivate** ASPIRE environment::

      source deactivate

.. note::
   Depending on your Conda distribution, in some cases you should run::

      conda activate aspire  # for activation
      conda deactivate # for deactivation


Installing finufftpy
^^^^^^^^^^^^^^^^^^^^

Assuming all Linux packages from `Requirements <#linux-packages>`__ are installed, run::

    make finufftpy

Downloading data files
^^^^^^^^^^^^^^^^^^^^^^
For some commands, you'll need to have certain binary files available for Aspire.
To download them simply Run::

   make data

Assuming no errors, you can now use Aspire tool.
