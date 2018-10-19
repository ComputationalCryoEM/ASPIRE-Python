.. This is README.rst which shows on the Github repo
   We use the same file for the documentation by importing parts of it (see markers)

.. raw:: html

    <p align="center">
        <img src="http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg"/>
    </p>


|Python 3.6| |Documentation Status|

.. |Python 3.6| image:: https://img.shields.io/badge/python-3.6-blue.svg
   :target: https://www.python.org/downloads/release/python-360/
.. |Documentation Status| image:: https://readthedocs.org/projects/aspire-python/badge/?version=latest
   :target: https://aspire-python.readthedocs.io/en/latest/?badge=latest

ASPIRE-Python
-------------

.. The following marker is for Sphinx documentation. Please don't remove any marker
   without being 100% sure you know what you're doing

.. marker-install-start


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

.. marker-install-end


Usage
-----
.. marker-usage-start

Invoking Aspire CLI tool
^^^^^^^^^^^^^^^^^^^^^^^^
Aspire is a command-line-interface (CLI) application allowing you to perform actions on a stack of
CRYO projections (MRC files). To invoke the tool simply run::

   python aspire.py

You'll see a help message showing you the various available commands.

Arguments, options and flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Arguments** are mandatory inputs.
   For example, when running 'compare' command, you must provide 2 MRC files to compoare.
- **Options** are, like their name suggests, optional inputs.
   For example, ``aspire.py`` accepts option '*-v 2*' for setting verbosity level to 2.
   All options have a default value set for them.
- **Flags** are optional values which tells Aspire to activate/deactivate certain behaviour.
   | A good example would be '*-\\-debug*'.
   | All flags also have a default value pre-set for them, '*-\\-no-debug*' in case of the *debug* flag.

Aspire CLI is built in levels. A level is basically a command which can
be followed by another command. The most basic command is ``aspire.py``
itself, the base layer. It accepts its own flags such as '*-\\-help*',
'*-\\-debug*' or '*-v N*'. Each of those optional flags will be directed into the **preceding** level.

Then we can call Aspire with a consequtive subcommand such as ``compare``, and
provide another layer of arguments, options and flags. In case of ``compare`` these can be:

.. code-block:: console

   $ python aspire.py -v 2 --debug compare  a.mrc  b.mrc --max-error=0.123

.. note::
   It is important to note that each command has to be followed by its own
   options/arguments/flags of that specific level, not more, not less.

Basic Examples
^^^^^^^^^^^^^^

-  If you want to **view the help message for a specific command**, please place '-\\-help' **after**
   that command. will only present the help message for the highest layer.::

      python aspire.py compare --help  # help for compare
      python aspire.py --help compare  # help for aspire (root command)

-  **Crop a stack of projections of an mrc file to squares of 42x42 px**,
   in debug mode and with maximum verbosity::

      python aspire.py --debug -v 3 crop demmo.mrc 42


**Common errors:**

-  ``ModuleNotFoundError: No module named 'click'``

   You're outside Conda's environment!
   Please `activate conda's env <installing.html#activating-conda-environment>`__
   (or `create conda's env <installing.html#creating-conda-environment>`__
   if you skipped the previous step 'Creating Conda environment'.

.. marker-usage-end
