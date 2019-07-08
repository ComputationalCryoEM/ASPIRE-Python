.. This is README.rst which shows on the Github repo

.. raw:: html

    <p align="center">
        <img src="http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg"/>
    </p>


ASPIRE-Python
-------------

|Python 3.6| |Documentation Status|

.. |Python 3.6| image:: https://img.shields.io/badge/python-3.6-blue.svg
   :target: https://www.python.org/downloads/release/python-360/
.. |Documentation Status| image:: https://readthedocs.org/projects/aspire-python/badge/?version=latest
   :target: https://aspire-python.readthedocs.io/en/latest/?badge=latest

Installation
------------

Check out our `ReadTheDocs page <https://aspire-python.readthedocs.io/en/latest/install.html>`__
for a more detailed step-by-step installation process, common errors and how to solve them.
If you still encounter troubles setting up things, please write directly to devs.aspire@gmail.com.

Requirements (Linux)
^^^^^^^^^^^^^^^^^^^^

1. Conda
""""""""
Conda for **Python3**, either
`Anaconda <https://www.anaconda.com/download/#linux>`__ or
`Miniconda <https://conda.io/miniconda.html>`__, click on the right
distribution to view Conda's installation instructions.

If you're not sure which distribution is right for you, go with `Miniconda <https://conda.io/miniconda.html>`__


2. Linux packages
"""""""""""""""""

In order to install Python package ``Finufftpy`` you need to have these installed. Run::

   sudo apt install -y --upgrade gcc g++ libfftw3-bin libfftw3-dev


Conda environment
^^^^^^^^^^^^^^^^^

Creating environment for ASPIRE
"""""""""""""""""""""""""""""""
Run::

   conda env create -f environment.yml


Activating ASPIRE environment
"""""""""""""""""""""""""""""

Run::

   source activate aspire


Installing finufftpy
^^^^^^^^^^^^^^^^^^^^

Assuming all Linux packages from `Requirements <#linux-packages>`__ are installed, run::

    make finufftpy

Downloading data files
^^^^^^^^^^^^^^^^^^^^^^
For certain commands, you'll need to have some binary files available for Aspire.
To download them simply Run::

   make data

Assuming no errors, you can now use Aspire tool.


Usage
-----

For the complete documentation, list of available commands, common errors and more, please check out our `ReadTheDocs page <https://aspire-python.readthedocs.io/en/latest/usage.html>`__

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


Basic Examples
^^^^^^^^^^^^^^

-  If you want to **view the help message for a specific command**, please place '-\\-help' **after**
   that command. will only present the help message for the highest layer.::

      python aspire.py compare --help  # help for compare
      python aspire.py --help compare  # help for aspire (root command)

-  **Crop a stack of projections of an mrc file to squares of 42x42 px**,
   in debug mode and with maximum verbosity::

      python aspire.py --debug -v 3 crop demmo.mrc 42
