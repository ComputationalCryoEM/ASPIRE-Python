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

For most end users, simply installing the package is sufficient to use ASPIRE.
The commands in this section should install ASPIRE directly from the ``Python Package Index`` into your activated environment.
This does not require checking out source code.
If you are interested in checking out and working with the source code, running tests, or a different flavor of install,
then skip to the next section now instead.

Once ``conda`` is installed and available on the path, create and activate the environment using:

::

   conda create -n aspire_env python=3.6 pip
   conda activate aspire_env
   pip install aspire

.. note::
    Installing the package installs ASPIRE to the ``site-packages`` folder of your active environment.
    This is only desirable if you are not going to be doing any development on ASPIRE,
    but simply want to run scripts that depend on the ASPIRE package.


Alternative Developer Installations
************************************

Developers are expected to be able to manage their own code and environments.
However, for consistency and newcomers, we recommend the following procedure using `conda`.

::

   # Acquire the code.
   git clone git@github.com:ComputationalCryoEM/ASPIRE-Python.git
   cd ASPIRE-Python

   # Create's the conda environment and installs base dependencies.
   conda env create -f environment.yml --name aspire_dev

   # Command to install the aspire package in a locally editable way:
   pip install -e .

We recommend using ``conda`` or a ``virutalenv`` environment managing solutions because ASPIRE may have conflicts or change installed versions of Python packages on your system.

Again, we recommend the above for consistency and safety.
However, ASPIRE is a ``pip`` package,
so you can attempt to install it using standard ``pip`` or ``setup.py`` commands.
ASPIRE should generally be compatible with newer version of Python,
and newer dependent packages, but this is still being tested.
There are a few known issues.
While we can try to help,
you may be on your own for support of this method of installation.

::
   # Standard pip site-packages installation command
   cd path/to/aspire-repo
   pip install .

   # Standard pip developer installation
   cd path/to/aspire-repo
   pip install -e .

Note that we hope to have a better automated coverage of ``pip`` installations on recent major Python versions for future releases.

Test the package
****************

Make sure all unit tests run correctly by doing:

::

    cd /path/to/git/clone/folder
    pytest tests

Tests currently take around 5 minutes to run.


Generating Documentation
************************

Sphinx Documentation of the source (a local copy of what you're looking at right now) can be generated using:

::

    cd /path/to/git/clone/folder/docs
    sphinx-apidoc -f -o ./source ../src -H Modules
    make clean
    make html

The built html files can be found at ``/path/to/git/clone/folder/docs/build/html``
