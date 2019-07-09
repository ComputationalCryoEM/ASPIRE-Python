Getting Started
===============

The following top-level scripts are currently available in the ``scripts`` folder to run different stages of the Cryo-EM data pipeline.
To run these, make sure you have activated the proper conda environment for ASPyRE (see :doc:`installation` for installation instructions).

1. Particle-Picking
*******************

The ``apple.py`` script takes in a folder of one or more ``*.mrc`` files, picks particles using the Apple-Picker algorithm described at
:cite:`DBLP:journals/corr/abs-1802-00469`, and generates ``*.star`` files, one for each ``*.mrc`` file processed, at an output folder location.

For example, to run the script on sample data included in ASPyRE (a single ``falcon_2012_06_12-14_33_35_0.mrc`` file provided from the 5.3 GB
`Beta-galactosidase Falcon-II micrographs EMPIAR dataset <https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/>`_) and save results to a
``particles`` folder:

::

    cd scripts
    mkdir apple_output
    python apple.py --mrc_file ../tests/saved_test_data/mrc_files/falcon_2012_06_12-14_33_35_0.mrc --output_dir particles

Use the ``--help`` argument with the script and look for the several ``config.apple.*`` arguments that you can tweak for this script.

2. Simulation
*************

The ``simulation.py`` script simulates a virtual particle made up of multiple Gaussian blobs, generates of set of (noisy) images,
runs the ASPyRE pipeline to determine the estimated mean volume and estimated covariance on the mean volume,
and runs evaluations on these estimated quantities (against the `true` values which we know from the simulation).

::

    cd scripts
    python simulation.py

Use the ``--help`` argument with the script to look for configurable values. You can select the no. of distinct Gaussian blobs, the no. of images,
the resolution of the (square) images generated etc.

3. Processing Starfiles
***********************

The ``starfile.py`` script takes in a ``*.star`` file, processes the images (*.mrcs files) found in the starfile, and runs the ASPyRE pipeline
to determine the estimated mean volume and estimated covariance on the mean volume. No results are saved by the script, but this script is
a good way to exercise most parts of the ASPyRE pipeline.

For example, to run the script on sample data included in ASPyRE:

::

    cd scripts
    python starfile.py --starfile ../tests/saved_test_data/starfile.star --ignore_missing_files --pixel_size 1.338 -L 8 --config.covar.cg_tol 0.2

.. note::

    Pay special attention to the flags specified in the example above. The ``--ignore_missing_files`` flag ignores any *.mrcs files
    referenced by the starfile but not found in the filesystem (as is the case with sample data provided in ASPyRE). The ``-L 8``
    flag down-samples images to 8x8 pixels (needed otherwise you may run out of memory, and/or the script may take way too long to execute).
    The ``config.covar.cg_tol`` flag is explained below.

The ``starfile.py`` script also provides an example of how to override configuration values (that are read from ``config.json`` file
provided in the package. In the above example, ``config.covar.cg_tol 0.2`` sets very liberal (and unrealistic) limits on optimization convergence
tolerance, which is needed for such a small dataset. For real datasets, you typically *do not* want to override this parameter.

All configuration variables of the package (found in ``config.json``) can be modified on a per-script basis by specifying the ``--config.*`` arguments
to the script. Use the ``--help`` option for a script to see all available configuration variables.

4. Invoking Aspire CLI tool
***************************

`aspire` is a command-line-interface (CLI) application allowing you to perform actions on a stack of
CRYO projections (MRC files). To invoke the tool simply run::

   python aspire.py

You'll see a help message showing you the various available commands.

Arguments, options and flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Arguments** are mandatory inputs.
   For example, when running 'compare' command, you must provide 2 MRC files to compare.
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


.. bibliography:: references.bib