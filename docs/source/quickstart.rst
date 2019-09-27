Getting Started
===============

After installing ASPIRE, the module can be invoked as a script, allowing you to perform several actions on a stack of
CRYO projections (MRC files).

.. code-block:: console

    python -m aspire <command>

Running the ``aspire`` module as a script allows one to run different stages of the Cryo-EM data pipeline.
Substitute ``<command>`` with one of the available ``aspire`` commands. Use the ``-h`` flag to display all available commands.

Currently, the following operations can be run with ASPIRE:

1. Particle-Picking
*******************

The ``apple`` command takes in a folder of one or more ``*.mrc`` files, picks particles using the Apple-Picker algorithm described at
:cite:`DBLP:journals/corr/abs-1802-00469`, and generates ``*.star`` files, one for each ``*.mrc`` file processed, at an output folder location.

For example, to run the command on sample data included in ASPIRE (a single ``sample.mrc`` file provided from the 5.3 GB
`Beta-galactosidase Falcon-II micrographs EMPIAR dataset <https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/>`_) and save results to a
``particles`` folder:

.. code-block:: console

    mkdir apple_output
    python -m aspire apple --mrc_file /path/to/aspire/data/sample.mrc --output_dir particles

Use the ``--help`` argument with the command to see the several options associated with this command.

2. Simulation
*************

The ``simulation`` command simulates a virtual particle made up of multiple gaussian blobs, generates of set of (noisy) images,
runs the ASPIRE pipeline to determine the estimated mean volume and estimated covariance on the mean volume,
and runs evaluations on these estimated quantities (against the `true` values which we know from the simulation).

.. code-block:: console

    python -m aspire simulation

Use the ``--help`` argument to look for configurable options. You can select the no. of distinct gaussian blobs, the no. of images,
the resolution of the (square) images generated etc.

3. Reconstructing a mean volume with covariance
***********************************************

The ``cov3d`` command takes in a ``*.star`` file, processes the images (``*.mrcs`` files) found in the starfile, and runs the ASPIRE pipeline
to determine the estimated mean volume and estimated covariance on the mean volume. No results are saved currently, but this command is
a good way to exercise most parts of the ASPIRE pipeline.

For example, to run the command on a sample data included in ASPIRE:

.. code-block:: console

    python -m aspire cov3d --starfile /path/to/aspire/data/sample_relion_data.star --pixel_size 1.338 --max_resolution 8 --cg_tol 0.2

.. note::

    Pay special attention to the flags specified in the example above. The ``--max_resolution 8``
    flag down-samples images to 8x8 pixels (needed otherwise you may run out of memory, and/or the script may take way
    too long to execute). ``--cg_tol 0.2`` sets very liberal (and unrealistic) limits on optimization convergence
    tolerance, which is needed for such a small dataset. For real datasets, you typically *do not* want to override this
    parameter.

Use the ``--help`` argument to look for configurable options.

4. Crop a set of projections
****************************

The ``crop`` command crops a stack of projections of an mrc file to squares of a given size (in pixels). For example,

.. code-block:: console

      python -m aspire --debug -v 3 crop demo.mrc 42

.. note::

    This command will crop images found in `demo.mrc` to images of size 42x42, in debug mode and with maximum verbosity.


Arguments, options and flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Arguments** are mandatory inputs.
   For example, when running 'compare' command, you must provide 2 MRC files to compare.

- **Options** are, like their name suggests, optional inputs.
   For example, ``aspire`` accepts option '*-v 2*' for setting verbosity level to 2.
   All options have a default value set for them.

- **Flags** are optional values which tells Aspire to activate/deactivate certain behaviour.
   | A good example would be '*-\\-debug*'.
   | All flags also have a default value pre-set for them, '*-\\-no-debug*' in case of the *debug* flag.

Aspire CLI is built in levels. A level is basically a command which can
be followed by another command. The most basic command is ``aspire``
itself, the base layer. It accepts its own flags such as '*-\\-help*',
'*-\\-debug*' or '*-v N*'. Each of those optional flags will be directed into the **preceding** level.

Then we can call ``aspire`` with a command such as ``compare``, and
provide another layer of arguments, options and flags. For example, in case of ``compare`` these can be:

.. code-block:: console

   $ python -m aspire -v 2 --debug compare  a.mrc  b.mrc --max-error=0.123


.. bibliography:: references.bib