Getting Started
===============

After installing ASPIRE, the module can be invoked as a script, allowing you to perform several actions on a stack of
CRYO projections (MRC files).

.. code-block:: console

    aspire <command>

Running the ``aspire`` module as a script allows one to run different stages of the Cryo-EM data pipeline.
Substitute ``<command>`` with one of the available ``aspire`` commands. Use ``aspire --help`` to display all available commands and ``aspire <command> --help`` to display configurable options for a particular ``<command>``.

Currently, the following operations can be run with ASPIRE:

1. Particle-Picking
###################

The ``apple`` command takes in a file or folder of one or more ``*.mrc`` files, picks particles using the Apple-Picker algorithm described at
:cite:`DBLP:journals/corr/abs-1802-00469`, and generates ``*.star`` files, one for each ``*.mrc`` file processed, at an output folder location.

For example, to run the command on sample data included in ASPIRE (a single ``sample.mrc`` file provided from the 5.3 GB
`Beta-galactosidase Falcon-II micrographs EMPIAR dataset <https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/>`_) and save results to a
``particles`` folder:

.. code-block:: console

    mkdir apple_output
    aspire apple --mrc_path /path/to/aspire/data/sample.mrc --output_dir particles

2. Particle Extraction
######################

Given a dataset of full micrographs (``*.mrc`` file) and corresponding coordinate files containing the locations
of picked particles in the ``*.mrc``, the ``extract-particles`` command extracts the particles into one or more ``.mrcs``
stacks and generates a ``.star`` file.

Example usage:

.. code-block::

    aspire extract-particles --mrc_paths=my/data/sample.mrc --coord_paths=my/data/coords/sample.coord --starfile_out=my_dataset_stack.star --particle_size=256 --centers

3. Estimate Contrast Transfer Function
######################################

The ``estimate-ctf`` command estimates the CTF from experimental data and saves the CTF as an ``.mrc`` file.  For example,

.. code-block:: console

      aspire estimate-ctf --data_folder path_to_input_data_folder

.. note::

    This command expects data files are in the directory prescribed by ``--data_folder``,
    and will process all files with the extension ``.mrc`` and ``.mrcs`` contained there.
    This command will output ``.mrc`` files to a ``--output_dir`` (``./results`` by default).

4. Image Preprocessing
######################

The ``preprocess`` command takes in a ``*.star`` file representing particle stacks and applies a selection of preprocessing
methods such as phase flipping, downsampling, normalization to background noise, noise whitening, and contrast inversion.
Resulting images are saved as a starfile.

For example, to run the command on sample data included in ASPIRE:

.. code-block:: console

   aspire preprocess --starfile_in path/to/aspire/data/sample_relion_data.star --starfile_out preprocess_output.star --downsample 8

5. Image Denoising
##################

The ``denoise`` command takes in a ``*.star`` file, downsamples the images (``*.mrcs`` files) found in the starfile
to a desired resolution, then estimates the noise of the images and whitens that noise using the covariance
Weiner filtering method. The denoised images (``*.mrcs``) and a corresponding starfile are saved in an output folder.

For example, to run the command on sample data included in ASPIRE:

.. code-block:: console

   mkdir denoise_output
   aspire denoise --starfile_in path/to/aspire/data/sample_relion_data.star --starfile_out denoise_output/denoised_images.star

6. Orientation Estimation
#########################

The ``orient3d`` command takes in a ``*.star`` file contaning images and performs an orientation estimation using the
common lines algorithm employing synchronization and voting described at :cite:`DBLP:journals/siamis/ShkolniskyS12`.
The estimated rotations are saved in a starfile along with the original images.

For example, to run the command on sample data included in ASPIRE:

.. code-block:: console

   aspire orient3d --starfile_in path/to/aspire/data/sample_relion_data.star --starfile_out orient3d_output.star

7. Reconstructing a mean volume with covariance
###############################################

The ``cov3d`` command takes in a ``*.star`` file, processes the images (``*.mrcs`` files) found in the starfile, and runs the ASPIRE pipeline
to determine the estimated mean volume and estimated covariance on the mean volume. No results are saved currently, but this command is
a good way to exercise most parts of the ASPIRE pipeline.

For example, to run the command on a sample data included in ASPIRE:

.. code-block:: console

    aspire cov3d --starfile /path/to/aspire/data/sample_relion_data.star --pixel_size 1.338 --max_resolution 8 --cg_tol 0.2

.. note::

    Pay special attention to the flags specified in the example above. The ``--max_resolution 8``
    flag down-samples images to 8x8 pixels (needed otherwise you may run out of memory, and/or the script may take way
    too long to execute). ``--cg_tol 0.2`` sets very liberal (and unrealistic) limits on optimization convergence
    tolerance, which is needed for such a small dataset. For real datasets, you typically *do not* want to override this
    parameter.

Arguments, options and flags
############################

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

   $ aspire -v 2 --debug compare  a.mrc  b.mrc --max-error=0.123


.. bibliography:: references.bib
