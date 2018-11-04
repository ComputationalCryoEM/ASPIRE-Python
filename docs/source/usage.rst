.. include:: header.rst


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
