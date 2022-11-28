"""
ASPIRE Configuration
====================

This tutorial reviews the default ASPIRE configuration
and common patterns for overriding.
"""

# %%
# Default Configuration
# ---------------------
# ASPIRE uses the `confuse library`_ for managing configuration.
# While this document should cover common uses,
# advanced users and developers should consider reviewing their documentation.
#
# The ASPIRE package ships with ``config_default.yaml``.
# This represents a base configuration.
# The shipped configuration for this version of ASPIRE is:
#
# .. literalinclude:: ../../../src/aspire/config_default.yaml


# %%
# System Overrides
# ----------------
# From here we can override with a custom config file in your home dir,
# specifically ``$HOME/.config/ASPIRE/config.yaml`` on most Linux platforms.
# Items in this file will take precedence over the default configuration.
# For other platforms, refer to the `confuse` documentation.
#
# As an example, suppose you want to change the ``ray`` ``temp_dir`` variable
# when working on a specific machine.
# By creating ``$HOME/.config/ASPIRE/config.yaml`` with the following contents
# on that machine, ASPIRE's configuration utility will overload
# the ``temp_dir`` directory from a ``/tmp/ray`` folder to ``/scratch/tmp/ray``.
#
#     .. code-block:: yaml
#
#       ray:
#         temp_dir: /scratch/tmp/ray
#

# %%
# Override Configuration Directory
# --------------------------------
# Users may specify a directory containing the configuration file.
# This is done by using the enviornment variable ``ASPIREDIR``
# If you wanted a file in your working directory to take
# precedence over system-overrides, we can create a local ``config.yaml``.
#
# Suppose you want to store ASPIRE logs at ``/tmp/my_proj/aspire_logs``
# when working on a specific project. Create the following ``config.yaml``.
#
#     .. code-block:: yaml
#
#       logging:
#         log_dir: /tmp/my_proj/logs
#
# This directory must then be set before invoking any code.
#
#     .. code-block:: bash
#
#       export ASPIREDIR=$PWD
#
# Similarly, you could specify any directory you like that might contain a configuration.
# This allows you to store configurations for reuse.

# %%
# In-Code Overrides
# -----------------
# You can also specify your own file from an arbitrary location from within Python code.
# For precise behavior refer to the confuse documentation.
#
#     .. code-block:: python
#
#       aspire.config.set_file('/path/to/some_experimental_config.yaml')
#
#
# Or simply set specific variables as needed.
# Here we will disable progress bars displayed by
# ``aspire.utils.trange`` and ``aspire.utils.tqdm``.
#

import time

from aspire import config
from aspire.utils import trange

# Progress bars are displayed by default.

print("Disabling progress bars")
config["logging"]["tqdm_disable"] = True

for _ in trange(3):
    time.sleep(1)
print("Done Loop 1\n")

print("Re-enabling progress bars")
config["logging"]["tqdm_disable"] = False

for _ in trange(3):
    time.sleep(1)
print("Done Loop 2\n")


# %%
# Resolution
# ----------
# ASPIRE logs the ``config_dir()`` for your system at startup,
# along with the configuration sources and resolved configuration at import time.
# This should give an accurate snapshot of the configuration before any in-code overrides.
# To view these as saved in your log, you will need to locate your `log_dir`.
# If you are not sure where it is, we can ask the config:

import aspire

print(aspire.config["logging"]["log_dir"].as_filename())

# %%
# You can also resolve the config in code
print(aspire.config.dump())


# %%
# .. _confuse library: https://confuse.readthedocs.io/en/latest/index.html
