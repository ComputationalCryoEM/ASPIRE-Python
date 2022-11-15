"""
ASPIRE Configuration
====================

Review the default ASPIRE configuration and common patterns for overriding.
"""

# %%
# Default Configuration
# ---------------------
# ASPIRE uses the `confuse library`_ for managing configuration.
# While this document should cover common uses,
# advanced users and developers should consider reviewing their documentation,
#
# The ASPIRE package ships with ``config_default.yaml``.
# This represents a base configuration.
# The final block in this tutorial should print the complete configuration
# for this version of ASPIRE.

# %%
# System Overrides
# ----------------
# From here we can override with a custom config file in your home dir,
# specifically ``$HOME/.config/ASPIRE/config.yaml`` on most Linux platforms.
# Items in this file will take precedence over the default configuration.
#
# Consider wanting to to always store ASPIRE logs at ``/tmp/aspire_logs``
# when working on a specific machine.
# By creating ``$HOME/.config/ASPIRE/config.yaml`` with the following contents
# on that machine, ASPIRE's configuration utility will overload
# the logging directory from a local ``logs`` folder to ``/tmp/aspire_logs``.

# %%
"""
logging:
   # I prefer to log to /tmp so they get cleaned up when I reboot.
   log_dir: /tmp/logs
"""

# %%
# Local Overrides
# ---------------
# A file in your working directory will take precedence over system-overrides.
# Consider a user who would like to run a specific experiment using a different `nufft`.

# %%
# In-Code Overrides
# -----------------
# You can also specify your own file from an arbitrary location from within Python code.
#
# Or simply set specific variables.

# %%
# Resolution
# ----------
# ASPIRE logs the ``config_dir()`` for your system at startup,
# along with the configuration sources and resolved configuration at import time.
# This should give an accurate snapshot of the configuration before any in-code overrides.
# To view these as save in your log, you will need to locate your `log_dir`.
# If you are not sure where it is, we can ask the config:

import aspire

print(aspire.config["logging"]["log_dir"].as_filename())

# %%
# You can also resolve the config in code
print(aspire.config.dump())


# %%
# .. _confuse library: https://confuse.readthedocs.io/en/latest/index.html
