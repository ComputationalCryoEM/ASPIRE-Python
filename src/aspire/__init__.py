import importlib
import logging.config
import os
import pkgutil
from datetime import datetime
from pathlib import Path

import confuse
import pooch

import aspire
from aspire.exceptions import handle_exception

# version in maj.min.bld format
__version__ = "0.13.0"


# Setup `confuse` config
config = confuse.Configuration("ASPIRE", __name__)

# Ensure the log_dir exists.
log_dir_path = Path(config["logging"]["log_dir"].get(confuse.Filename(cwd=".")))
log_dir_path.mkdir(parents=True, exist_ok=True)
# We'll reassign the evaluated log_dir back into the config so it displays well.
config["logging"]["log_dir"] = log_dir_path.as_posix()

# log output file prefix
log_prefix = config["logging"]["log_prefix"].get(str)

# DEBUG, INFO, etc.
_logging_level_names = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
console_level = config["logging"]["console_level"].as_choice(_logging_level_names)
log_file_level = config["logging"]["log_file_level"].as_choice(_logging_level_names)

# Generates file name details and opens log file defined in config file.
# The default is to use the current time stamp provided in the dictionary,
#   but that is not required if a user wishes to customize logging config.
logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), "logging.conf"),
    defaults={
        "console_level": console_level,
        "log_file_level": log_file_level,
        "log_dir": log_dir_path.as_posix(),
        "log_prefix": log_prefix,
        "dt_stamp": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"),
    },
)

# Log where the package resolves `config_dir()`.
logging.debug(f"ASPIRE configuration directory is {config.config_dir()}")

# Log the resolution of configuration (ie overrides).
# The list is a stack of configuration sources,
#   with each entry specifying the variables overridden.
logging.debug(
    f"ASPIRE configuration resolution details {list(aspire.config.resolve())}"
)

# Dump the config at `aspire` import time to our log.
logging.debug(f"Resolved config.yaml:\n{aspire.config.dump()}\n")

# Set cache location for ASPIRE example data.
if not config["common"]["cache_dir"].get():
    config["common"]["cache_dir"] = pooch.os_cache("ASPIRE-data").as_posix()

# Implements some code that writes out exceptions to 'aspire.err.log'.
if config["logging"]["log_exceptions"].get(int):
    import sys

    sys.excepthook = handle_exception

# Collect set of all module names in package
_modules = set(item[1] for item in pkgutil.iter_modules(aspire.__path__))

# Automatically add modules
__all__ = []
for modname in _modules:
    __all__.append(modname)  # Add module to __all_


# Dynamically load and return attributes
def __getattr__(attr):
    if attr in _modules:
        return importlib.import_module(f"aspire.{attr}")
    else:
        raise AttributeError(f"module `{__name__}` has no attribute `{attr}`.")
