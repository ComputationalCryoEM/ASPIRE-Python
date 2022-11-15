import importlib
import logging.config
import os
import pkgutil
from datetime import datetime
from importlib.resources import read_text
from pathlib import Path

import confuse

import aspire
from aspire.exceptions import handle_exception

# version in maj.min.bld format
__version__ = "0.10.0"


# Setup `confuse` config
config = confuse.LazyConfig("ASPIRE", __name__)

# Ensure the log_dir exists.
# TODO: Discuss the behavior/location of log_dir
log_dir = config["logging"]["log_dir"].get(confuse.Filename(cwd="."))
Path(log_dir).mkdir(parents=True, exist_ok=True)

# Generates file name details and opens log file defined in config file.
# The default is to use the current time stamp provided in the dictionary,
#   but that is not required if a user wishes to customize logging config.
logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), "logging.conf"),
    defaults={
        "dt_stamp": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"),
        "log_dir": log_dir,
    },
)

# Log where the package resolves `config_dir()` at this time
logging.debug(f"ASPIRE initial configuration directory is {config.config_dir()}")

# Implements some code that writes out exceptions to 'aspire.err.log'.
if config["logging"]["log_exceptions"].get(int):
    import sys

    sys.excepthook = handle_exception


__all__ = []
for _, modname, _ in pkgutil.iter_modules(aspire.__path__):
    __all__.append(modname)  # Add module to __all_
    importlib.import_module(f"aspire.{modname}")  # Import the module
