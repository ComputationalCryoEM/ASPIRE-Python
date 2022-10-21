import importlib
import logging.config
import os
import pkgutil
from datetime import datetime
from importlib.resources import read_text
from pathlib import Path

import aspire
from aspire.config import Config
from aspire.exceptions import handle_exception

# version in maj.min.bld format
__version__ = "0.10.0"

# Implements some code that writes out exceptions to 'aspire.err.log'.
config = Config(read_text(aspire, "config.ini"))
if config.logging.log_exceptions:
    import sys

    sys.excepthook = handle_exception

# Ensure the log_dir exists
Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)

# Generates file name details and opens log file defined in config file.
# The default is to use the current time stamp provided in the dictionary,
#   but that is not required if a user wishes to customize logging config.
logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), "logging.conf"),
    defaults={
        "dt_stamp": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"),
        "log_dir": config.logging.log_dir,
    },
)

__all__ = []
for _, modname, _ in pkgutil.iter_modules(aspire.__path__):
    __all__.append(modname)  # Add module to __all_
    importlib.import_module(f"aspire.{modname}")  # Import the module
