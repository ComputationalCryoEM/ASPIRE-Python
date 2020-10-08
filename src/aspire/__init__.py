import os
import logging.config

from datetime import datetime
from importlib_resources import read_text

import aspire
from aspire.exceptions import handle_exception
from aspire.utils.config import Config

# version in maj.min.bld format
__version__ = '0.6.0'

# Generates file name and opens log file defined in config file.
# The default is to use the current time stamp provided in the dictionary,
#   but that is not required if a user wishes to override filename.
logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), 'logging.conf'),
    defaults={'dt_stamp': datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')})

# Implements some code that writes out exceptions to 'aspire.err.log'.
config = Config(read_text(aspire, 'config.ini'))
if config.common.log_exceptions:
    import sys
    sys.excepthook = handle_exception
