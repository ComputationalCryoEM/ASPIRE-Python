from .version import version as __version__

from importlib_resources import read_text
import logging.config

import aspire
from aspire.utils.config import Config
from aspire.exceptions import handle_exception


logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "simple_formatter": {
            "format": "%(asctime)s %(message)s",
            "datefmt": "%Y/%m/%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple_formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        },
        "error_file": {
            "class": "logging.FileHandler",
            "mode": "w",
            "filename": "aspire.err.log",
            "formatter": "simple_formatter",
            "level": 1  # A 'low' number as compared to other handlers.
        }
    },
    "loggers": {
        "aspire": {
            "level": 1,
            "handlers": ["console", "error_file"]
        }
    }
})

config = Config(read_text(aspire, 'config.ini'))
if config.common.log_errors:
    import sys
    sys.excepthook = handle_exception
