import logging.config

from importlib_resources import read_text

import aspire
from aspire.exceptions import handle_exception
from aspire.utils.config import Config

# version in maj.min.bld format
__version__ = '0.5.28'

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
        }
    },
    "loggers": {
        "aspire": {
            "level": "DEBUG",
            "handlers": ["console"]
        }
    }
})

config = Config(read_text(aspire, 'config.ini'))
if config.common.log_errors:
    import sys
    sys.excepthook = handle_exception
