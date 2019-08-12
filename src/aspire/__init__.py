from .version import version as __version__


from importlib_resources import read_text
import logging.config

import aspire
from aspire.utils.config import Config


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
        "errfile": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "aspire.err.log",
            "formatter": "simple_formatter",
            "level": "ERROR"
        }
    },
    "loggers": {
        "aspire": {
            "level": "DEBUG",
            "handlers": ["console", "errfile"]
        }
    }
})

config = Config(read_text(aspire, 'config.ini'))
