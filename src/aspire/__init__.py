__version__ = "0.5.1"


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
