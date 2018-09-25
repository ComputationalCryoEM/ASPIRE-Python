import logging

from logging.handlers import  RotatingFileHandler

from aspire.config import AspireConfig


formatter = logging.Formatter('%(asctime)s %(name)-7s %(levelname)-7s %(message)s')

logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler()
file_handler = RotatingFileHandler("aspire.log", mode=AspireConfig.log_file_mode)

stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.setLevel(logging.INFO)
