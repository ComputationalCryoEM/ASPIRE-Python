import click

from aspire import config
from aspire.utils.logging import pythonLoggingLevelDict

# universal option for aspire commands
_log_levels = pythonLoggingLevelDict().keys()
log_level_option = click.option(
    "--loglevel",
    type=click.Choice(_log_levels, case_sensitive=False),
    default=config["logging"]["console_level"].as_choice(_log_levels),
    help="Logging verbosity level of console output.",
)
