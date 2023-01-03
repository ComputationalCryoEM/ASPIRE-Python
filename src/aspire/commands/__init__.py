import click

from aspire import config
from aspire.utils.logging import LOGGING_LEVEL_NAMES

# universal option for aspire commands
log_level_option = click.option(
    "--loglevel",
    type=click.Choice(LOGGING_LEVEL_NAMES, case_sensitive=False),
    default=config["logging"]["console_level"].as_choice(LOGGING_LEVEL_NAMES),
    help="Logging verbosity level of console output.",
)
