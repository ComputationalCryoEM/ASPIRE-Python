import click
from aspire import config

# universal option for aspire commands
log_level_option = click.option(
    "--loglevel",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default=config["logging"]["console_level"].get(),
    help="Logging verbosity level of console output.",
)
