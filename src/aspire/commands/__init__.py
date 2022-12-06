import click

from aspire import config

# universal option for aspire commands
_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
log_level_option = click.option(
    "--loglevel",
    type=click.Choice(
        _log_levels, case_sensitive=False
    ),
    default=config["logging"]["console_level"].as_choice(_log_levels),
    help="Logging verbosity level of console output.",
)
