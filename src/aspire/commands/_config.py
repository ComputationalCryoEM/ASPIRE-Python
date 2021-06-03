import functools

import click

# Wraps click.option to set global defaults for ASPIRE commands.
# Currently this just sets show_default.
# To use, also import the following when importing and defining click commands:
# # `import aspire.commands._config`
click.option = functools.partial(click.option, show_default=True)
