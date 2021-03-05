import pkgutil

from click.core import Command, Group

import aspire.commands


def main_entry():
    main = Group(chain=False)

    # TODO: Add options
    # @click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
    # @click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')

    for importer, modname, _ in pkgutil.iter_modules(aspire.commands.__path__):
        module = importer.find_module(modname).load_module(modname)
        commands = [v for v in module.__dict__.values() if isinstance(v, Command)]
        for command in commands:
            main.add_command(command)

    main.main(prog_name="aspire")


if __name__ == "__main__":
    main_entry()
