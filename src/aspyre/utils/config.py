import configparser
from contextlib import contextmanager
from argparse import ArgumentParser


@contextmanager
def config_override(config, args):
    try:
        # save original section values so we can reapply them later
        old_values = {}
        for section_name in config.sections():
            section = getattr(config, section_name)
            old_values[section_name] = {}
            for k, v in section.items():
                old_values[section_name][k] = v

        for k, v in args.__dict__.items():
            if k.startswith('config.'):
                _, section_name, key_name = k.split('.')
                section = getattr(config, section_name)
                setattr(section, key_name, type(old_values[section_name][key_name])(v))
        yield args
    finally:
        for section_name in old_values:
            section = getattr(config, section_name)
            for k, v in old_values[section_name].items():
                setattr(section, k, v)


class ConfigSection:
    """
    A thin wrapper over a ConfigParser's SectionProxy object,
    that tries to infer the types of values, and makes them available as attributes
    Currently int/float/str are supported.
    """
    def __init__(self, section_proxy):
        self.d = {}  # key value dict where the value is typecast to int/float/str
        for k, v in section_proxy.items():
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if ',' in v:
                        v = [t.strip() for t in v.split(',')]
                    self.d[k] = v
                else:
                    self.d[k] = v
            else:
                self.d[k] = v

    def __setattr__(self, key, value):
        if key == 'd':
            return super().__setattr__(key, value)
        else:
            self.d[key] = value

    def __getattr__(self, item):
        if item != 'd':
            return self.d[item]

    def items(self):
        return self.d.items()


class Config:
    def __init__(self, ini_string=None):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.init_from_string(ini_string)

    def init_from_string(self, ini_string):
        self.config.read_string(ini_string)
        self._read_sections()

    def init_from_fp(self, ini_fp):
        self.config.read_file(ini_fp)
        self._read_sections()

    def _read_sections(self):
        for section in self.config.sections():
            setattr(self, section, ConfigSection(self.config[section]))

    def sections(self):
        return self.config.sections()


class ConfigArgumentParser(ArgumentParser):
    """
    An ArgumentParser that adds arguments found in the (flat) 'config' (of type Config) object used in it's
    constructor. By default, the aspyre.config Config object is used.
    All arguments to the parser are added with the 'config.' prefix
    """

    def __init__(self, *args, **kwargs):
        if 'config' in kwargs:
            self._config = kwargs['config']
            kwargs.pop('config')
        else:
            from aspyre import config
            self._config = config

        super().__init__(*args, **kwargs)

        config_group = self.add_argument_group('config')
        for section in self._config.sections():
            for k, v in getattr(self._config, section).items():
                config_group.add_argument(f'--config.{section}.{k}', default=v)

    def parse_args(self, *args, **kwargs):
        """
        A context manager that parses command line arguments,
        tweaks the Config object associated with this ArgumentParser within the 'with' block,
        and reverts it back to it's original values once the block exits.
        """
        args = super().parse_args(*args, *kwargs)
        return config_override(self._config, args)
