import os
import pathlib
import types
from typing import Any, ContextManager, TextIO, Union

import importlib_resources

Package = Union[types.ModuleType, str]
Resource = str

# The purpose of this file is to suppress deprecation warnings arising
# from using deprecated legacy methods in the importlib_resources library.
# The methods below are still supported in importlib_resources._legacy
# but they raise warnings alerting that the new API should be used.

# Details on the deprecation can be found at:
# https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy

# We have copied the methods required as a temporary measure from:
# from https://github.com/python/importlib_resources/blob/main/importlib_resources/_legacy.py
# These are wrappers for code using the preferred files() API that can be
# dropped in to replace the deprecated methods

# We plan to move to importlib.resources, which importlib_resources is a
# backport of, once we have dropped support for Python 3.6. At that point
# the package resource management code in the tests will need to be
# rewritten with importlib.resources. This file will then be deleted.
# importlib.resources is in the standard library from 3.7 on:
# https://docs.python.org/3/library/importlib.html


def normalize_path(path):
    # type: (Any) -> str
    str_path = str(path)
    parent, file_name = os.path.split(str_path)
    if parent:
        raise ValueError(f"{path!r} must be only a file name")
    return file_name


def open_text(
    package: Package,
    resource: Resource,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> TextIO:
    return (importlib_resources.files(package) / normalize_path(resource)).open(
        "r", encoding=encoding, errors=errors
    )


def read_text(
    package: Package,
    resource: Resource,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    with open_text(package, resource, encoding, errors) as fp:
        return fp.read()


def path(
    package: Package,
    resource: Resource,
) -> ContextManager[pathlib.Path]:
    return importlib_resources.as_file(
        importlib_resources.files(package) / normalize_path(resource)
    )
