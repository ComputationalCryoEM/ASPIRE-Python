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

# We plan to move to importlib.resources, which importlib_resources is a
# backport of, once we have dropped support for Python 3.6. At that point
# the package resource management code in the tests will need to be
# rewritten with importlib.resources. This file will then be deleted.


def normalize_path(path):
    # type: (Any) -> str
    """Normalize a path by ensuring it is a string.
    If the resulting string contains path separators, an exception is raised.
    """
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
    """Return a file-like object opened for text reading of the resource."""
    return (importlib_resources.files(package) / normalize_path(resource)).open(
        "r", encoding=encoding, errors=errors
    )


def read_text(
    package: Package,
    resource: Resource,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    """Return the decoded string of the resource.
    The decoding-related arguments have the same semantics as those of
    bytes.decode().
    """
    with open_text(package, resource, encoding, errors) as fp:
        return fp.read()


def path(
    package: Package,
    resource: Resource,
) -> ContextManager[pathlib.Path]:
    """A context manager providing a file path object to the resource.
    If the resource does not already exist on its own on the file system,
    a temporary file will be created. If the file was created, the file
    will be deleted upon exiting the context manager (no exception is
    raised if the file was deleted prior to the context manager
    exiting).
    """
    return importlib_resources.as_file(
        importlib_resources.files(package) / normalize_path(resource)
    )
