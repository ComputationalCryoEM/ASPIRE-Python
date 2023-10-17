import logging
import os
import re
import sys
from glob import glob

logger = logging.getLogger(__name__)


def check_blank_line_above_param_section(file_path):
    """
    Check that every docstring with both a body section and a parameter
    section separates the two sections with exactly one blank line. Log
    errors and return count.

    :param file_path: File path to check for error.
    :return: Per file error count.
    """
    error_count = 0
    with open(file_path, "r") as file:
        content = file.read()

    regex = (
        r" {4,}\"\"\"\n(?:^[^:]+?[^\n])+(\n|\n\n\n+) {4,}(:p|:r)(?:.*\n)+? {4,}\"\"\""
    )

    bad_docstrings = re.finditer(regex, content, re.MULTILINE)
    for docstring in bad_docstrings:
        line_number = content.count("\n", 0, docstring.start()) + 1

        # Log error message.
        msg = "Must have exactly 1 blank line between docstring body and parameter sections."
        logger.error(f"{file_path}: {line_number}: {msg}")
        error_count += 1

    return error_count


def process_directory(directory):
    """
    Recursively walk through directories and check for docstring errors.
    If any errors found, log error count and exit.

    :param directory: Directory path to walk.
    """
    error_count = 0
    for file in glob(os.path.join(directory, "**/*.py"), recursive=True):
        error_count += check_blank_line_above_param_section(file)
    if error_count > 0:
        logger.error(f"Found {error_count} docstring errors.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.warning("Usage: python check_docstrings.py <directory>")
        sys.exit(1)

    target_directory = sys.argv[1]
    if not os.path.isdir(target_directory):
        raise RuntimeError(f"Invalid target directory path: {target_directory}")
    process_directory(target_directory)
