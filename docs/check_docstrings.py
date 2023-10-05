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

    docstrings = re.finditer(r"\"\"\"(.*?)\"\"\"", content, re.DOTALL)
    for docstring in docstrings:
        lines = docstring[0].split("\n")

        # Search for first occurence of either a ':param' or ':return' string.
        for i, line in enumerate(lines):
            if line.strip().startswith((r":param", r":return")):
                # If a body section exists but is not followed by exactly 1
                # new line log an error message and add to the count.
                body_section = "\n".join(lines[1:i])
                if not body_section:
                    break
                elif body_section != body_section.rstrip() + "\n":
                    # Get line number.
                    line_number = content.count("\n", 0, docstring.start()) + i

                    # Log error message.
                    msg = "Must have exactly 1 blank line between docstring body and parameter sections."
                    logger.error(f"{file_path}: {line_number}: {msg}")
                    error_count += 1
                break

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
