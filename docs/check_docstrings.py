import logging
import os
import re
import sys

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

    docstrings = re.findall(r"\"\"\"(.*?)\"\"\"", content, re.DOTALL)

    for docstring in docstrings:
        lines = docstring.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(r":param") or line.strip().startswith(
                r":return"
            ):
                body_section = "\n".join(lines[:i])
                if not body_section:
                    break
                elif body_section != body_section.rstrip() + "\n":
                    # Get line number of error.
                    # Using `re.escape` to deal with non-alphanumeric characters.
                    match = re.search(re.escape(body_section), content)
                    docstring_start = match.start()
                    line_number = content.count("\n", 0, docstring_start) + i

                    # Log error message.
                    msg = "Must have exactly 1 blank line between docstring body and parameter sections."
                    logger.error(f"{file_path}: {line_number}: {msg}")
                    error_count += 1
                    break
                else:
                    break
    return error_count


def process_directory(directory):
    """
    Recursively walk through directories and check for docstring errors.
    If any errors found, log error count and exit.

    :param directory: Directory path to walk.
    """
    error_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                logger.info(f"Processing file: {file_path}")
                error_count += check_blank_line_above_param_section(file_path)
    if error_count > 0:
        logger.error(f"Found {error_count} docstring errors.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.warning("Usage: python check_docstrings.py <directory>")
        sys.exit(1)

    target_directory = sys.argv[1]
    process_directory(target_directory)
