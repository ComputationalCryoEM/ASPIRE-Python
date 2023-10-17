import os
import subprocess


def test_check_docstrings():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")
    DOCS_CHECKER = os.path.join(
        os.path.dirname(__file__), "..", "docs", "check_docstrings.py"
    )

    result = subprocess.run(
        ["python", DOCS_CHECKER, DATA_DIR],
        capture_output=True,
        text=True,
    )

    good_doc_line_nums = [2, 16, 25]
    bad_doc_line_nums = [35, 45, 57]

    # Check that good docstrings do not log error
    for line_num in good_doc_line_nums:
        msg = f"sample_docstrings.py: {line_num}: Must have exactly 1 blank line"
        assert msg not in result.stderr

    # Check that bad docstrings log error
    for line_num in bad_doc_line_nums:
        msg = f"sample_docstrings.py: {line_num}: Must have exactly 1 blank line"
        assert msg in result.stderr

    # Check total error count log
    msg = f"Found {len(bad_doc_line_nums)} docstring errors"
    assert msg in result.stderr
