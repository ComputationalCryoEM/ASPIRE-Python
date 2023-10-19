import logging
import os

from docs import check_docstrings


def test_check_blank_line(caplog):
    test_string = os.path.join(
        os.path.dirname(__file__), "saved_test_data", "sample_docstrings.py"
    )

    caplog.clear()
    caplog.set_level(logging.ERROR)
    error_count = check_docstrings.check_blank_line_above_param_section(test_string)

    # Line numbers of good and bad docstrings in sample_docstrings.py
    good_doc_line_nums = [2, 16, 25, 35]
    bad_doc_line_nums = [43, 53, 65]

    # Check that good docstrings do not log error
    for line_num in good_doc_line_nums:
        msg = f"sample_docstrings.py: {line_num}: Must have exactly 1 blank line"
        assert msg not in caplog.text

    # Check that bad docstrings log error
    for line_num in bad_doc_line_nums:
        msg = f"sample_docstrings.py: {line_num}: Must have exactly 1 blank line"
        assert msg in caplog.text

    # Check total error count log
    assert error_count == len(bad_doc_line_nums)
