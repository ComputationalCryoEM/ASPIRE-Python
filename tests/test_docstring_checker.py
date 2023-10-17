import subprocess


def test_check_docstrings():
    result = subprocess.run(
        ["python", "docs/check_docstrings.py", "tests/saved_test_data"],
        capture_output=True,
        text=True,
    )

    good_doc_1 = "sample_docstrings.py: 2: Must have exactly 1 blank line"
    good_doc_2 = "sample_docstrings.py: 16: Must have exactly 1 blank line"
    err_1 = "sample_docstrings.py: 24: Must have exactly 1 blank line"
    err_2 = "sample_docstrings.py: 34: Must have exactly 1 blank line"

    # Check that good docstrings do not log error
    assert good_doc_1 not in result.stderr
    assert good_doc_2 not in result.stderr

    # Check that bad docstrings log error
    assert err_1 in result.stderr
    assert err_2 in result.stderr
