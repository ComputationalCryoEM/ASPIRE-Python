import subprocess


def test_check_docstrings():
    result = subprocess.run(
        ["python", "docs/check_docstrings.py", "tests/saved_test_data"],
        capture_output=True,
        text=True,
    )

    err_1 = "sample_docstrings.py: 22"
    err_2 = "sample_docstrings.py: 31"

    assert result.stdout == ""
    assert err_1 in result.stderr
    assert err_2 in result.stderr
