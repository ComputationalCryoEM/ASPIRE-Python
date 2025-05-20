import pytest

from aspire.downloader import download_all


@pytest.mark.scheduled
def test_download_all(caplog):
    """Fail if a hash mismatch warning is logged during download."""
    caplog.clear()
    with caplog.at_level("WARNING"):
        _ = download_all()

    if "Hash mismatch" in caplog.text:
        pytest.fail(f"Hash mismatch warning was logged.\nCaptured logs:\n{caplog.text}")
