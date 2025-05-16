import pytest

from aspire.downloader import download_all


@pytest.mark.scheduled
def test_download_all():
    """This test will throw a warning if any hashes have changed"""
    _ = download_all()
