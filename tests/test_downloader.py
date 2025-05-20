from pathlib import Path

import pytest

from aspire.downloader import download_all
from aspire.downloader.data_fetcher import _data_fetcher, fetch_data


@pytest.mark.scheduled
def test_download_all(caplog):
    """Fail if a hash mismatch warning is logged during download."""
    caplog.clear()
    with caplog.at_level("WARNING"):
        _ = download_all()

    if "Hash mismatch" in caplog.text:
        pytest.fail(f"Hash mismatch warning was logged.\nCaptured logs:\n{caplog.text}")


def test_fetch_data_warning(caplog):
    """Test that we get expected warning on hash mismatch."""
    # Use the smallest dataset in the registry
    dataset_name = "emdb_3645.map"

    # Remove file from cache if it exists
    cached_path = Path(_data_fetcher.path) / dataset_name
    if cached_path.exists():
        cached_path.unlink()

    # Save original hash from the registry
    original_hash = _data_fetcher.registry.get(dataset_name)
    assert original_hash is not None

    # Temporarily override the hash to simulate a mismatch
    _data_fetcher.registry[dataset_name] = "md5:invalidhash123"

    try:
        caplog.clear()
        with caplog.at_level("WARNING"):
            path = fetch_data(dataset_name)
        assert path  # Should return the path to the downloaded file
        assert f"Hash mismatch for {dataset_name}" in caplog.text
    finally:
        # Restore original hash
        _data_fetcher.registry[dataset_name] = original_hash
