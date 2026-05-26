import logging
import os
import tempfile
import warnings
from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import matplotlib
import numpy as np
import pytest
from pytest import raises

import aspire
from aspire import __version__
from aspire.utils import (
    LogFilterByCount,
    all_pairs,
    all_triplets,
    check_pixel_size,
    get_full_version,
    mem_based_cpu_suggestion,
    num_procs_suggestion,
    physical_core_cpu_suggestion,
    powerset,
    rename_with_timestamp,
    utest_tolerance,
    virtual_core_cpu_suggestion,
)
from aspire.utils.misc import (
    bump_3d,
    fuzzy_mask,
    gaussian_1d,
    gaussian_2d,
    gaussian_3d,
    grid_3d,
)

logger = logging.getLogger(__name__)


def test_log_filter_by_count(caplog):
    msg = "A is for ASCII"

    # Should log.
    logger.info(msg)
    assert msg in caplog.text
    caplog.clear()

    with LogFilterByCount(logger, 1):
        # Should log.
        logger.info(msg)
        assert msg in caplog.text
        caplog.clear()

        # Should not log.
        # with caplog.at_level(logging.INFO):
        logger.info(msg)
        assert msg not in caplog.text
        caplog.clear()

    # Should log.
    logger.info(msg)

    with LogFilterByCount(logger, 1):
        logger.error(Exception("Should work with exceptions."))
        assert "Should work" in caplog.text
        caplog.clear()

        # Should not log (we've seen above twice).
        logger.info(msg)
        assert msg not in caplog.text
        caplog.clear()

    with LogFilterByCount(logger, 4):
        # Should log (we've seen above thrice).
        logger.info(msg)
        assert msg in caplog.text
        caplog.clear()


def test_get_full_version():
    """
    Test typical version string response is coherent with package.
    """
    assert get_full_version().startswith(__version__)


def test_get_full_version_path(monkeypatch):
    """
    Test when the directory doesn't exist, use the version of the package.
    """
    with monkeypatch.context() as m:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake")
            m.setattr(aspire, "__path__", [path])
            assert get_full_version() == __version__


def test_get_full_version_src(monkeypatch):
    """
    Test subprocess exception case of get_full_version.
    """
    with monkeypatch.context() as m:
        with tempfile.TemporaryDirectory() as tmp:
            m.setattr(aspire, "__path__", [tmp])
            assert get_full_version() == __version__ + ".src"


def test_get_full_version_unexpected(monkeypatch):
    """
    Test unexpected exception case of get_full_version.
    """
    with monkeypatch.context() as m:
        m.setattr("subprocess.check_output", lambda: RuntimeError)
        assert get_full_version() == __version__ + ".x"


def test_rename_with_timestamp(caplog):
    with tempfile.TemporaryDirectory() as tmpdir_name:
        filepath = os.path.join(tmpdir_name, "test_file.name")
        base, ext = os.path.splitext(filepath)

        # Create file on disk.
        with open(filepath, "w") as f:
            f.write("Test file")

        # Mock datetime to return a fixed timestamp.
        mock_datetime_value = datetime(2024, 10, 18, 12, 0, 0)
        mock_timestamp = mock_datetime_value.strftime("%y%m%d_%H%M%S")

        with mock.patch("aspire.utils.misc.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_datetime_value
            mock_datetime.strftime = datetime.strftime

            # Case 1: move=False should return the new file name with appended timestamp.
            renamed_file = rename_with_timestamp(filepath, move=False)
            assert renamed_file == f"{base}_{mock_timestamp}{ext}"

            # Case 2: move=True (default) should rename file on disk.
            with caplog.at_level(logging.INFO):
                renamed_file = rename_with_timestamp(filepath)

                # Check log for renaming operation.
                assert f"Renaming {filepath} as {renamed_file}" in caplog.text

                # Check that the original file no longer exists.
                assert not os.path.exists(filepath)

                # Check that the new file exists on disk with the expected name.
                assert os.path.exists(renamed_file)

        # Case 3: Test when the file does not exist.
        non_existent_file = os.path.join(tmpdir_name, "non_existent_file.name")
        with caplog.at_level(logging.WARNING):
            result = rename_with_timestamp(non_existent_file)

            # Check that None is returned since the file doesn't exist.
            assert result is None

            # Check log for the warning about file not found.
            assert (
                f"File '{non_existent_file}' not found, could not rename."
                in caplog.text
            )


def test_power_set():
    ref = sorted([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
    s = range(1, 4)
    assert sorted(list(powerset(s))) == ref


def test_get_test_tol():
    assert 1e-8 == utest_tolerance(np.float64)
    assert 1e-5 == utest_tolerance(np.float32)
    with raises(TypeError):
        utest_tolerance(int)


@pytest.mark.parametrize("indexing", ["yx", "xy"])
def test_gaussian_2d(indexing):
    L = 100
    # Note, `mu` and `sigma` are in (x, y) order.
    mu = (7, -3)
    sigma = (5, 6)

    g = gaussian_2d(L, mu=mu, sigma=sigma, indexing=indexing)

    # The normalized sum across an axis should correspond to a 1d gaussian with appropriate mu, sigma, peak.
    # Set axes based on 'indexing'.
    x, y = 0, 1
    if indexing == "yx":
        x, y = y, x

    g_x = np.sum(g, axis=y) / np.sum(g)
    g_y = np.sum(g, axis=x) / np.sum(g)

    # Corresponding 1d gaussians
    peak_x = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2)
    peak_y = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2)
    g_1d_x = peak_x * gaussian_1d(L, mu=mu[0], sigma=sigma[0])
    g_1d_y = peak_y * gaussian_1d(L, mu=mu[1], sigma=sigma[1])

    # Assert all-close
    assert np.allclose(g_x, g_1d_x)
    assert np.allclose(g_y, g_1d_y)

    # Test errors are raised with improper `mu` and `sigma` length.
    with raises(ValueError, match="`mu` must be len(2)*"):
        gaussian_2d(L, mu=(1,), sigma=sigma, indexing=indexing)
    with raises(ValueError, match="`sigma` must be*"):
        gaussian_2d(L, mu=mu, sigma=(1, 2, 3), indexing=indexing)


@pytest.mark.parametrize("indexing", ["zyx", "xyz"])
def test_gaussian_3d(indexing):
    L = 100
    # Note, `mu` and `sigma` are in (x, y, z) order.
    mu = (0, 5, 10)
    sigma = (5, 7, 9)

    G = gaussian_3d(L, mu, sigma, indexing=indexing)

    # The normalized sum across two axes should correspond to a 1d gaussian with appropriate mu, sigma, peak.
    # Set axes based on 'indexing'.
    x, y, z = 0, 1, 2
    if indexing == "zyx":
        x, y, z = z, y, x

    G_x = np.sum(G, axis=(y, z)) / np.sum(G)
    G_y = np.sum(G, axis=(x, z)) / np.sum(G)
    G_z = np.sum(G, axis=(x, y)) / np.sum(G)

    # Corresponding 1d gaussians
    peak_x = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2)
    peak_y = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2)
    peak_z = 1 / np.sqrt(2 * np.pi * sigma[2] ** 2)
    g_1d_x = peak_x * gaussian_1d(L, mu=mu[0], sigma=sigma[0])
    g_1d_y = peak_y * gaussian_1d(L, mu=mu[1], sigma=sigma[1])
    g_1d_z = peak_z * gaussian_1d(L, mu=mu[2], sigma=sigma[2])

    # Assert all-close
    assert np.allclose(G_x, g_1d_x)
    assert np.allclose(G_y, g_1d_y)
    assert np.allclose(G_z, g_1d_z)

    # Test errors are raised with improper `mu` and `sigma` length.
    with raises(ValueError, match="`mu` must be len(3)*"):
        gaussian_3d(L, mu=(1, 2), sigma=sigma, indexing=indexing)
    with raises(ValueError, match="`sigma` must be*"):
        gaussian_3d(L, mu=mu, sigma=(1, 2), indexing=indexing)


def test_all_pairs():
    n = 25
    pairs, pairs_to_linear = all_pairs(n, return_map=True)
    nchoose2 = n * (n - 1) // 2
    # Build all pairs using a loop to ensure numpy upper_triu() ordering matches.
    pairs_from_loop = [[i, j] for i in range(n - 1) for j in range(i + 1, n)]
    assert len(pairs) == nchoose2
    assert len(pairs[0]) == 2
    assert (pairs == pairs_from_loop).all()

    # Test the pairs_to_linear index mapping.
    assert (pairs_to_linear[pairs[:, 0], pairs[:, 1]] == np.arange(nchoose2)).all()


def test_all_triplets():
    n = 25
    triplets = all_triplets(n)
    nchoose3 = n * (n - 1) * (n - 2) // 6
    assert len(triplets) == nchoose3
    assert len(triplets[0]) == 3


def test_check_pixel_size():
    px_sizes_diff = np.array([1, 1.2, 1])
    px_sizes_same = np.array([1, 1, 1])
    px_sz = 1.0

    # Check not matching many to one
    with pytest.warns(UserWarning, match="does not match pixel_size"):
        assert not check_pixel_size(px_sizes_diff, px_sz)

    # Check matching many to one
    assert check_pixel_size(px_sizes_same, px_sz)

    # Check not matching one to one
    with pytest.warns(UserWarning, match="does not match pixel_size"):
        assert not check_pixel_size(px_sz + 1.2, px_sz)

    # Check matching one to one
    assert check_pixel_size(px_sz, px_sz)

    # Check non-scalar provided pixel_size
    with pytest.raises(ValueError, match="must be a scalar"):
        check_pixel_size(px_sizes_diff, px_sizes_same)


def test_gaussian_scalar_param():
    L = 100
    sigma = 5
    mu_2d = (2, 3)
    sigma_2d = (sigma, sigma)
    mu_3d = (2, 3, 5)
    sigma_3d = (sigma, sigma, sigma)

    g_2d = gaussian_2d(L, mu_2d, sigma_2d)
    g_2d_scalar = gaussian_2d(L, mu_2d, sigma)

    g_3d = gaussian_3d(L, mu_3d, sigma_3d)
    g_3d_scalar = gaussian_3d(L, mu_3d, sigma)

    np.testing.assert_allclose(g_2d, g_2d_scalar)
    np.testing.assert_allclose(g_3d, g_3d_scalar)


@pytest.mark.parametrize("L", [29, 30])
def test_bump_3d(L):
    L = L
    dtype = np.float64
    a = 10

    # Build volume of 1's and apply bump function
    volume = np.ones((L,) * 3, dtype=dtype)
    bump = bump_3d(L, spread=a, dtype=dtype)
    bumped_volume = np.multiply(bump, volume)

    # Define support for volume
    g = grid_3d(L, dtype=dtype)
    inside = g["r"] < (L - 1) / L
    outside = g["r"] >= 1

    # Test that volume is zero outside of support
    assert bumped_volume[outside].all() == 0

    # Test that volume is positive inside support
    assert (bumped_volume[inside] > 0).all()

    # Test that the center is still 1
    assert np.allclose(bumped_volume[(L // 2,) * 3], 1)


def test_fuzzy_mask():
    results = np.array(
        [
            [
                2.03406033e-06,
                7.83534653e-05,
                9.19567967e-04,
                3.73368194e-03,
                5.86559882e-03,
                3.73368194e-03,
                9.19567967e-04,
                7.83534653e-05,
            ],
            [
                7.83534653e-05,
                2.35760928e-03,
                2.15315317e-02,
                7.15226076e-02,
                1.03823087e-01,
                7.15226076e-02,
                2.15315317e-02,
                2.35760928e-03,
            ],
            [
                9.19567967e-04,
                2.15315317e-02,
                1.48272439e-01,
                3.83057355e-01,
                5.00000000e-01,
                3.83057355e-01,
                1.48272439e-01,
                2.15315317e-02,
            ],
            [
                3.73368194e-03,
                7.15226076e-02,
                3.83057355e-01,
                7.69781837e-01,
                8.96176913e-01,
                7.69781837e-01,
                3.83057355e-01,
                7.15226076e-02,
            ],
            [
                5.86559882e-03,
                1.03823087e-01,
                5.00000000e-01,
                8.96176913e-01,
                9.94134401e-01,
                8.96176913e-01,
                5.00000000e-01,
                1.03823087e-01,
            ],
            [
                3.73368194e-03,
                7.15226076e-02,
                3.83057355e-01,
                7.69781837e-01,
                8.96176913e-01,
                7.69781837e-01,
                3.83057355e-01,
                7.15226076e-02,
            ],
            [
                9.19567967e-04,
                2.15315317e-02,
                1.48272439e-01,
                3.83057355e-01,
                5.00000000e-01,
                3.83057355e-01,
                1.48272439e-01,
                2.15315317e-02,
            ],
            [
                7.83534653e-05,
                2.35760928e-03,
                2.15315317e-02,
                7.15226076e-02,
                1.03823087e-01,
                7.15226076e-02,
                2.15315317e-02,
                2.35760928e-03,
            ],
        ]
    )
    fmask = fuzzy_mask((8, 8), results.dtype, r0=2, risetime=2)
    np.testing.assert_allclose(results, fmask, atol=1e-7)

    # Smoke test for 1D, 2D, and 3D fuzzy_mask.
    for dim in range(1, 4):
        _ = fuzzy_mask((32,) * dim, np.float32)

    # Check that we raise an error for bad dimension.
    with pytest.raises(RuntimeError, match=r"Only 1D, 2D, or 3D fuzzy_mask*"):
        _ = fuzzy_mask((8,) * 4, np.float32)

    # Check we raise for bad 2D shape.
    with pytest.raises(ValueError, match=r"A 2D fuzzy_mask must be square*"):
        _ = fuzzy_mask((2, 3), np.float32)

    # Check we raise for bad 3D shape.
    with pytest.raises(ValueError, match=r"A 3D fuzzy_mask must be cubic*"):
        _ = fuzzy_mask((2, 3, 3), np.float32)


def test_multiprocessing_utils():
    """
    Smoke tests.
    """
    assert isinstance(mem_based_cpu_suggestion(), int)
    assert isinstance(physical_core_cpu_suggestion(), int)
    assert isinstance(virtual_core_cpu_suggestion(), int)
    assert isinstance(num_procs_suggestion(), int)


@contextmanager
def matplotlib_no_gui():
    """
    Context manager for disabling and restoring matplotlib plots, and
    ignoring associated warnings.
    """

    # Save current backend
    backend = matplotlib.get_backend()

    # Use non GUI backend.
    matplotlib.use("Agg")

    # Save and restore current warnings list.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Matplotlib is currently using agg.*")

        # Ignore the specific UserWarning about non-interactive FigureCanvasAgg
        warnings.filterwarnings(
            "ignore", r"FigureCanvasAgg is non-interactive, and thus cannot be shown"
        )

        yield

    # Explicitly close all figures before making backend changes.
    matplotlib.pyplot.close("all")

    # Restore backend
    matplotlib.use(backend)


def matplotlib_dry_run(func):
    """
    Decorator that wraps function in `matplotlib_no_gui` context.
    """

    def wrapper(*args, **kwargs):
        with matplotlib_no_gui():
            return func(*args, **kwargs)

    return wrapper
