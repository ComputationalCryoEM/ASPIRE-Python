import logging
import os
import tempfile

import numpy as np
import pytest
from PIL import Image as PILImage

from aspire.image import Image
from aspire.source import ArrayMicrographSource, DiskMicrographSource

from .test_utils import matplotlib_no_gui

logger = logging.getLogger(__name__)


# ====================
# Unit test parameters
# ====================


FILE_TYPES = [".mrc", ".tiff", ".tif"]
MICROGRAPH_COUNTS = [1, len(FILE_TYPES)]  # Sized to exercise all types
MICROGRAPH_SIZES = [101, 100]
DTYPES = [np.float32, np.float64]


# ========
# Fixtures
# ========


@pytest.fixture(params=DTYPES, ids=lambda x: str(x))
def dtype(request):
    return request.param


@pytest.fixture(params=MICROGRAPH_SIZES, ids=lambda x: f"Size {x}")
def micrograph_size(request):
    return request.param


@pytest.fixture(params=MICROGRAPH_COUNTS, ids=lambda x: f"Count {x}")
def micrograph_count(request):
    return request.param


@pytest.fixture(params=FILE_TYPES, ids=lambda x: f"File Type {x}")
def file_type(request):
    return request.param


@pytest.fixture()
def image_data_fixture(micrograph_count, micrograph_size, dtype):
    """
    This generates a Numpy array with prescribed shape and dtype.
    """
    img_np = np.random.rand(micrograph_count, micrograph_size, micrograph_size)
    return img_np.astype(dtype, copy=None)


# =====
# Tests
# =====


# Test MicrographSource vs Numpy Array


def test_array_backed_micrograph(image_data_fixture):
    """
    Test construction of MicrographSource initialized with a Numpy array.
    """

    mg_src = ArrayMicrographSource(image_data_fixture)

    np.testing.assert_allclose(mg_src.asnumpy(), image_data_fixture)


def test_2d_array_backed_micrograph(image_data_fixture):
    """
    Test construction of MicrographSource initialized with a Numpy array.
    """

    mg_src = ArrayMicrographSource(image_data_fixture[0])

    np.testing.assert_allclose(mg_src.asnumpy(), image_data_fixture[0:1])


def test_array_backed_micrograph_explicit_dtype(image_data_fixture):
    """
    Test construction of MicrographSource initialized with a Numpy array,
    with explicit dtype.
    """

    for dtype in DTYPES:
        mg_src = ArrayMicrographSource(image_data_fixture, dtype=dtype)

        # Check contents
        np.testing.assert_allclose(mg_src.asnumpy(), image_data_fixture)
        # Check types
        assert mg_src.dtype == dtype


# Test MicrographSource when loading from files.


def test_dir_backed_micrograph(image_data_fixture, file_type):
    """
    Test construction of MicrographSource initialized from directory.

    This test first saves each micrograph image directly,
    then reads them back using the MicrographSource interface.
    """

    # Note ordering is implicit to match MicrographSource glob.
    if file_type == ".mrc":
        imgs = [Image(img) for img in image_data_fixture]
    else:
        imgs = [PILImage.fromarray(img) for img in image_data_fixture]

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        for i, img in enumerate(imgs):
            fname = os.path.join(tmp_output_dir, f"{i}{file_type}")
            img.save(fname)
        mg_src = DiskMicrographSource(tmp_output_dir)

        # Ensure contents match
        np.testing.assert_allclose(mg_src.asnumpy(), image_data_fixture)

    # Note, the image formats are limited to single precision.
    if image_data_fixture.dtype != np.float64:
        # np.float64 image save is not implemented, but we can check float32 matches.
        assert mg_src.dtype == image_data_fixture.dtype, "Dtype mismatch"


# Test MicrographSource when loading from files.


def test_file_backed_micrograph(image_data_fixture):
    """
    Test construction of MicrographSource initialized from file list.

    This test first saves each micrograph image directly,
    then reads them back using the MicrographSource interface.
    """

    # Note ordering is explicit.
    # We'll exercise both a single file name `str` and a `list`.
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        if len(image_data_fixture) == 1:
            # String case
            fname = os.path.join(tmp_output_dir, "singleton.mrc")
            file_list = fname
            Image(image_data_fixture[0]).save(fname)

        else:
            # List cast
            file_list = []
            for i, img in enumerate(image_data_fixture):
                # Cover all the testable file types.
                file_type = FILE_TYPES[i % len(FILE_TYPES)]
                fname = os.path.join(tmp_output_dir, f"{i}{file_type}")
                if file_type == ".mrc":
                    Image(img).save(fname)
                else:
                    PILImage.fromarray(img).save(fname)

                file_list.append(fname)

        mg_src = DiskMicrographSource(file_list)

        # Ensure contents match
        np.testing.assert_allclose(mg_src.asnumpy(), image_data_fixture)

    # Note, the image formats are limited to single precision.
    if image_data_fixture.dtype != np.float64:
        # np.float64 image save is not implemented, but we can check float32 matches.
        assert mg_src.dtype == image_data_fixture.dtype, "Dtype mismatch"


def test_file_backed_micrograph_explicit_dtype(image_data_fixture):
    """
    Test construction of MicrographSource with explicit dtype.
    """

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        # Save the files, not these will all save as np.float32 at time of writing
        for i, img in enumerate(image_data_fixture):
            fname = os.path.join(tmp_output_dir, f"{i}.mrc")
            Image(img).save(fname)

        # Load with explicit dtype
        for dtype in DTYPES:
            mg_src = DiskMicrographSource(tmp_output_dir, dtype=dtype)
            # Check contents
            np.testing.assert_allclose(mg_src.asnumpy(), image_data_fixture)
            # Check types
            assert mg_src.dtype == dtype


# Test junk inputs


def test_none_micrograph_source():
    """
    Test empty MicrographSource raises.
    """
    for cls in [ArrayMicrographSource, DiskMicrographSource]:
        with pytest.raises(RuntimeError, match=r".*not implemented.*"):
            _ = cls(None)


def test_empty_micrograph_source():
    """
    Test empty MicrographSource raises.
    """
    for x in [[], ""]:
        with pytest.raises(RuntimeError, match=r"Must supply non-empty.*"):
            _ = DiskMicrographSource(x)


def test_no_files_micrograph_source():
    """
    Test MicrographSource initialized with empty path raises.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir should be created empty
        with pytest.raises(RuntimeError, match=r"No suitable files were found.*"):
            _ = DiskMicrographSource(tmp_dir)

        # create a non micrograph file, should still raise.
        open("not_a_micrograph.txt", "a").close()
        with pytest.raises(RuntimeError, match=r"No suitable files were found.*"):
            _ = DiskMicrographSource(tmp_dir)


def test_existential_crisis():
    """
    Test MicrographSource initialized with non existent path raises.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Fabricate a directory that does not exit.
        dne_dir = os.path.join(tmp_dir, "dne")
        with pytest.raises(
            RuntimeError, match=r".*does not appear to be a directory or a file.*"
        ):
            _ = DiskMicrographSource(dne_dir)


def test_bad_input_micrograph_source():
    """
    Test MicrographSource raises when instantiated with something weird.
    """
    with pytest.raises(NotImplementedError, match=r".*not implemented.*"):
        _ = ArrayMicrographSource(123)

    with pytest.raises(NotImplementedError, match=r".*not implemented.*"):
        _ = DiskMicrographSource(123)


def test_wrong_dim_micrograph_source():
    """
    Test MicrographSource raises with incorrect dim.
    """
    for shape in [(49), (1, 2, 7, 7)]:
        imgs_np = np.empty(shape)
        with pytest.raises(RuntimeError, match=r"Incompatible.*"):
            ArrayMicrographSource(imgs_np)


def test_rectangular_micrograph_source_array():
    """
    Test non-square micrograph source raises.
    """
    # Test with Numpy array input
    imgs_np = np.empty((3, 7, 8))
    with pytest.raises(RuntimeError, match=r"Incompatible.*"):
        ArrayMicrographSource(imgs_np)


def test_rectangular_micrograph_source_files():
    """
    Test non-square micrograph source raises.
    """

    # Test inconsistent mrc files
    imgs = [np.zeros((7, 7)), np.zeros((8, 8))]
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        # Save the files
        for i, img in enumerate(imgs):
            fname = os.path.join(tmp_output_dir, f"{i}.mrc")
            Image(img).save(fname)

        # Load them
        mg_src = DiskMicrographSource(tmp_output_dir)

        # Check the inconsistent shape raises when dynamically loading
        with pytest.raises(
            NotImplementedError, match=r"Micrograph.*has inconsistent shape.*"
        ):
            _ = mg_src.images[:]


# Test utilities


def test_show(image_data_fixture):
    """
    Test show doesn't crash.
    """
    mg_src = ArrayMicrographSource(image_data_fixture)
    with matplotlib_no_gui():
        mg_src.show()


def test_mrc_save(image_data_fixture):
    """
    Tests the base MicrographSource.save functionality.

    Specifically image_data_fixture -> save to disk -> load from files.
    """

    mg_src1 = ArrayMicrographSource(image_data_fixture)

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        path = os.path.join(tmp_output_dir, "test")

        # Write MRC files
        file_list = mg_src1.save(path)

        # Test we can load from dir `path`
        mg_src2 = DiskMicrographSource(path)
        np.testing.assert_allclose(mg_src2.asnumpy(), image_data_fixture)

        # Test we can load from `file_list`
        mg_src2 = DiskMicrographSource(file_list)
        np.testing.assert_allclose(mg_src2.asnumpy(), image_data_fixture)
