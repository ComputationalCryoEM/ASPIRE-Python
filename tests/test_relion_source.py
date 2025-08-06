import logging
import os

import numpy as np
import pytest

from aspire.source import RelionSource
from aspire.utils import RelionStarFile
from aspire.volume import SymmetryGroup

from .test_starfile_stack import StarFileTestCase

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class StarFileMainCase(StarFileTestCase):
    def setUp(self):
        pass

    # This is a workaround to use a `pytest` fixture with `unittest` style cases.
    # We use it below to capture and inspect the log
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def testIncompletCTFWarning(self):
        with self._caplog.at_level(logging.WARN):
            # This call will instantiate RelionSource
            # During the starfile parsing we should log warning
            #  regarding incomplete CTF params.
            self.setUpStarFile("sample_incomplete_ctf_params.star")
            assert "partially populated CTF Params" in self._caplog.text


def test_symmetry_group(caplog):
    starfile_with_symmetry = os.path.join(
        DATA_DIR, "sample_relion_symmetry_group_D4.star"
    )
    starfile_without_symmetry = os.path.join(DATA_DIR, "sample_particles_relion30.star")

    # Check default symmetry_group.
    src = RelionSource(starfile_without_symmetry)
    assert isinstance(src.symmetry_group, SymmetryGroup)
    assert str(src.symmetry_group) == "C1"

    # Check symmetry_group attribute.
    src_sym = RelionSource(starfile_with_symmetry)
    assert isinstance(src_sym.symmetry_group, SymmetryGroup)
    assert str(src_sym.symmetry_group) == "D4"

    # Check overriding symmetry_group with RelionSource argument.
    caplog.clear()
    msg = "Overriding metadata with supplied symmetry group"
    caplog.set_level(logging.WARN)
    assert msg not in caplog.text
    src_override_sym = RelionSource(starfile_with_symmetry, symmetry_group="C6")
    assert msg in caplog.text

    assert isinstance(src_override_sym.symmetry_group, SymmetryGroup)
    assert str(src_override_sym.symmetry_group) == "C6"


def test_pixel_size(caplog):
    """
    Instantiate RelionSource from starfiles containing the following pixel size
    field variations:
        - "_rlnImagePixelSize"
        - "_rlnDetectorPixelSize" and "_rlnMagnification"
        - User provided pixel size
        - No pixel size provided
    and check src.pixel_size is correct.
    """
    starfile_im_pix_size = os.path.join(DATA_DIR, "sample_particles_relion31.star")
    starfile_detector_pix_size = os.path.join(
        DATA_DIR, "sample_particles_relion30.star"
    )
    starfile_no_pix_size = os.path.join(DATA_DIR, "rln_proj_64.star")

    # Check pixel size from _rlnImagePixelSize, set to 1.4000 in starfile.
    src = RelionSource(starfile_im_pix_size)
    np.testing.assert_equal(src.pixel_size, 1.4)

    # Check pixel size calculated from _rlnDetectorPixelSize and _rlnMagnification.
    src = RelionSource(starfile_detector_pix_size)
    det_pix_size = src.get_metadata(["_rlnDetectorPixelSize"])[0]
    mag = src.get_metadata("_rlnMagnification")[0]
    pix_size = 10000 * det_pix_size / mag
    np.testing.assert_equal(src.pixel_size, pix_size)

    # Check user provided pixel size.
    pix_size = 1.234
    src = RelionSource(starfile_im_pix_size, pixel_size=pix_size)
    np.testing.assert_equal(src.pixel_size, pix_size)

    # Check pixel size defaults to 1 if not provided.
    with caplog.at_level(logging.WARNING):
        msg = "No pixel size found in STAR file. Defaulting to 1.0 Angstrom"
        src = RelionSource(starfile_no_pix_size)
        assert msg in caplog.text
        np.testing.assert_equal(src.pixel_size, 1.0)


def test_offsets_conversion():
    """
    Check that offset convention gets converted to Relion >= 3.1 convention.
    """
    starfile = os.path.join(DATA_DIR, "sample_particles_relion30.star")

    # Extract pixel valued offsets from starfile prior to source instantiation.
    metadata = RelionStarFile(starfile).get_merged_data_block()
    pixel_offsets = np.column_stack((metadata["_rlnOriginX"], metadata["_rlnOriginY"]))

    # Create Relion Source and extract offsets from metadata using updated field names.
    src = RelionSource(starfile)
    angst_offsets = src.get_metadata(["_rlnOriginXAngst", "_rlnOriginYAngst"])

    # Check that old convention offset fields have been removed.
    assert "_rlnOriginX" not in src._metadata
    assert "_rlnOriginY" not in src._metadata

    # Check that offsets in metadata match up to pixel/angstrom conversion.
    np.testing.assert_allclose(angst_offsets / src.pixel_size, pixel_offsets)

    # src.offsets should still return pixel valued offsets.
    np.testing.assert_allclose(src.offsets, pixel_offsets)


def test_offsets():
    """
    Check that offsets are loaded properly with starfile field _rlnOriginX(Y)Angst.
    """
    # This starfile has offsets stored with angstrom values as _rlnOriginX(Y)Angst.
    starfile = os.path.join(DATA_DIR, "sample_particles_relion31.star")

    # Create a RelionSource
    src = RelionSource(starfile)

    # Check offsets are angstrom valued in metadata and correspond to src.offsets.
    angst_offsets = src.get_metadata(["_rlnOriginXAngst", "_rlnOriginYAngst"])
    np.testing.assert_allclose(src.offsets * src.pixel_size, angst_offsets)


def test_offsets_save(tmp_path):
    """
    Test that saving a RelionSource that was loaded with pixel offsets
    saves with angstrom valued offsets.
    """
    # Starfile with pixel offsets.
    starfile = os.path.join(DATA_DIR, "sample_particles_relion30.star")

    # Extract pixel valued offsets from starfile prior to source instantiation.
    metadata = RelionStarFile(starfile).get_merged_data_block()
    pixel_offsets = np.column_stack((metadata["_rlnOriginX"], metadata["_rlnOriginY"]))

    # Create and RelionSource and save to starfile.
    src = RelionSource(starfile)
    save_path = tmp_path / "test_file.star"
    src.save(save_path)

    # Saved starfile should have angstrom valued offsets.
    metadata = RelionStarFile(save_path).get_merged_data_block()
    angst_offsets = np.column_stack(
        (metadata["_rlnOriginXAngst"], metadata["_rlnOriginYAngst"])
    )

    # Check saved offsets match original up to pixel_size scaling.
    np.testing.assert_allclose(angst_offsets / src.pixel_size, pixel_offsets)
