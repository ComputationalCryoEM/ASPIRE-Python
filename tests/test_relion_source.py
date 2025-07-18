import logging
import os

import numpy as np
import pytest

from aspire.source import RelionSource
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

    # Check we raise if pixel_size not provided and not found in metadata.
    with pytest.raises(ValueError, match=r".*No pixel size found in metadata.*"):
        src = RelionSource(starfile_no_pix_size)
