import itertools
import logging
import os

import numpy as np
import pytest

from aspire.source import RelionSource, Simulation
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
        - Both user provided and metadata pixel size
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
    src = RelionSource(starfile_no_pix_size, pixel_size=pix_size)
    np.testing.assert_equal(src.pixel_size, pix_size)

    # Check we raise if pixel_size not provided and not found in metadata.
    with pytest.raises(ValueError, match="No pixel size found in metadata"):
        src = RelionSource(starfile_no_pix_size)

    # Check we warn if both provided and mismatched.
    with pytest.warns(UserWarning, match="does not match pixel_size"):
        src = RelionSource(starfile_im_pix_size, pixel_size=1.234)  # 1.4 in metadata
        # Ensure we prefer user provided
        np.testing.assert_allclose(src.pixel_size, 1.234)


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


dtypes = {np.float32, np.float64}
dtype_triplets = list(itertools.product(dtypes, repeat=3))


@pytest.mark.parametrize("sim_dtype, offset_dtype, px_sz_dtype", dtype_triplets)
def test_offsets_roundtrip(sim_dtype, offset_dtype, px_sz_dtype, tmp_path):
    """
    Test that offsets remain unchanged after saving and loading.
    """
    L = 32
    n = 10
    offsets = (np.pi * np.ones((n, 2))).astype(offset_dtype)

    sim = Simulation(
        n=n,
        L=L,
        offsets=offsets,
        dtype=sim_dtype,
        pixel_size=np.array(np.e, dtype=px_sz_dtype),
    )

    # Check Simulation offsets
    assert sim.offsets.dtype == np.float64
    np.testing.assert_allclose(sim.offsets, offsets)

    # Save file and reload as RelionSource
    filepath = tmp_path / "sim_src.star"
    sim.save(filepath)

    rln_src_64 = RelionSource(filepath, dtype=np.float64)
    rln_src_32 = RelionSource(filepath, dtype=np.float32)

    # offsets should be doubles
    assert rln_src_64.offsets.dtype == np.float64
    assert rln_src_32.offsets.dtype == np.float64

    # Check roundtrip offsets are allclose up to expected resolution.
    atol = 0
    if px_sz_dtype == np.float32:
        atol = np.finfo(np.float32).resolution
    np.testing.assert_allclose(rln_src_64.offsets, sim.offsets, rtol=0, atol=atol)
    np.testing.assert_allclose(rln_src_32.offsets, sim.offsets, rtol=0, atol=atol)


def test_save_downsample(tmp_path):
    """
    Test that saving a downsampled RelionSource saves the correctly
    adjusted pixel_size and offsets.
    """
    # Starfile with pixel offsets. This starfile has _rlnDetectorPixelSize
    # and _rlnMagnification metadata fields that will be used to determine
    # the image pixel_size.
    starfile = os.path.join(DATA_DIR, "sample_particles_relion30.star")

    # Create RelionSource.
    src = RelionSource(starfile)

    # Check that detector metadata exists prior to downsample.
    assert src.has_metadata("_rlnDetectorPixelSize")
    assert src.has_metadata("_rlnMagnification")

    # Downsample and check that detector metadata is removed.
    L_ds = src.L // 2
    src_ds = src.downsample(L_ds)
    assert not src_ds.has_metadata("_rlnDetectorPixelSize")
    assert not src_ds.has_metadata("_rlnMagnification")

    # Save downsampled source and check that correct values are saved.
    save_path = tmp_path / "downsampled_data.star"
    src_ds.save(save_path)

    metadata = RelionStarFile(save_path).get_merged_data_block()
    saved_px_sz = metadata["_rlnImagePixelSize"][0]
    saved_offsets_angst = np.column_stack(
        (metadata["_rlnOriginXAngst"], metadata["_rlnOriginYAngst"])
    )

    np.testing.assert_allclose(saved_px_sz, src_ds.pixel_size)
    np.testing.assert_allclose(saved_offsets_angst / saved_px_sz, src_ds.offsets)
