import logging

import numpy as np
import pytest

from aspire.abinitio import CLSymmetryC3C4, CLSymmetryCn, CLSync3N, CLSyncVoting
from aspire.source import OrientedSource, Simulation
from aspire.volume import CnSymmetricVolume

logger = logging.getLogger(__name__)


ESTIMATOR_SYMMETRY = [
    (CLSyncVoting, None),
    (CLSync3N, None),
    (CLSymmetryC3C4, "C4"),
    pytest.param((CLSymmetryCn, "C6"), marks=pytest.mark.expensive),
]


def src_fixture_id(params):
    estimator, symmetry = params
    return f"Orientation estimator: {estimator}, Symmetry: {symmetry}"


# Create a source fixture that provides an original source and an oriented source.
@pytest.fixture(params=ESTIMATOR_SYMMETRY, ids=src_fixture_id)
def src_fixture(request):
    estimator, symmetry = request.param
    L = 8
    n = 10
    vol = None

    # Symmetric Volume and additional kwargs for symmetric orientation estimation.
    estimator_kwargs = {}
    if symmetry:
        order = int(symmetry[1:])
        vol = CnSymmetricVolume(L=L, order=order, C=1).generate()
        estimator_kwargs.update({"n_theta": 36, "symmetry": symmetry})

    # Generate an origianl source and an oriented source.
    og_src = Simulation(L=L, n=n, vols=vol, offsets=0)
    orient_est = estimator(og_src, max_shift=1 / L, mask=False, **estimator_kwargs)
    oriented_src = OrientedSource(og_src, orient_est)

    return og_src, oriented_src


def test_repr(src_fixture):
    og_src, oriented_src = src_fixture

    # Check that original source is mentioned in repr
    logger.debug(f"repr(OrientedSrc): {repr(oriented_src)}")
    assert type(og_src).__name__ in repr(oriented_src)


def test_images(src_fixture):
    og_src, oriented_src = src_fixture
    assert np.allclose(og_src.images[:], oriented_src.images[:])


def test_rotations(src_fixture):
    _, oriented_src = src_fixture
    # Smoke test for rotations
    _ = oriented_src.rotations


def test_angles(src_fixture):
    _, oriented_src = src_fixture
    # Smoke test for angles
    _ = oriented_src.angles


def test_symmetry_group(src_fixture):
    og_src, oriented_src = src_fixture
    assert str(og_src.symmetry_group) == str(oriented_src.symmetry_group)


def test_default_estimator(src_fixture):
    og_src, _ = src_fixture

    # Instantiate OrientedSource without providing orientation estimator.
    oriented_src = OrientedSource(og_src)
    assert isinstance(oriented_src.orientation_estimator, CLSync3N)


def test_estimator_error(src_fixture):
    og_src, _ = src_fixture
    junk_estimator = 123
    with pytest.raises(
        ValueError,
        match="`orientation_estimator` should be subclass of `CLOrient3D`,* ",
    ):
        _ = OrientedSource(og_src, junk_estimator)


def test_lazy_evaluation(src_fixture, caplog):
    """
    Test that both images and rotations are evaluated in a lazy fashion.
    """
    _, oriented_src = src_fixture

    # Check that instantiated oriented source does not have rotation metadata.
    rotation_metadata = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
    assert not oriented_src.has_metadata(rotation_metadata)

    # Check that the oriented source's `orientation_estimator` does not have the attribute
    # `_pf`, an indicator that images have not yet been requested by the estimator.
    assert oriented_src.orientation_estimator._pf is None

    # Request rotations and check that metadata is populated.
    _ = oriented_src.rotations
    assert oriented_src.has_metadata(rotation_metadata)

    # Chec that `_pf` is not None.
    assert oriented_src.orientation_estimator._pf is not None

    # Check that requesting rotations again logs a debug message about skippin orientation.
    caplog.clear()
    msg = f"{oriented_src.__class__.__name__} already oriented, skipping"
    caplog.set_level(logging.DEBUG)
    assert msg not in caplog.text
    _ = oriented_src.rotations
    assert msg in caplog.text
