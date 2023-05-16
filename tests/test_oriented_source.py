import logging

import numpy as np
import pytest

from aspire.abinitio import CLSymmetryC3C4, CLSyncVoting
from aspire.source import OrientedSource, Simulation
from aspire.volume import CnSymmetricVolume

logger = logging.getLogger(__name__)


@pytest.fixture
def sim_fixture():
    L = 8
    n = 10
    sim = Simulation(L=L, n=n, C=1)
    vol_C4 = CnSymmetricVolume(L=L, order=4, C=1)
    sim_C4 = Simulation(L=L, n=n, vols=vol_C4.generate(), offsets=0)
    return sim, sim_C4


@pytest.fixture
def oriented_src_fixture(sim_fixture):
    # Original sources
    sim, sim_C4 = sim_fixture

    # Orientation estimators
    estimator = CLSyncVoting(sim)
    estimator_C4 = CLSymmetryC3C4(sim_C4, symmetry="C4", n_theta=72)

    # `OrientedSource`s
    src = OrientedSource(sim, orientation_estimator=estimator)
    src_C4 = OrientedSource(sim_C4, orientation_estimator=estimator_C4)

    return src, src_C4


def test_repr(oriented_src_fixture, sim_fixture):
    sim, sim_C4 = sim_fixture
    src, src_C4 = oriented_src_fixture

    # Check that original source is mentioned in repr
    logger.debug(f"repr(OrientedSrc): {repr(src)}")
    assert type(sim).__name__ in repr(src)

    logger.debug(f"repr(OrientedSrc): {repr(src_C4)}")
    assert type(sim_C4).__name__ in repr(src_C4)


def test_images(oriented_src_fixture, sim_fixture):
    src, src_C4 = oriented_src_fixture
    sim, sim_C4 = sim_fixture
    assert np.allclose(src.images[:], sim.images[:])
    assert np.allclose(src_C4.images[:], sim_C4.images[:])


def test_rotations(oriented_src_fixture):
    src, src_C4 = oriented_src_fixture
    # Smoke test for rotations
    _ = src.rotations
    _ = src_C4.rotations


def test_angles(oriented_src_fixture):
    src, src_C4 = oriented_src_fixture
    # Smoke test for angles
    _ = src.angles
    _ = src_C4.angles


def test_symmetry_group(oriented_src_fixture):
    src, src_C4 = oriented_src_fixture
    assert str(src.symmetry_group) == "C1"
    assert str(src_C4.symmetry_group) == "C4"


def test_default_estimator(sim_fixture):
    sim, _ = sim_fixture
    oriented_src = OrientedSource(sim)
    assert isinstance(oriented_src.orientation_estimator, CLSyncVoting)


def test_estimator_error(sim_fixture):
    sim, _ = sim_fixture
    junk_estimator = 123
    with pytest.raises(
        ValueError,
        match="`orientation_estimator` should be subclass of `CLOrient3D`,* ",
    ):
        _ = OrientedSource(sim, junk_estimator)


def test_lazy_orientation(oriented_src_fixture, caplog):
    for oriented_source in oriented_src_fixture:
        # Check that instantiated oriented sources don't have rotation metadata.
        rotation_metadata = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
        assert not oriented_source.has_metadata(rotation_metadata)

        # Request rotations and check that metadata is populated.
        _ = oriented_source.rotations
        assert oriented_source.has_metadata(rotation_metadata)

        # Check that requesting rotations again logs a debug message about skippin orientation.
        caplog.clear()
        msg = f"{oriented_source.__class__.__name__} already oriented, skipping"
        caplog.set_level(logging.DEBUG)
        assert msg not in caplog.text
        _ = oriented_source.rotations
        assert msg in caplog.text
