import logging

import numpy as np
import pytest

from aspire.downloader import emdb_8012
from aspire.operators import CTFFilter
from aspire.source import Simulation
from aspire.utils import Rotation

logger = logging.getLogger(__name__)


@pytest.fixture
def sim_fixture():
    """
    Generate a very small simulation and slice it.
    """
    sim = Simulation(L=8, n=10, C=1, symmetry_group="D3")
    sim2 = sim[0::2]  # Slice the evens
    return sim, sim2


def test_remapping(sim_fixture):
    sim, sim2 = sim_fixture

    # Check images are served correctly, using internal index.
    np.testing.assert_allclose(
        sim.images[sim2.index_map].asnumpy(), sim2.images[:].asnumpy(), atol=1e-6
    )

    # Check images are served correctly, using known index (evens).
    index = list(range(0, sim.n, 2))
    np.testing.assert_allclose(
        sim.images[index].asnumpy(), sim2.images[:].asnumpy(), atol=1e-6
    )

    # Check meta is served correctly.
    assert np.all(sim.get_metadata(indices=sim2.index_map) == sim2.get_metadata())

    # Check symmetry_group pass-through.
    assert sim.symmetry_group == sim2.symmetry_group


def test_repr(sim_fixture):
    sim, sim2 = sim_fixture

    logger.debug(f"repr(IndexedSource): {repr(sim2)}")

    # Check `sim` is mentioned in the repr
    assert type(sim).__name__ in repr(sim2)

    # Check index counts are mentioned in the repr
    assert f"{sim2.n} of {sim.n}" in repr(sim2)


# @pytest.mark.expensive
def test_filter_mapping():
    """
    This test is designed to ensure that `unique_filters` and `filter_indices`
    are being remapped correctly upon slicing.

    Additionally it tests that a realistic preprocessing pipeline is equivalent.
    """

    # Generate N projection images,
    #   using N//2 rotations and
    #   N//2 ctf filters such that images[0::2] == images[1::2].
    N = 100
    SEED = 1234
    DT = np.float64
    DS = 129

    v = emdb_8012().astype(DT)

    # Generate N//2 rotations
    rots = Rotation.generate_random_rotations(N // 2, dtype=DT, seed=SEED)
    angles = Rotation(np.repeat(rots, 2, axis=0)).angles

    # Generate N//2 rotations and repeat indices
    defoci = np.linspace(1000, 25000, N // 2)
    ctf_filters = [
        CTFFilter(
            v.pixel_size,
            200,
            defocus_u=defoci[d],
            defocus_v=defoci[-d],
            defocus_ang=np.pi / (N // 2) * d,
            Cs=2.0,
            alpha=0.1,
        )
        for d in range(N // 2)
    ]
    ctf_indices = np.repeat(np.arange(N // 2), 2)

    # Construct the source
    src = Simulation(
        vols=v,
        n=N,
        dtype=DT,
        seed=SEED,
        unique_filters=ctf_filters,
        filter_indices=ctf_indices,
        angles=angles,
        offsets=0,
        amplitudes=1,
    ).cache()

    srcA = src[0::2]
    srcB = src[1::2]

    # Sanity check the images before proceeding
    np.testing.assert_allclose(srcA.images[:], src.images[::2])
    np.testing.assert_allclose(srcB.images[:], src.images[1::2])
    # Confirm the intention of the test
    np.testing.assert_allclose(srcB.images[:], srcA.images[:])

    # Preprocess the `src` stack
    pp = (
        src.phase_flip()
        .downsample(DS)
        .normalize_background()
        .legacy_whiten()
        .invert_contrast()
        .cache()
    )

    # Preprocess the indexed sources
    ppA = (
        srcA.phase_flip()
        .downsample(DS)
        .normalize_background()
        .legacy_whiten()
        .invert_contrast()
        .cache()
    )
    ppB = (
        srcB.phase_flip()
        .downsample(DS)
        .normalize_background()
        .legacy_whiten()
        .invert_contrast()
        .cache()
    )

    # Confirm we match the original images
    np.testing.assert_allclose(ppA.images[:], pp.images[::2])
    np.testing.assert_allclose(ppB.images[:], pp.images[1::2])
    # Confirm A and B are equivalent
    np.testing.assert_allclose(ppB.images[:], ppA.images[:])
