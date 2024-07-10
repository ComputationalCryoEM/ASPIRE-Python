import logging

import numpy as np
import pytest

from aspire.source import Simulation

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
    assert np.allclose(sim.images[sim2.index_map].asnumpy(), sim2.images[:].asnumpy())

    # Check images are served correctly, using known index (evens).
    index = list(range(0, sim.n, 2))
    assert np.allclose(sim.images[index].asnumpy(), sim2.images[:].asnumpy())

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
