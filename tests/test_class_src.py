import logging
import os
from itertools import product, repeat

import numpy as np
import pytest

from aspire.basis import FFBBasis2D
from aspire.classification import (
    BandedSNRImageQualityFunction,
    BFRAverager2D,
    ContrastClassSelector,
    ContrastWithRepulsionClassSelector,
    DistanceClassSelector,
    GlobalClassSelector,
    GlobalWithRepulsionClassSelector,
    RandomClassSelector,
    RIRClass2D,
    TopClassSelector,
)
from aspire.denoising import ClassicClassAvgSource, DebugClassAvgSource
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


RESOLUTIONS = [32]
DTYPES = [np.float64]
CLS_SRCS = [DebugClassAvgSource, ClassicClassAvgSource]


def sim_fixture_id(params):
    res = params[0]
    dtype = params[1]
    return f"res={res}, dtype={dtype.__name__}"


@pytest.fixture(params=product(RESOLUTIONS, DTYPES), ids=sim_fixture_id)
def class_sim_fixture(request):
    """
    Construct a Simulation with explicit viewing angles forming
    synthetic classes.
    """

    # Configuration
    res, dtype = request.param
    n_inplane_rots = 40

    # Platonic solids can generate our views.
    # Start with a cube, 8 vertices (use +-1 wlog),
    # each represents an viewing axis.
    cube_vertices = list(product(*repeat((-1, 1), 3)))
    inplane_rots = np.linspace(0, 2 * np.pi, n_inplane_rots, endpoint=False)
    # We want the first rotation to have angle 2pi instead of 0,
    # so the norm isn't degenerate (0) later.
    inplane_rots[0] = 2 * np.pi
    logger.info(f"inplane_rots: {inplane_rots}")

    # Total rotations will be number of axis  * number of angles
    # ie. vertices * n_inplane_rots
    n = len(cube_vertices) * n_inplane_rots
    logger.info(f"Constructing {n} rotations.")

    # Generate Rotations
    # Normalize the rotation axes to 1
    rotvecs = cube_vertices / np.linalg.norm(cube_vertices, axis=0)
    logger.info(f"rotvecs: {rotvecs}")
    # renormalize by broadcasting with angle amounts in inplane_rots
    rotvecs = (rotvecs[np.newaxis].T * inplane_rots).T.reshape(n, 3)
    # Construct rotation object
    true_rots = Rotation.from_rotvec(rotvecs, dtype=dtype)

    # Load sample molecule volume
    # TODO, probably our default volume should work for this stuff... tighter var?
    v = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")), dtype=dtype
    ).downsample(res)

    # Contruct the Simulation source
    src = Simulation(
        L=res, n=n, vols=v, offsets=0, amplitudes=1, C=1, angles=true_rots.angles
    )
    # Prefetch all the images
    src.cache()

    return src


@pytest.mark.parametrize(
    "test_src_cls", CLS_SRCS, ids=lambda param: f"ClassSource={param}"
)
def test_run(class_sim_fixture, test_src_cls):
    # Images from the original Source
    orig_imgs = class_sim_fixture.images[:5]

    # Classify, Select, and average images using test_src_cls
    test_src = test_src_cls(classification_src=class_sim_fixture)
    test_imgs = test_src.images[:5]

    # Sanity check
    assert np.allclose(
        np.linalg.norm((orig_imgs - test_imgs).asnumpy(), axis=(1, 2)), 0, atol=0.001
    )


@pytest.fixture()
def cls_fixture(class_sim_fixture):
    """
    Classifier fixture.
    """
    # Create the classifier
    c2d = RIRClass2D(class_sim_fixture, nn_implementation="sklearn")
    # Compute the classification
    # (classes, reflections, distances)
    return c2d.classify()


LOCAL_SELECTORS = [
    ContrastClassSelector,
    ContrastWithRepulsionClassSelector,
    DistanceClassSelector,
    RandomClassSelector,
    TopClassSelector,
]


@pytest.mark.parametrize(
    "selector", LOCAL_SELECTORS, ids=lambda param: f"Selector={param}"
)
def test_custom_local_selector(cls_fixture, selector):
    # classes, reflections, distances = cls_fixture
    selection = selector().select(*cls_fixture)
    logger.info(f"{selector}: {selection}")


GLOBAL_SELECTORS = [
    GlobalClassSelector,
    GlobalWithRepulsionClassSelector,
]


@pytest.mark.parametrize(
    "selector", GLOBAL_SELECTORS, ids=lambda param: f"Selector={param}"
)
def test_custom_global_selector(class_sim_fixture, cls_fixture, selector):
    basis = FFBBasis2D(class_sim_fixture.L, dtype=class_sim_fixture.dtype)

    averager = BFRAverager2D(basis, class_sim_fixture, num_procs=1)

    fun = BandedSNRImageQualityFunction()

    # classes, reflections, distances = cls_fixture
    selection = selector(averager, fun).select(*cls_fixture)
    logger.info(f"{selector}: {selection}")
