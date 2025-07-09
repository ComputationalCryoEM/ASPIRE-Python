import logging
import os
import tempfile
from heapq import heappush, heappushpop
from itertools import product, repeat

import numpy as np
import pytest

from aspire.basis import FBBasis2D, FFBBasis2D, FLEBasis2D, FPSWFBasis2D, PSWFBasis2D
from aspire.classification import (
    BandedSNRImageQualityFunction,
    BFRAverager2D,
    BumpWeightedVarianceImageQualityFunction,
    DistanceClassSelector,
    GlobalClassSelector,
    GlobalWithRepulsionClassSelector,
    NeighborVarianceClassSelector,
    NeighborVarianceWithRepulsionClassSelector,
    RampWeightedVarianceImageQualityFunction,
    RandomClassSelector,
    RIRClass2D,
    TopClassSelector,
    VarianceImageQualityFunction,
)
from aspire.classification.class_selection import _HeapItem
from aspire.denoising import (
    DebugClassAvgSource,
    DefaultClassAvgSource,
    LegacyClassAvgSource,
)
from aspire.image import Image
from aspire.source import RelionSource, Simulation
from aspire.utils import Rotation
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

# RNG SEED, should help small class average tests be deterministic.
SEED = 5552368

IMG_SIZES = [
    16,
    pytest.param(15, marks=pytest.mark.expensive),
]
DTYPES = [
    np.float64,
    pytest.param(np.float32, marks=pytest.mark.expensive),
]
CLS_SRCS = [
    DebugClassAvgSource,
    DefaultClassAvgSource,
    LegacyClassAvgSource,
]


BASIS = [
    FFBBasis2D,
    pytest.param(FBBasis2D, marks=pytest.mark.expensive),
    pytest.param(FLEBasis2D, marks=pytest.mark.expensive),
    pytest.param(PSWFBasis2D, marks=pytest.mark.expensive),
    pytest.param(FPSWFBasis2D, marks=pytest.mark.expensive),
]


@pytest.fixture(params=BASIS, ids=lambda x: f"basis={x}", scope="module")
def basis(request, img_size, dtype):
    cls = request.param
    # Setup a Basis
    basis = cls(img_size, dtype=dtype)
    return basis


def sim_fixture_id(params):
    res = params[0]
    dtype = params[1]
    return f"res={res}, dtype={dtype}"


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}", scope="module")
def img_size(request):
    return request.param


@pytest.fixture(scope="module")
def class_sim_fixture(dtype, img_size):
    """
    Construct a Simulation with explicit viewing angles forming
    synthetic classes.
    """

    # Configuration
    n_inplane_rots = 40

    # Platonic solids can generate our views.
    # Start with a cube, 8 vertices (use +-1 wlog),
    # each represents an viewing axis.
    cube_vertices = list(product(*repeat((-1, 1), 3)))
    inplane_rots = np.linspace(0, 2 * np.pi, n_inplane_rots, endpoint=False)
    # We want the first rotation to have angle 2pi instead of 0,
    # so the norm isn't degenerate (0) later.
    inplane_rots[0] = 2 * np.pi
    logger.debug(f"inplane_rots: {inplane_rots}")

    # Total rotations will be number of axis  * number of angles
    # ie. vertices * n_inplane_rots
    n = len(cube_vertices) * n_inplane_rots
    logger.debug(f"Constructing {n} rotations.")

    # Generate Rotations
    # Normalize the rotation axes to 1
    rotvecs = cube_vertices / np.linalg.norm(cube_vertices, axis=0)
    logger.debug(f"rotvecs: {rotvecs}")
    # renormalize by broadcasting with angle amounts in inplane_rots
    rotvecs = (rotvecs[np.newaxis].T * inplane_rots).T.reshape(n, 3)
    # Construct rotation object
    true_rots = Rotation.from_rotvec(rotvecs, dtype=dtype)

    # Load sample molecule volume
    # TODO, probably our default volume should "just work" for this stuff... tighter var?
    v = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")), dtype=dtype
    ).downsample(img_size)

    # Contruct the Simulation source.
    # Note using a single volume via C=1 is critical to matching
    # alignment without the complexity of remapping via states etc.
    src = Simulation(
        L=img_size,
        n=n,
        vols=v,
        offsets=0,
        amplitudes=1,
        C=1,
        angles=true_rots.angles,
        symmetry_group="C4",  # For testing symmetry_group pass-through.
    )
    # Prefetch all the images
    src = src.cache()

    return src


@pytest.fixture(
    params=CLS_SRCS,
    ids=lambda param: f"ClassSource={param.__class__.__name__}",
    scope="module",
)
def test_src_cls(request):
    return request.param


@pytest.fixture(scope="module")
def classifier(class_sim_fixture):
    return RIRClass2D(
        class_sim_fixture,
        fspca_components=63,
        bispectrum_components=51,  # Compressed Features after last PCA stage.
        n_nbor=10,
        sample_n=50000,
        large_pca_implementation="legacy",
        nn_implementation="legacy",
        bispectrum_implementation="legacy",
        seed=SEED,
    )


def test_basic_averaging(class_sim_fixture, test_src_cls, basis, classifier):
    """
    Test that the default `ClassAvgSource` implementations return
    class averages.
    """

    cmp_n = 5

    test_src = test_src_cls(src=class_sim_fixture, classifier=classifier)

    test_imgs = test_src.images[:cmp_n]

    # Fetch reference images from the original source.
    # We need remap the indices back to the original ids because
    # selectors will potentially reorder the classes.
    remapped_indices = test_src.selection_indices[list(range(cmp_n))]
    orig_imgs = class_sim_fixture.images[remapped_indices]

    # Sanity check
    assert np.allclose(
        np.linalg.norm((orig_imgs - test_imgs).asnumpy(), axis=(1, 2)), 0, atol=0.001
    )

    # Check we can slice the source and retrieve remapped attributes
    src2 = test_src[::3]
    # Check we match selection between automatic and manual slice.
    np.testing.assert_equal(src2.selection_indices, test_src.selection_indices[::3])
    # Check we match class indices between automatic and manual slice.
    k = len(src2.class_indices)
    np.testing.assert_equal(src2.class_indices, test_src.class_indices[::3][:k])

    # Check symmetry_group pass-through.
    assert test_src.symmetry_group == class_sim_fixture.symmetry_group


# Test the _HeapItem helper class
def test_heap_helper():
    dtype = np.dtype(np.float64)

    # Test the static method
    assert _HeapItem.nbytes(img_size=2, dtype=dtype) == 4 * dtype.itemsize + 16

    # Create an empty heap
    test_heap = []

    _img = Image(np.empty((2, 2), dtype=dtype))

    # Push item onto the heap
    a = _HeapItem(123, 0, _img)
    heappush(test_heap, a)

    # Push a better item onto heap, pop off the worst item.
    b = _HeapItem(456, 1, _img)
    popped = heappushpop(test_heap, b)

    assert popped == a, "Failed to pop min item"


@pytest.fixture(scope="module")
def cls_fixture(class_sim_fixture):
    """
    Classifier fixture.
    """
    # Create the classifier
    c2d = RIRClass2D(
        class_sim_fixture,
        fspca_components=63,
        bispectrum_components=51,  # Compressed Features after last PCA stage.
        n_nbor=10,
        sample_n=50000,
        nn_implementation="sklearn",
        seed=SEED,
    )
    # Compute the classification
    # (classes, reflections, distances)
    return c2d.classify()


# These are selectors that do not need to pass over all the global set
# of aligned and stacked class averages.
ONLINE_SELECTORS = [
    NeighborVarianceClassSelector,
    NeighborVarianceWithRepulsionClassSelector,
    DistanceClassSelector,
    RandomClassSelector,
    TopClassSelector,
]


@pytest.mark.parametrize(
    "selector", ONLINE_SELECTORS, ids=lambda param: f"Selector={param}"
)
def test_online_selector(cls_fixture, selector):
    # classes, reflections, distances = cls_fixture
    selection = selector().select(*cls_fixture)
    # Smoke test.
    logger.info(f"{selector}: {selection}")


# These are selectors which compute the entire global set of class
# averages before applying some criterion to select the "best"
# classes.  These are closer to the methods used historically in
# MATLAB experiments, sometimes called "out-of-core" in legacy code.
GLOBAL_SELECTORS = [
    GlobalClassSelector,
    GlobalWithRepulsionClassSelector,
]

QUALITY_FUNCTIONS = [
    BandedSNRImageQualityFunction,
    VarianceImageQualityFunction,
    BumpWeightedVarianceImageQualityFunction,
    RampWeightedVarianceImageQualityFunction,
]


@pytest.mark.parametrize(
    "selector", GLOBAL_SELECTORS, ids=lambda param: f"Selector={param}"
)
@pytest.mark.parametrize(
    "quality_function", QUALITY_FUNCTIONS, ids=lambda param: f"Quality Function={param}"
)
@pytest.mark.expensive
def test_global_selector(
    class_sim_fixture, cls_fixture, selector, quality_function, basis
):
    averager = BFRAverager2D(basis, class_sim_fixture)

    fun = quality_function()

    # Note: classes, reflections, distances = cls_fixture
    selection = selector(averager, fun).select(*cls_fixture)
    # smoke test
    logger.info(f"{selector}: {selection}")


# Try to put methods in the `DefaultClassAvgSource`s under continual
# test.  RIRClass2D, BFRAverager2D, and stacking are covered
# elsewhere, so that leaves manually testing contrast selection,
def test_contrast_selector(dtype):
    """
    Test selector is actually ranking by contrast.
    """

    n_classes = 5
    n_nbor = 32

    # Generate test data
    classes = np.arange(n_nbor * n_classes, dtype=int).reshape(n_nbor, n_classes).T
    reflections = np.random.rand(n_classes, n_nbor) > 0.5  # Random bool
    distances = np.random.rand(n_classes, n_nbor).astype(dtype)  # [0,1)

    # Compute reference manually.
    V = distances.var(axis=1)
    ref_class_ids = np.argsort(V)
    ref_scores = V[ref_class_ids]

    # Compute using class under test.
    selector = NeighborVarianceClassSelector()
    selection = selector.select(classes, reflections, distances)

    # Compare indices and scores.
    assert np.all(selection == ref_class_ids)
    assert np.allclose(selector._quality_scores, ref_scores)


def test_avg_src_starfileio(class_sim_fixture, test_src_cls, classifier):
    src = test_src_cls(src=class_sim_fixture, classifier=classifier)

    # Save and load the source as a STAR file.
    # Saving should force classification and selection to occur,
    #   and the attributes will be checked below.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.star")

        # Save
        src.save(path, overwrite=True)

        # Load
        saved_src = RelionSource(path)

    # Get entire metadata table
    a = src.get_metadata(as_dict=True)
    b = saved_src.get_metadata(as_dict=True)

    # Ensuring src has attributes following classification and selection.
    for attr in ("_class_indices", "_class_refl", "_class_distances"):
        assert attr in a.keys(), f"Attribute {attr} not in test Source."
        assert attr in b.keys(), f"Attribute {attr} not in Source read from disk."
        assert all(a[attr] == b[attr]), f"Attribute {attr} does not match."
