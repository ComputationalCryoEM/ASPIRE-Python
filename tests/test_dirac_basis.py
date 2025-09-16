import logging

import numpy as np
import pytest

from aspire.basis import DiracBasis2D, DiracBasis3D
from aspire.numeric import xp
from aspire.reconstruction import MeanEstimator
from aspire.source import Simulation
from aspire.utils import grid_2d, grid_3d

logger = logging.getLogger(__name__)


SIZES = [31, 32]
DTYPES = [np.float32, np.float64]
MASKS = [None, 16]  # will be created in `mask` fixture
DIMS = [2, 3]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=SIZES, ids=lambda x: f"size={x}", scope="module")
def size(request):
    return request.param


@pytest.fixture(params=DIMS, ids=lambda x: f"dim={x}", scope="module")
def dim(request):
    return request.param


@pytest.fixture(params=MASKS, ids=lambda x: f"mask={x}", scope="module")
def mask(request, size, dim):
    mask = request.param
    # When provided a mask radius ...
    if mask is not None:
        if dim == 2:
            grid = grid_2d
        elif dim == 3:
            grid = grid_3d

        # ... compute mask of `size` < radius
        mask = grid(size, normalized=False)["r"] < mask

    return mask


@pytest.fixture
def basis(size, dtype, mask, dim):
    # get size, agnostic to 1d, 2d `size`
    s = np.atleast_1d(size)[0]

    if mask is not None:
        mask = np.pad(mask, s - mask.shape[-1])  # standard basis implicitly square
        mask = mask[:s, :s]  # crop to `size` (mainly for odd)

    if dim == 2:
        dirac_basis = DiracBasis2D
    elif dim == 3:
        dirac_basis = DiracBasis3D

    basis = dirac_basis(size, mask=mask, dtype=dtype)

    return basis


def test_roundtrip(basis, mask):

    # basis._cls is `Image` or `Volume`
    x = basis._cls(np.random.random(basis.sz).astype(basis.dtype))

    coef = basis.evaluate_t(x)
    _x = basis.evaluate(coef)

    if mask is not None:
        # Mask case
        ref = x * mask
        np.testing.assert_allclose(_x, ref)
        # Negated mask joined with outer values should all be zero
        assert np.all((ref * ~mask).asnumpy() == 0)
    else:
        np.testing.assert_allclose(_x, x)

    _coef = basis.expand(_x)
    np.testing.assert_allclose(_coef, coef)


def test_dirac_mean_vol_est(size, dtype):
    """
    Test the DiracBasis3D passes through MeanEstimator.
    """

    basis = DiracBasis3D(size, dtype=dtype)

    src = Simulation(
        n=300,
        L=size,
        C=1,
        dtype=dtype,
        offsets=0,
        amplitudes=1,
    )

    est_vol = MeanEstimator(src, basis=basis).estimate()

    np.testing.assert_array_less(np.mean(src.vols - est_vol), 1e-5)


def test_dirac_basis_ndarray_private_roundtrip(basis, mask):
    """
    Tests private `evaluate_t` and `evaluate` methods are returning intended types, shapes, and values.
    """
    stack_len = 7
    # Form a numpy test array stack of `stack_len` items
    x = ref = np.random.random((stack_len, *basis.sz)).astype(basis.dtype)

    # convert to an array of coeficients
    c = basis._evaluate_t(x)
    assert c.shape == (stack_len, basis.count), "Incorrect shape"
    assert isinstance(c, np.ndarray), "Incorrect return object type"

    # convert back to array in original domain
    y = basis._evaluate(c)
    if mask is not None:
        # Mask case
        ref = ref * mask
        # Negated mask joined with outer values should all be zero
        assert np.all((ref * ~mask) == 0)

    np.testing.assert_equal(
        y, ref, err_msg="Arrays should be identical up to any `mask`."
    )


def test_dirac_basis_xparray_passthrough(basis, mask):
    """
    Tests private `evaluate_t` and `evaluate` methods are returning intended types, shapes, and values.
    """
    # If we can import cupy then `xp` could be cupy array depending on config.
    pytest.importorskip("cupy")

    stack_len = 7
    # Form a numpy test array stack of `stack_len` items
    x = ref = xp.random.random((stack_len, *basis.sz)).astype(basis.dtype)

    # convert to an array of coeficients
    c = basis._evaluate_t(x)
    assert c.shape == (stack_len, basis.count), "Incorrect shape"
    assert isinstance(c, xp.ndarray), "Incorrect return object type"

    # convert back to array in original domain
    y = basis._evaluate(c)
    if mask is not None:
        # Mask case
        # Note `DiracBasis` classes internally handle their own mask conversion as needed.
        _mask = xp.asarray(mask)
        ref = ref * _mask
        # Negated mask joined with outer values should all be zero
        assert xp.all((ref * ~_mask) == 0)

    np.testing.assert_equal(
        xp.asnumpy(y),
        xp.asnumpy(ref),
        err_msg="Arrays should be identical up to any `mask`.",
    )
