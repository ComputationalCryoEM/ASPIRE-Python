import logging

import numpy as np
import pytest

from aspire.basis import DiracBasis2D, DiracBasis3D
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
