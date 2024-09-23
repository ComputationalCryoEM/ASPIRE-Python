import logging

import numpy as np
import pytest

from aspire.basis import DiracBasis2D
from aspire.image import Image

logger = logging.getLogger(__name__)


SIZES = [31, (32, 32)]
DTYPES = [np.float32, np.float64]
MASKS = [None, np.full((16, 16), True)]  # will be padded in `basis` fixture


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=SIZES, ids=lambda x: f"size={x}", scope="module")
def size(request):
    return request.param


@pytest.fixture(params=MASKS, ids=lambda x: f"mask={x}", scope="module")
def mask(request):
    return request.param


@pytest.fixture
def basis(size, dtype, mask):
    # get size, agnostic to 1d, 2d `size`
    s = np.atleast_1d(size)[0]

    if mask is not None:
        mask = np.pad(mask, s - mask.shape[-1])  # standard basis implicitly square
        mask = mask[:s, :s]  # crop to `size` (mainly for odd)

    basis = DiracBasis2D(size, mask=mask, dtype=dtype)

    return basis


def test_roundtrip(basis, mask):

    img = Image(np.random.random(basis.sz).astype(basis.dtype))

    coef = basis.evaluate_t(img)
    _img = basis.evaluate(coef)

    if mask is not None:
        # Mask case
        ref = img * basis.mask
        np.testing.assert_allclose(_img, ref)
        # Negated mask joined with outer values should all be zero
        np.all(img * ~basis.mask == 0)
    else:
        np.testing.assert_allclose(_img, img)

    _coef = basis.expand(_img)
    np.testing.assert_allclose(_coef, coef)
