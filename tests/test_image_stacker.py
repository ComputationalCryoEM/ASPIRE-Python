import itertools
import logging

import numpy as np
import pytest

from aspire.image import (
    Image,
    MeanImageStacker,
    MedianImageStacker,
    SigmaRejectionImageStacker,
)

logger = logging.getLogger(__name__)

SAMPLE_N = 32
IMAGE_SIZES = [8, 9]
DTYPES = [np.float32, np.float64]


def xx_fixture_id(params):
    image_size = params[0]
    dtype = params[1]
    return f"image_size={image_size}, dtype={dtype.__name__}"


@pytest.fixture(params=itertools.product(IMAGE_SIZES, DTYPES), ids=xx_fixture_id)
def xx_fixture(request):
    # unpack fixture params
    img_size, dtype = request.param

    # Generate test data.
    A = np.ones((SAMPLE_N, img_size, img_size), dtype=dtype)

    # Force outliers
    for i in range(2, 7):
        A[0, i - 1] = i

    return Image(A)


def test_mean_stacker(xx_fixture):
    img_np = xx_fixture.asnumpy()
    ref_result = img_np.mean(axis=0)
    n = img_np.shape[0]

    stacker = MeanImageStacker()

    # Check stacker(Image(...)) == numpy
    stacked = stacker(xx_fixture)
    assert isinstance(stacked, Image)
    assert np.allclose(stacked.asnumpy(), ref_result)

    # Check stacker(ndarray) == numpy
    stacked = stacker(xx_fixture.asnumpy().reshape(n, -1))
    assert isinstance(stacked, np.ndarray)
    assert np.allclose(stacked, ref_result.reshape(1, -1))


def test_median_stacker(xx_fixture):
    stacker = MedianImageStacker()

    stacked = stacker(xx_fixture)

    # Median should ignore the single outliers
    assert np.allclose(stacked.asnumpy(), 1)


def test_sigma_stacker(xx_fixture):
    stacker = SigmaRejectionImageStacker()

    stacked = stacker(xx_fixture)

    # Sigma should ignore the single outliers
    assert np.allclose(stacked.asnumpy(), 1)
