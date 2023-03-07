import itertools
import logging

import numpy as np
import pytest

from aspire.image import (
    Image,
    MeanImageStacker,
    MedianImageStacker,
    SigmaRejectionImageStacker,
    WinsorizedImageStacker,
)

logger = logging.getLogger(__name__)

SAMPLE_N = 32
IMAGE_SIZES = [8, 9]
DTYPES = [np.float32, np.float64]


def simple_data_fixture_id(params):
    image_size = params[0]
    dtype = params[1]
    return f"image_size={image_size}, dtype={dtype.__name__}"


@pytest.fixture(
    params=itertools.product(IMAGE_SIZES, DTYPES), ids=simple_data_fixture_id
)
def simple_data_fixture(request):
    # unpack fixture params
    img_size, dtype = request.param

    # Generate test data.
    A = np.ones((SAMPLE_N, img_size, img_size), dtype=dtype)

    # Force outliers, single outlier per pixel across stack.
    for i in range(2, 7):
        A[0, i - 1] = i

    return Image(A)


def test_mean_stacker(simple_data_fixture):
    img_np = simple_data_fixture.asnumpy()
    ref_result = img_np.mean(axis=0)
    n = img_np.shape[0]

    stacker = MeanImageStacker()

    # Check stacker(Image(...)) == numpy
    stacked = stacker(simple_data_fixture)
    assert isinstance(stacked, Image)
    assert np.allclose(stacked.asnumpy(), ref_result)

    # Check stacker(ndarray) == numpy
    stacked = stacker(simple_data_fixture.asnumpy().reshape(n, -1))
    assert isinstance(stacked, np.ndarray)
    assert np.allclose(stacked, ref_result.reshape(1, -1))


def test_median_stacker(simple_data_fixture):
    stacker = MedianImageStacker()

    stacked = stacker(simple_data_fixture)

    # Median should ignore the single outliers
    assert np.allclose(stacked.asnumpy(), 1)


def test_sigma_stacker(simple_data_fixture):
    stacker = SigmaRejectionImageStacker(rejection_sigma=2)

    stacked = stacker(simple_data_fixture)

    # Sigma should ignore the single outliers
    assert np.allclose(stacked.asnumpy(), 1)


def test_sigma_fw_stacker(simple_data_fixture):
    stacker = SigmaRejectionImageStacker(rejection_sigma="FWHM")

    # Manipulate simple_data_fixture's outliers to be small.
    intensities = np.square(1 / simple_data_fixture.asnumpy().reshape(SAMPLE_N, -1))
    stacked = stacker(intensities)

    # Sigma FWHM should ignore the small outliers
    assert np.allclose(stacked, 1)


def test_windsor_stacker(simple_data_fixture):
    stacker = WinsorizedImageStacker(percentile=0.1)

    stacked = stacker(simple_data_fixture)

    # Winsorize should ignore the outliers
    assert np.allclose(stacked.asnumpy(), 1)
