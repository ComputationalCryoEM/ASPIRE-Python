import logging
import os.path
from unittest import TestCase

import numpy as np
import pytest

from aspire.image import Image
from aspire.operators import (
    ArrayFilter,
    CTFFilter,
    FunctionFilter,
    IdentityFilter,
    PowerFilter,
    RadialCTFFilter,
    ScalarFilter,
    ScaledFilter,
    ZeroFilter,
)
from aspire.utils import gaussian_2d, grid_2d, utest_tolerance

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class SimTestCase(TestCase):
    test_filter = ArrayFilter(np.random.randn(8, 8))
    filter_eval_kwargs = dict()

    def setUp(self):
        self.dtype = np.float32
        # A 2 x 256 ndarray of spatial frequencies
        self.omega = np.load(os.path.join(DATA_DIR, "omega_2_256.npy"))

    def tearDown(self):
        pass

    def testFunctionFilter(self):
        filt = FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2))
        result = filt.evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(
            np.allclose(
                result[:5],
                [
                    5.17231862e-05,
                    1.64432545e-04,
                    4.48039823e-04,
                    1.04633750e-03,
                    2.09436945e-03,
                ],
            )
        )

    def testZeroFilter(self):
        result = ZeroFilter().evaluate(self.omega)
        # For all filters, we should get a 1d ndarray back on evaluate
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, np.zeros(256)))

    def testIdentityFilter(self):
        result = IdentityFilter().evaluate(self.omega, **self.filter_eval_kwargs)
        # For all filters, we should get a 1d ndarray back on evaluate
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, np.ones(256)))

    def testScalarFilter(self):
        result = ScalarFilter(value=1.5).evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, np.repeat(1.5, 256)))

    def testPowerFilter(self):
        filt = PowerFilter(
            filter=FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2)),
            power=0.5,
        )
        result = filt.evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(
            np.allclose(
                result[:5],
                np.array(
                    [
                        5.17231862e-05,
                        1.64432545e-04,
                        4.48039823e-04,
                        1.04633750e-03,
                        2.09436945e-03,
                    ]
                )
                ** 0.5,
            )
        )

    def testScaledFilter(self):
        scale_value = 2.5
        result1 = self.test_filter.evaluate(self.omega, **self.filter_eval_kwargs)

        filt2 = ScaledFilter(self.test_filter, scale_value)
        result2 = filt2.evaluate(self.omega * scale_value, **self.filter_eval_kwargs)
        self.assertTrue(np.allclose(result1, result2, atol=utest_tolerance(self.dtype)))

    def testDualFilter(self):
        result = self.test_filter.evaluate(-self.omega, **self.filter_eval_kwargs)
        dual_filter = self.test_filter.dual()
        dual_result = dual_filter.evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertTrue(np.allclose(result, dual_result))

    def testFilterSigns(self):
        signs = np.sign(
            self.test_filter.evaluate(self.omega, **self.filter_eval_kwargs)
        )
        sign_filter = self.test_filter.sign
        self.assertTrue(
            np.allclose(
                sign_filter.evaluate(self.omega, **self.filter_eval_kwargs), signs
            )
        )


class SimTestCaseCTFFilter(SimTestCase):
    """
    Covers same tests as SimTestCase, but use CTFFilter in place of ArrayFilter.
    """

    test_filter = CTFFilter()
    filter_eval_kwargs = dict(pixel_size=1)

    def testCTFFilter(self):
        filter = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result = filter.evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertEqual(result.shape, (256,))

    def testRadialCTFFilter(self):
        filter = RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertEqual(result.shape, (256,))

    def testCTFScale(self):
        filt = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result1 = filt.evaluate(self.omega, **self.filter_eval_kwargs)
        scale_value = 2.5
        filt = filt.scale(scale_value)
        # scaling a CTFFilter scales the pixel size which cancels out
        # a corresponding scaling in omega
        result2 = filt.evaluate(self.omega * scale_value, **self.filter_eval_kwargs)
        self.assertTrue(np.allclose(result1, result2, atol=utest_tolerance(self.dtype)))


DTYPES = [np.float32, np.float64]
EPS = [None, 0.01]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=EPS, ids=lambda x: f"epsilon={x}", scope="module")
def epsilon(request):
    return request.param


def test_power_filter_safeguard(dtype, epsilon, caplog):
    L = 25
    arr = np.ones((L, L), dtype=dtype)
    power = -0.5

    # Set a few values below default safeguard.
    num_eps = 3
    eps = epsilon
    if eps is None:
        eps = (100 * np.finfo(dtype).eps) ** (-1 / power)
    arr[L // 2, L // 2 : L // 2 + num_eps] = eps / 2

    # For negative powers, values below machine eps will be set to zero.
    filt = PowerFilter(
        filter=ArrayFilter(arr),
        power=power,
        epsilon=epsilon,
    )

    caplog.clear()
    caplog.set_level(logging.WARN)
    filt_vals = filt.evaluate_grid(L, dtype=dtype)

    # Check that extreme values are set to zero.
    ref = np.ones((L, L), dtype=dtype)
    ref[L // 2, L // 2 : L // 2 + num_eps] = 0

    np.testing.assert_array_equal(filt_vals, ref)

    # Check caplog for warning.
    msg = f"setting {num_eps} extremal filter value(s) to zero."
    assert msg in caplog.text


def test_array_filter_dtype_passthrough(dtype):
    """
    We upcast to use scipy's fast interpolator. We do not recast
    on exit, so this is an expected fail for singles.
    """
    if dtype == np.float32:
        pytest.xfail(reason="ArrayFilter currently upcasts singles.")

    L = 8
    arr = np.ones((L, L), dtype=dtype)

    filt = ArrayFilter(arr)
    filt_vals = filt.evaluate_grid(L, dtype=dtype)

    assert filt_vals.dtype == dtype


def test_ctf_reference():
    """
    Test CTFFilter against a MATLAB reference.
    """
    fltr = CTFFilter(
        voltage=200,
        defocus_u=10000,
        defocus_v=15000,
        defocus_ang=1.23,
        Cs=2.0,
        alpha=0.1,
    )
    h = fltr.evaluate_grid(5, pixel_size=4.56)

    # Compare with MATLAB.  Note DF converted to nm
    # >> n=5; V=200; DF1=1000; DF2=1500; theta=1.23; Cs=2.0; A=0.1; pxA=4.56;
    # >> ref_h=cryo_CTF_Relion(n,V,DF1,DF2,theta,Cs,pxA,A)
    #
    # Note we transpose the reference array.
    # Python keeps the filter C order because the images we will convolve with are C order.
    # MATLAB is F and F respectively.
    #
    # The floating point values were truncated to four decimal digits.
    ref_h = np.array(
        [
            [-0.6152, 0.0299, -0.5638, 0.9327, 0.9736],
            [-0.9865, 0.2598, -0.7543, 0.9383, 0.1733],
            [-0.1876, -0.9918, -0.1000, -0.9918, -0.1876],
            [0.1733, 0.9383, -0.7543, 0.2598, -0.9865],
            [0.9736, 0.9327, -0.5638, 0.0299, -0.6152],
        ]
    ).T

    # Test match all significant digits above
    np.testing.assert_allclose(h, ref_h, atol=5e-5)

def test_rectangular_ctf():
    """
    Compare a truncated rectangular CTF application with the
    application of CTF to a full square image.
    """
    # Configure square and truncated rectangle size
    L = 128
    rows, cols = 96, L
    assert rows <= L and cols <= L and min(rows, cols) < L

    # Create a test image of a disk
    # A = gaussian_2d(size=L, mu=(0,-L//10), sigma=L//8, dtype=np.float64)
    A = gaussian_2d(size=L, mu=(0, -L // 10), sigma=L // 32, dtype=np.float64)

    full_img = Image(A)
    truncated_img = Image(A[:rows, :cols])

    # Create a CTFFilter
    ctf_filter = CTFFilter(pixel_size=2)

    # Apply to both Image instances
    full_img_with_ctf = full_img.filter(ctf_filter)
    truncated_img_with_ctf = truncated_img.filter(ctf_filter)

    # Truncate the full square result
    full_img_with_ctf_truncated = full_img_with_ctf.asnumpy()[:, :rows, :cols]

    # Create mask for circular convolution effects
    mask = (grid_2d(L, normalized=True)["r"] < 0.5)[:rows, :cols]

    # Compare, we should be the same up to masked off differences in
    #   circular convolution wrap around.
    np.testing.assert_allclose(
        truncated_img_with_ctf * mask, full_img_with_ctf_truncated * mask, atol=1e-6
    )

