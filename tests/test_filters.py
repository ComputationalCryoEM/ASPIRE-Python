import logging
import os.path
from unittest import TestCase

import numpy as np
import pytest

from aspire.downloader import emdb_2660
from aspire.operators import (
    ArrayFilter,
    CTFFilter,
    DualFilter,
    FunctionFilter,
    IdentityFilter,
    LambdaFilter,
    MultiplicativeFilter,
    PowerFilter,
    RadialCTFFilter,
    ScalarFilter,
    ScaledFilter,
    ZeroFilter,
)
from aspire.source import ArrayImageSource, Simulation
from aspire.utils import utest_tolerance

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")
SEED = 707


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
        # Scaling a CTFFilter is a no op; as of v15.0 scaling controlled by the pixel size.
        result2 = filt.evaluate(self.omega, **self.filter_eval_kwargs)
        self.assertTrue(np.allclose(result1, result2, atol=utest_tolerance(self.dtype)))

        # However, we can still test scaling pixel_size against scaling omega grid
        px_sz = self.filter_eval_kwargs["pixel_size"]
        # Scaling a CTFFilter pixel size should match a corresponding scaling in omega.
        result3 = filt.evaluate(
            self.omega / scale_value, pixel_size=px_sz
        )  # scale omega
        result4 = filt.evaluate(
            self.omega, pixel_size=px_sz * scale_value
        )  # scale pixel size
        self.assertTrue(np.allclose(result4, result3, atol=utest_tolerance(self.dtype)))


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


ALL_FILTER_TYPES = [
    ArrayFilter,
    CTFFilter,
    DualFilter,
    FunctionFilter,
    IdentityFilter,
    LambdaFilter,
    MultiplicativeFilter,
    PowerFilter,
    RadialCTFFilter,
    ScalarFilter,
    ScaledFilter,
    ZeroFilter,
]


@pytest.fixture(params=ALL_FILTER_TYPES, ids=lambda x: f"filter={x}", scope="module")
def filter_type(request):
    """
    Instantiate a basis filter of each type, handle filters that need args.
    """
    # instantiate a filter
    _f = request.param

    id_filter = IdentityFilter()

    if _f == ArrayFilter:
        f = _f(np.ones((8, 8)))
    elif _f == DualFilter:
        f = _f(id_filter)
    elif _f == FunctionFilter:
        f = _f(lambda x, y: np.exp(-(x**2 + y**2) / 2))
    elif _f == LambdaFilter:
        f = _f(id_filter, np.abs)
    elif _f == MultiplicativeFilter:
        f = _f(id_filter, id_filter)
    elif _f == PowerFilter:
        f = _f(id_filter, 0.5)
    elif _f == ScaledFilter:
        f = _f(id_filter, 2.0)
    else:
        f = _f()

    return f


def test_repr(filter_type):
    """Smoke test repr"""
    logger.debug(repr(filter_type))


def test_str(filter_type):
    """Smoke test str"""
    logger.debug(f"{filter_type})")


def test_len(filter_type):
    """Smoke test `len`"""
    n = len(filter_type)
    assert n == 1, "Length of filter should be 1"


def test_ctf_len():
    """
    Test filter stack `len` for CTFFilter
    """
    n = 3
    filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    assert len(filt) == n, f"Length of filter should be {n}"


def test_mul_len():
    """
    Test filter stack `len` for MultiplicativeFilter
    """

    n = 3
    id_filter = IdentityFilter()
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    assert len(ctf_filt) == n, f"Length of ctf_filter should be {n}"

    filt = MultiplicativeFilter(id_filter, ctf_filt)
    assert len(filt) == n, f"Length of MultiplicativeFilter should be {n}"


def test_pow_len():
    """
    Test filter stack `len` for PowerFilter
    """

    n = 3
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    assert len(ctf_filt) == n, f"Length of ctf_filter should be {n}"

    filt = PowerFilter(ctf_filt, 0.5)
    assert len(filt) == n, f"Length of MultiplicativeFilter should be {n}"


def test_scl_len():
    """
    Test filter stack `len` for ScaledFilter
    """

    n = 3
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    assert len(ctf_filt) == n, f"Length of ctf_filter should be {n}"

    filt = ScaledFilter(ctf_filt, 2.0)
    assert len(filt) == n, f"Length of ScaledFilter should be {n}"


def test_ctf_params(filter_type):
    """
    Test calling _ctf_params().

    Test raise when there are is not a CTFFilter subclass in the filter chain,
    and return parameter stack when CTFFilter.
    More complicated pass through for filter chains and multiplicative
    filters will be tested seperately.
    """

    if isinstance(filter_type, CTFFilter):
        params = filter_type._ctf_params()
        assert len(params) == len(filter_type)
    elif isinstance(filter_type, MultiplicativeFilter):
        # Multiplicative filters need to cycle through all possible underlying filters
        msg = "No CTF parameters found."
        with pytest.raises(RuntimeError, match=msg):
            _ = filter_type._ctf_params()
    else:
        msg = "_ctf_params not implemented for"
        with pytest.raises(NotImplementedError, match=msg):
            _ = filter_type._ctf_params()


def test_ctf_params_stack():
    """
    Test ctf_params for filter stack is passing through other filter (`ScaledFilter`).
    """
    n = 3
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))

    params = ctf_filt._ctf_params()
    assert len(params) == n

    # Scaling the CTFFilter should still return params from the underlying CTFFilter
    scaled_filt = ScaledFilter(ctf_filt, 2)
    _params = scaled_filt._ctf_params()
    np.testing.assert_allclose(_params, params)


def test_ctf_params_mult_stack():
    """
    Test ctf_params for filter stack is passing through `MultiplicativeFilter`.
    Additioinally tests that multiple CTF filter stacks in a chain raises error.
    """

    n = 3
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))

    params = ctf_filt._ctf_params()
    assert len(params) == n

    # MultiplicativeFilter should still return params from the underlying CTFFilter
    # Three filters are multiplied together here.
    scalar_filt = ScalarFilter(2)
    arr_filt = ArrayFilter(np.ones((8, 8)))
    mul_filt = MultiplicativeFilter(ctf_filt, scalar_filt, arr_filt)
    _params = mul_filt._ctf_params()
    np.testing.assert_allclose(_params, params)

    # Error on mulitple underlying CTFFilters
    lamb_filt = LambdaFilter(ctf_filt, np.sign)
    # both ctf_filt and lamb_filt resolve to a CTFFilter
    mul_filt = MultiplicativeFilter(ctf_filt, scalar_filt, lamb_filt)
    with pytest.raises(
        RuntimeError, match="Multiple filters with CTF parameters found"
    ):
        _params = mul_filt._ctf_params()


def test_mismatch_filter_lens():
    """
    MultiplicativeFilter should raise when two non singleton filters have differing stack lengths.
    """

    n = 3
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    rctf_filt = RadialCTFFilter(B=np.array([0, 0.1]))

    with pytest.raises(RuntimeError, match="Incoherent filter lengths"):
        _ = MultiplicativeFilter(ctf_filt, rctf_filt)


def test_ctf_eq():
    """
    Test CTFFilter equality.
    """
    n = 3
    ctf_filt = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    ctf_filt2 = CTFFilter(defocus_ang=np.linspace(0, 2 * np.pi, n))
    assert ctf_filt2 == ctf_filt, "CTFFilters should be equal"


def test_ctf_ineq():
    """
    Test CTFFilter inequality.
    """
    n = 3
    ctf_filt = CTFFilter(
        defocus_u=10000, defocus_v=15000, defocus_ang=np.linspace(0, 2 * np.pi, n)
    )
    ctf_filt2 = CTFFilter(
        defocus_u=12500, defocus_v=12500, defocus_ang=np.linspace(0, np.pi, n)
    )
    rctf_filt = ctf_filt.to_radial()
    assert ctf_filt != ctf_filt2, "Filters should not be equal"
    assert ctf_filt != rctf_filt, "Filters should not be equal"


def test_ctf_to_radial():
    """
    Test CTFFilter equality with RadialCTFFilter of equivalant parameters (up to angle).
    """

    n = 3
    angs = np.zeros(n)
    ctf_filt = CTFFilter(
        defocus_u=10000, defocus_v=np.linspace(15000, 20000, 3), defocus_ang=angs
    )
    # Manually average the defocus to make a radial filter.
    # Note this only works up to zero angles (RadialCTFFilter defaults to 0)
    avg_defocus = (ctf_filt.defocus_u + ctf_filt.defocus_v) / 2
    ctf_filt2 = CTFFilter(
        defocus_u=avg_defocus, defocus_v=avg_defocus, defocus_ang=angs
    )
    rctf_filt = ctf_filt.to_radial()

    # Averaging the defocus should yield equal CTF params
    assert ctf_filt2 == rctf_filt, "Filters should be equal"

    rctf_filt2 = rctf_filt.to_radial()  # should be a no-op
    assert rctf_filt2 == rctf_filt, "Filters should be equal"


def test_ctf_getitem():
    """
    Test various CTFFilters' __getitem__ method.
    """

    n = 3
    angs = np.linspace(0, 2 * np.pi, n)
    ctf_filt = CTFFilter(defocus_u=10000, defocus_v=15000, defocus_ang=angs)
    seq_ctf_filt = [
        CTFFilter(defocus_u=10000, defocus_v=15000, defocus_ang=ang) for ang in angs
    ]

    # Averaging the defocus should yield equal CTF params
    for i, filt in enumerate(ctf_filt):
        assert filt == ctf_filt[i], "Filters should be equal"
        assert ctf_filt[i] == seq_ctf_filt[i], "Filters should be equal"


def test_getitem():
    """
    Test __getitem__ on a non-trivial filter stack.
    """
    px = 1.34
    s = 2.0  # scale
    p = 0.5  # power
    n = 10  # number of filters in stack
    L = 32  # evaluate grid
    angs = np.linspace(0, 2 * np.pi, n)

    ctf_filter_stack = CTFFilter(defocus_u=10000, defocus_v=15000, defocus_ang=angs)
    filter_stack = ScaledFilter(ctf_filter_stack, s)
    more_filters = PowerFilter(ArrayFilter(np.random.random((L, L))), p)
    filter_stack = MultiplicativeFilter(filter_stack, more_filters)

    stack_eval = filter_stack.evaluate_grid(L, pixel_size=px)

    for i, flt in enumerate(filter_stack):
        # compute reference as singleton filters
        ref = MultiplicativeFilter(ScaledFilter(ctf_filter_stack[i], s), more_filters)
        ref_eval = ref.evaluate_grid(L, pixel_size=px)
        # compare singleton evaluation with __getitem__ from stack
        np.testing.assert_allclose(flt.evaluate_grid(L, pixel_size=px), ref_eval)
        np.testing.assert_allclose(stack_eval[i], ref_eval)


def test_batching_eval():
    """
    Test __getitem__ on a very large filter stack to induce the batching logic.
    """
    px = 1.34
    n = 3000  # number of filters in stack
    L = 65  # evaluate grid
    angs = np.linspace(0, 2 * np.pi, n)

    filter_stack = CTFFilter(defocus_u=10000, defocus_v=15000, defocus_ang=angs)
    # Reduce the max_size (required to switch into batching) so the test is faster
    filter_stack.max_size = n * L * L - 1

    stack_eval = filter_stack.evaluate_grid(L, pixel_size=px)

    for i in range(n):
        # compute reference as singleton filters
        ref_eval = filter_stack[i].evaluate_grid(L, pixel_size=px)
        # compare singleton evaluation with __getitem__ from stack
        np.testing.assert_allclose(stack_eval[i], ref_eval)


def testCTFdownsample():
    """
    Compare CTF Phaseflip -> Downsample vs Downsample -> Phaseflip
    """
    n = 10  # number of filters in stack
    K = 179  # simulation pixel downsampled

    angs = np.linspace(0, 2 * np.pi, n)
    filter_stack = [
        CTFFilter(defocus_u=10000, defocus_v=15000, defocus_ang=ang) for ang in angs
    ]

    vol = emdb_2660().astype(np.float64)
    sim = Simulation(
        n=n,
        vols=vol,
        offsets=0,
        amplitudes=1,
        unique_filters=filter_stack,
        filter_indices=np.arange(n),
        seed=SEED,
    )
    # Reduce possibility of simulation generation code interacting with the test.
    src = ArrayImageSource(sim.images[:])
    src.unique_filters = sim.unique_filters
    src.filter_indices = sim.filter_indices

    sim_pf_ds = src.phase_flip().downsample(K).images[:]
    sim_ds_pf = src.downsample(K).phase_flip().images[:]
    np.testing.assert_allclose(sim_ds_pf, sim_pf_ds)

    sim_dsc_pf = src.downsample(K).cache().phase_flip().images[:]
    np.testing.assert_allclose(sim_dsc_pf, sim_pf_ds)


def test_downsample_cache():
    """
    Compare Downsample Cache vs Downsample
    """
    n = 10  # number of filters in stack
    K = 179  # simulation pixel downsampled

    vol = emdb_2660().astype(np.float64)
    src = Simulation(
        n=n,
        vols=vol,
        offsets=0,
        amplitudes=1,
        seed=SEED,
    )

    sim_ds = src.downsample(K).images[:]
    sim_dsc = src.downsample(K).cache().images[:]

    np.testing.assert_allclose(sim_dsc, sim_ds)
