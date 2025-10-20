import numpy as np
import pytest

from aspire.basis import (
    Coef,
    FBBasis2D,
    FFBBasis2D,
    FLEBasis2D,
    FPSWFBasis2D,
    PSWFBasis2D,
)
from aspire.utils import utest_tolerance

IMG_SIZE = [
    31,
    32,
]
DTYPES = [
    np.float32,
    np.float64,
]
STACKS = [
    (),
    (1,),
    (2,),
    (3, 4),
]

ALLYOURBASES = [
    FBBasis2D,
    FFBBasis2D,
    PSWFBasis2D,
    FPSWFBasis2D,
    FLEBasis2D,
]


def sim_fixture_id(params):
    stack, count, dtype = params
    return f"stack={stack}, count={count}, dtype={dtype}"


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    """
    Dtypes for coef array
    """
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"basis_dtype={x}", scope="module")
def basis_dtype(request):
    """
    Dtypes for basis
    """
    return request.param


@pytest.fixture(params=IMG_SIZE, ids=lambda x: f"count={x}", scope="module")
def img_size(request):
    """
    Image size for basis.
    """
    return request.param


@pytest.fixture(params=STACKS, ids=lambda x: f"stack={x}", scope="module")
def stack(request):
    """
    Stack dimensions.
    """
    return request.param


@pytest.fixture(params=ALLYOURBASES, ids=lambda x: f"basis={x}", scope="module")
def basis(request, img_size, basis_dtype):
    """
    Parameterized `Basis` instantiation.
    """
    cls = request.param
    return cls(img_size, dtype=basis_dtype)


@pytest.fixture(scope="module")
def coef_fixture(basis, stack, dtype):
    """
    Construct parameterized testing coefficient array as `Coef`.
    """
    # Combine the stack and coeficent counts into multidimensional
    # shape.
    size = stack + (basis.count,)

    coef_np = np.random.random(size=size).astype(dtype, copy=False)
    pixel_size = 1.234

    return Coef(basis, coef_np, dtype=dtype, pixel_size=pixel_size)


@pytest.fixture(scope="module")
def rots(coef_fixture, dtype):
    # Rotations
    return np.linspace(-np.pi, np.pi, coef_fixture.stack_size).reshape(
        coef_fixture.stack_shape
    )


def test_mismatch_count(basis):
    """
    Confirm raises when instantiated with incorrect coef vector len.
    """
    # Derive an incorrect Coef
    x = np.empty(basis.count + 1, basis.dtype)
    with pytest.raises(RuntimeError, match=r".*does not match basis count.*"):
        _ = Coef(basis, x)


def test_incorrect_coef_type(basis):
    """
    Confirm raises when instantiated with incorrect coef type.
    """
    # Construct incorrect Coef type (list)
    x = list(range(basis.count + 1))
    with pytest.raises(ValueError, match=r".*should be instantiated with an.*"):
        _ = Coef(basis, x)


def test_0dim(basis):
    """
    Confirm raises when instantiated with 0dim scalar.
    """
    # Construct 0dim scalar object
    x = np.array(1)
    with pytest.raises(ValueError, match=r".*with shape.*"):
        _ = Coef(basis, x)


def test_not_a_basis():
    """
    Confirm raises when instantiated with something that is not a Basis.
    """
    # Derive an incorrect Coef
    x = np.empty(10)
    with pytest.raises(TypeError, match=r".*required to be a `Basis`.*"):
        _ = Coef(None, x)


def test_coef_key_dims(coef_fixture):
    """
    Test key lookup out of bounds dimension raises.
    """
    dim = coef_fixture.ndim
    # Construct a key with too many dims
    key = (0,) * (dim + 1)
    with pytest.raises(ValueError, match=r".*stack_dim is.*"):
        _ = coef_fixture[key]


def test_incorrect_reshape(basis):
    """
    Confirm raises when attempting incorrect stack reshape.
    """

    # create a multi dim coef array.
    x = np.empty((2, 3, 4, basis.count))
    c = Coef(basis, x)

    # Alter the stack shape, creating an incorrect shape.
    shp = list(c.stack_shape)
    shp[0] = shp[0] + 1

    with pytest.raises(ValueError, match=r".*cannot be reshaped to.*"):
        _ = c.stack_reshape(*shp)


def test_stack_reshape(basis):
    """
    Test stack_reshape matches corresponding pure Numpy reshape.
    """
    # create a multi dim coef array.
    x = np.empty((2, 3, 4, basis.count))
    c = Coef(basis, x)

    # Test -1 flatten
    ref = x.reshape(-1, basis.count)
    np.testing.assert_allclose(c.stack_reshape(-1).asnumpy(), ref)
    # Test 1d flatten
    np.testing.assert_allclose(c.stack_reshape(np.prod(x.shape[:-1])).asnumpy(), ref)
    # Test 2d reshape tuple (2,3,4) ~> ((6,4))
    ref = x.reshape(np.prod(x.shape[:-2]), x.shape[-2], basis.count)
    np.testing.assert_allclose(
        c.stack_reshape((np.prod(x.shape[:-2]), x.shape[-2])).asnumpy(), ref
    )
    # Test 2d reshape args (2,3,4) ~> (6,4)
    ref = x.reshape(np.prod(x.shape[:-2]), x.shape[-2], basis.count)
    np.testing.assert_allclose(
        c.stack_reshape(np.prod(x.shape[:-2]), x.shape[-2]).asnumpy(), ref
    )


def test_size(coef_fixture):
    """
    Confirm size matches.
    """
    np.testing.assert_equal(coef_fixture.size, coef_fixture.asnumpy().size)
    np.testing.assert_equal(coef_fixture.size, coef_fixture._data.size)


# Test basic arithmetic functions


def test_add(basis, coef_fixture):
    """
    Tests addition operation against pure Numpy.
    """
    # Make array
    x = np.random.random(size=coef_fixture.shape).astype(coef_fixture.dtype, copy=False)
    # Construct Coef
    c = Coef(basis, x)

    # Perform operation as array for reference
    ref = coef_fixture.asnumpy() + x

    # Perform operation as `Coef` for result
    res = coef_fixture + c

    # Compare result with reference
    np.testing.assert_allclose(res, ref)

    # Check pixel_size passthrough
    np.testing.assert_array_equal(coef_fixture.pixel_size, res.pixel_size)


def test_sub(basis, coef_fixture):
    """
    Tests subtraction operation against pure Numpy.
    """
    # Make array
    x = np.random.random(size=coef_fixture.shape).astype(coef_fixture.dtype, copy=False)
    # Construct Coef
    c = Coef(basis, x)

    # Perform operation as array for reference
    ref = coef_fixture.asnumpy() - x

    # Perform operation as `Coef` for result
    res = coef_fixture - c

    # Compare result with reference
    np.testing.assert_allclose(res, ref)

    # Check pixel_size passthrough
    np.testing.assert_array_equal(coef_fixture.pixel_size, res.pixel_size)


def test_neg(basis, coef_fixture):
    """
    Tests negation operation against pure Numpy.
    """
    # Perform operation as array for reference
    ref = -coef_fixture.asnumpy()

    # Perform operation as `Coef` for result
    res = -coef_fixture

    # Compare result with reference
    np.testing.assert_allclose(res, ref)

    # Check pixel_size passthrough
    np.testing.assert_array_equal(coef_fixture.pixel_size, res.pixel_size)


def test_mul(basis, coef_fixture):
    """
    Tests multiplication operation against pure Numpy.
    """
    # Make array
    x = np.random.random(size=coef_fixture.shape).astype(coef_fixture.dtype, copy=False)
    # Construct Coef
    c = Coef(basis, x)

    # Perform operation as array for reference
    ref = coef_fixture.asnumpy() * x

    # Perform operation as `Coef` for result
    res = coef_fixture * c

    # Compare result with reference
    np.testing.assert_allclose(res, ref)

    # Check pixel_size passthrough
    np.testing.assert_array_equal(coef_fixture.pixel_size, res.pixel_size)


# Test Passthrough Functions


def test_by_indices(coef_fixture, basis):
    """
    Test indice passthrough.
    """
    keys = [
        dict(),
        dict(angular=1),
        dict(radial=2),
        dict(angular=1, radial=2),
        dict(angular=basis.angular_indices > 0),
    ]

    for key in keys:
        np.testing.assert_allclose(
            coef_fixture.by_indices(**key),
            coef_fixture.asnumpy()[..., basis.indices_mask(**key)],
        )


def test_coef_evalute(coef_fixture, basis):
    """
    Test evaluate pass through.
    """
    np.testing.assert_allclose(
        coef_fixture.evaluate(),
        basis.evaluate(coef_fixture),
        rtol=1e-05,
        atol=utest_tolerance(basis.dtype),
    )


def test_coef_rotate(coef_fixture, basis, rots):
    """
    Test rotation pass through.
    """

    # Refl
    refl = (
        np.random.rand(coef_fixture.stack_size).reshape(coef_fixture.stack_shape) > 0.5
    )  # Random bool

    np.testing.assert_allclose(
        coef_fixture.rotate(rots), basis.rotate(coef_fixture, rots)
    )

    np.testing.assert_allclose(
        coef_fixture.rotate(rots, refl), basis.rotate(coef_fixture, rots, refl)
    )


# Test related Basis Coef checks.
# These are easier to test here via parameterization.
def test_evaluate_incorrect_type(coef_fixture, basis):
    """
    Test that evaluate raises when passed non Coef type.
    """
    with pytest.raises(TypeError, match=r".*should be passed a `Coef`.*"):
        # Pass something that is not a Coef, eg Numpy array.
        basis.evaluate(coef_fixture.asnumpy())


def test_to_real_incorrect_type(coef_fixture, basis):
    """
    Test to_real conversion raises on non `Coef` type.
    """
    # Convert Coef to complex, then to Numpy.
    x = basis.to_complex(coef_fixture).asnumpy()

    # Call to_real with Numpy array
    with pytest.raises(TypeError, match=r".*should be instance of `Coef`.*"):
        _ = basis.to_real(x)


def test_to_complex_incorrect_type(coef_fixture, basis):
    """
    Test to_complex conversion raises on non `Coef` type.
    """
    # Convert Coef to Numpy.
    x = coef_fixture.asnumpy()

    # Call to_complex with Numpy array
    with pytest.raises(TypeError, match=r".*should be instance of `Coef`.*"):
        _ = basis.to_complex(x)


def test_real_complex_real_roundtrip(coef_fixture, basis):
    rcoef = basis.to_real(basis.to_complex(coef_fixture))

    np.testing.assert_allclose(rcoef, coef_fixture, rtol=1e-05, atol=1e-08)


def test_complex_evaluate(coef_fixture):
    """
    Confirm using `ComplexCoef.evaluate` is equivalent to `Coef.evaluate`.
    """

    # Create a ComplexCoef
    complex_coef = coef_fixture.to_complex()

    # Compare
    np.testing.assert_allclose(
        complex_coef.evaluate(),
        coef_fixture.evaluate(),
        rtol=1e-05,
        atol=utest_tolerance(coef_fixture.basis.dtype),
    )


def test_complex_rotate(coef_fixture, rots):
    """
    Confirm using `ComplexCoef.rotate` is equivalent to `Coef.rotate`.
    """
    # Create a ComplexCoef
    complex_coef = coef_fixture.to_complex()

    # Compare
    np.testing.assert_allclose(
        complex_coef.rotate(rots),
        coef_fixture.rotate(rots).to_complex(),
        rtol=1e-05,
        atol=utest_tolerance(coef_fixture.basis.dtype),
    )


def test_shifts(coef_fixture, basis, rots):
    """
    Confirm using `Coef.shift` is equivalent to `basis.shift`.
    """
    if coef_fixture.stack_ndim > 1:
        pytest.xfail(reason="Shifts currently only support 1d stack axis.")

    # Create some shifts, by reusing the `rots` array.
    shifts = np.column_stack((rots, rots[::-1]))

    # Compare
    min_dtype = (
        np.float32
        if (basis.dtype == np.float32 or coef_fixture.dtype == np.float32)
        else np.float64
    )
    np.testing.assert_allclose(
        coef_fixture.shift(shifts),
        basis.shift(coef_fixture, shifts),
        rtol=1e-05,
        atol=utest_tolerance(min_dtype),
    )


def test_complex_shift(coef_fixture, rots):
    """
    Confirm using `ComplexCoef.shift` is equivalent to `Coef.shift`.
    """
    if coef_fixture.stack_ndim > 1:
        pytest.xfail(reason="Shifts currently only support 1d stack axis.")

    # Create a ComplexCoef
    complex_coef = coef_fixture.to_complex()

    # Create some shifts, by reusing the `rots` array.
    shifts = np.column_stack((rots, rots[::-1]))

    # Compare
    np.testing.assert_allclose(
        complex_coef.shift(shifts),
        coef_fixture.shift(shifts).to_complex(),
        rtol=1e-05,
        atol=utest_tolerance(coef_fixture.basis.dtype),
    )


def test_check_pixel_size(coef_fixture, basis):
    """
    Test all combinations of pixel sizes for the _check_pixel_size helper function.
    """
    # Make array
    x = np.random.random(size=coef_fixture.shape).astype(coef_fixture.dtype, copy=False)

    # Case: (None, None)
    A = Coef(basis, x)
    B = Coef(basis, x)
    assert Coef._check_pixel_size(A, B) is None

    # Case: (a, None)
    a = 1.2
    A = Coef(basis, x, pixel_size=a)
    B = Coef(basis, x)
    np.testing.assert_array_equal(Coef._check_pixel_size(A, B), a)

    # Case: (None, b)
    b = 2.3
    A = Coef(basis, x)
    B = Coef(basis, x, pixel_size=b)
    np.testing.assert_array_equal(Coef._check_pixel_size(A, B), b)

    # Case: (a, a)
    A = Coef(basis, x, pixel_size=a)
    B = Coef(basis, x, pixel_size=a)
    np.testing.assert_array_equal(Coef._check_pixel_size(A, B), a)

    # Case: (a, b)
    with pytest.warns(
        UserWarning, match=f"Pixel sizes do not match. Using pixel size {a}."
    ):
        A = Coef(basis, x, pixel_size=a)
        B = Coef(basis, x, pixel_size=b)
        np.testing.assert_array_equal(Coef._check_pixel_size(A, B), a)

    # Case: (a, np.array)
    A = Coef(basis, x, pixel_size=a)
    B = Coef(basis, x).asnumpy()
    np.testing.assert_array_equal(Coef._check_pixel_size(A, B), a)
