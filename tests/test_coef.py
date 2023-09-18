import numpy as np
import pytest

from aspire.basis import Coef, FFBBasis2D

IMG_SIZE = [
    32,
    pytest.param(31, marks=pytest.mark.expensive),
]
DTYPES = [
    np.float64,
    pytest.param(np.float32, marks=pytest.mark.expensive),
]
STACKS = [
    (),
    (1,),
    (2,),
    (3, 4),
]


def sim_fixture_id(params):
    stack, count, dtype = params
    return f"stack={stack}, count={count}, dtype={dtype}"


# Dtypes for coef array
@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


# Dtypes for basis
@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def basis_dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZE, ids=lambda x: f"count={x}")
def img_size(request):
    return request.param


@pytest.fixture(params=STACKS, ids=lambda x: f"stack={x}")
def stack(request):
    return request.param


ALLYOURBASES = [FFBBasis2D]


@pytest.fixture(params=ALLYOURBASES, ids=lambda x: f"basis={x}")
def basis(request, img_size, basis_dtype):
    cls = request.param
    return cls(img_size, dtype=basis_dtype)


@pytest.fixture
def coef_fixture(basis, stack, dtype):
    """
    Construct testing coefficient array.
    """
    # Combine the stack and coefficent counts into multidimensional
    # shape.
    size = stack + (basis.count,)

    coef_np = np.random.random(size=size).astype(dtype, copy=False)

    return Coef(basis, coef_np, dtype=dtype)


def test_coef_evalute(coef_fixture, basis):
    assert np.allclose(coef_fixture.evaluate(), basis.evaluate(coef_fixture))


def test_coef_rotate(coef_fixture, basis):
    # Rotations
    rots = np.linspace(-np.pi, np.pi, coef_fixture.stack_size).reshape(
        coef_fixture.stack_shape
    )

    # Refl
    refl = (
        np.random.rand(coef_fixture.stack_size).reshape(coef_fixture.stack_shape) > 0.5
    )  # Random bool

    assert np.allclose(coef_fixture.rotate(rots), basis.rotate(coef_fixture, rots))

    assert np.allclose(
        coef_fixture.rotate(rots, refl), basis.rotate(coef_fixture, rots, refl)
    )


def test_coef_shift(coef_fixture):
    pass
