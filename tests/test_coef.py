import pytest
import numpy as np
from aspire.basis import Coef, FFBBasis2D

COUNTS = [
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
    (3,4),
    ]
          

def sim_fixture_id(params):
    stack, count, dtype = params
    return f"stack={stack}, count={count}, dtype={dtype}"


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=COUNTS, ids=lambda x: f"count={x}")
def count(request):
    return request.param

@pytest.fixture(params=STACKS, ids=lambda x: f"stack={x}")
def stack(request):
    return request.param


@pytest.fixture
def coef_array_fixture(stack, count, dtype):
    """
    Construct testing coefficient array.
    """
    # Combine the stack and coefficent counts into multidimensional
    # shape.    
    size = stack + (count,)
    return np.random.random(size=size).astype(dtype, copy=False)

def test_coef_smoke(coef_array_fixture):
    basis = FFBBasis2D(123)
    c = Coef(basis, coef_array_fixture)
    
