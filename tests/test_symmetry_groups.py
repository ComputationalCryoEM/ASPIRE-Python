import itertools
import logging

import numpy as np
import pytest

from aspire.utils import Rotation
from aspire.volume import (
    CnSymmetryGroup,
    DnSymmetryGroup,
    OctahedralSymmetryGroup,
    SymmetryGroup,
    TSymmetryGroup,
)

logger = logging.getLogger(__name__)

GROUPS_WITH_ORDER = [
    CnSymmetryGroup,
    DnSymmetryGroup,
]
GROUPS_WITHOUT_ORDER = [
    TSymmetryGroup,
    OctahedralSymmetryGroup,
]
ORDERS = [2, 3, 4, 5]
DTYPES = [np.float32, np.float64]
PARAMS_ORDER = list(itertools.product(GROUPS_WITH_ORDER, DTYPES, ORDERS))
PARAMS = list(itertools.product(GROUPS_WITHOUT_ORDER, DTYPES))


def group_fixture_id(params):
    group_class = params[0]
    dtype = params[1]
    if len(params) > 2:
        order = params[2]
        return f"{group_class.__name__}, order={order}, dtype={dtype}"
    else:
        return f"{group_class.__name__}, dtype={dtype}"


# Create SymmetryGroup fixture for the set of parameters.
@pytest.fixture(params=PARAMS + PARAMS_ORDER, ids=group_fixture_id)
def group_fixture(request):
    params = request.param
    group_class = params[0]
    dtype = params[1]
    group_kwargs = dict(
        dtype=dtype,
    )
    if len(params) > 2:
        group_kwargs["order"] = params[2]

    return group_class(**group_kwargs)


def test_group_repr(group_fixture):
    """Test SymmetryGroup repr"""
    assert repr(group_fixture).startswith(f"{group_fixture.__class__.__name__}")
    logger.debug(f"SymmetryGroup object: {repr(group_fixture)}")


def test_group_str(group_fixture):
    """Test SymmetryGroup str"""
    sym_string = str(group_fixture)
    logger.debug(f"String for {group_fixture}: {sym_string}.")


def test_group_rotations(group_fixture):
    rotations = group_fixture.rotations
    assert isinstance(rotations, Rotation)


def test_from_string_error():
    junk_symmetry = "P12"
    with pytest.raises(
        ValueError, match=f"Symmetry type {junk_symmetry[0]} not supported.*"
    ):
        _ = SymmetryGroup.from_string(junk_symmetry, dtype=np.float32)
