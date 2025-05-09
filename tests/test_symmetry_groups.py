import itertools
import logging

import numpy as np
import pytest

from aspire.utils import Rotation
from aspire.volume.symmetry_groups import (
    CnSymmetryGroup,
    DnSymmetryGroup,
    IdentitySymmetryGroup,
    ISymmetryGroup,
    OSymmetryGroup,
    SymmetryGroup,
    TSymmetryGroup,
)

logger = logging.getLogger(__name__)

GROUPS_WITH_ORDER = [
    CnSymmetryGroup,
    DnSymmetryGroup,
]
GROUPS_WITHOUT_ORDER = [
    (TSymmetryGroup,),
    (OSymmetryGroup,),
    (ISymmetryGroup,),
]
ORDERS = [2, 3, 4, 5]
PARAMS_ORDER = list(itertools.product(GROUPS_WITH_ORDER, ORDERS))


def group_fixture_id(params):
    group_class = params[0]
    if len(params) > 1:
        order = params[1]
        return f"{group_class.__name__}, order={order}"
    else:
        return f"{group_class.__name__}"


# Create SymmetryGroup fixture for the set of parameters.
@pytest.fixture(params=GROUPS_WITHOUT_ORDER + PARAMS_ORDER, ids=group_fixture_id)
def group_fixture(request):
    params = request.param
    group_class = params[0]
    group_kwargs = dict()
    if len(params) > 1:
        group_kwargs["order"] = params[1]

    return group_class(**group_kwargs)


def test_group_repr(group_fixture):
    """Test SymmetryGroup repr"""
    assert repr(group_fixture).startswith(f"{group_fixture.__class__.__name__}")
    logger.debug(f"SymmetryGroup object: {repr(group_fixture)}")


def test_group_str(group_fixture):
    """Test SymmetryGroup str"""
    sym_string = str(group_fixture)
    logger.debug(f"String for {group_fixture}: {sym_string}.")


def test_group_equivalence(group_fixture):
    C2_symmetry_group = CnSymmetryGroup(order=2)
    if str(group_fixture) == "C2":
        assert C2_symmetry_group == group_fixture
    else:
        assert C2_symmetry_group != group_fixture


def test_group_rotations(group_fixture):
    rotations = group_fixture.rotations
    assert isinstance(rotations, Rotation)


def test_dtype(group_fixture):
    """Test SymmetryGroup matrices are always doubles."""
    np.testing.assert_equal(group_fixture.matrices.dtype, np.float64)


def test_parser_identity():
    result = SymmetryGroup.parse("C1")
    assert isinstance(result, IdentitySymmetryGroup)


def test_parser_with_group(group_fixture):
    """Test SymmetryGroup instance are parsed correctly."""
    result = SymmetryGroup.parse(group_fixture)
    assert result == group_fixture


def test_parser_error():
    junk_symmetry = "P12"
    with pytest.raises(
        ValueError, match=f"Symmetry type {junk_symmetry[0]} not supported.*"
    ):
        _ = SymmetryGroup.parse(junk_symmetry)


def test_group_order(group_fixture):
    """Check the number of elements (order) in each symmetry group."""
    if type(group_fixture) in GROUPS_WITH_ORDER:
        expected_orders = {
            CnSymmetryGroup: 1 * group_fixture.order,
            DnSymmetryGroup: 2 * group_fixture.order,
        }
    else:
        expected_orders = {
            IdentitySymmetryGroup: 1,
            TSymmetryGroup: 12,
            OSymmetryGroup: 24,
            ISymmetryGroup: 60,
        }
    expected_order = expected_orders[type(group_fixture)]
    assert len(group_fixture.matrices) == expected_order
