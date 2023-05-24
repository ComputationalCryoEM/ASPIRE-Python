import logging
import os

import pytest

from aspire.source import RelionSource
from aspire.volume import SymmetryGroup

from .test_starfile_stack import StarFileTestCase

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class StarFileMainCase(StarFileTestCase):
    def setUp(self):
        pass

    # This is a workaround to use a `pytest` fixture with `unittest` style cases.
    # We use it below to capture and inspect the log
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def testIncompletCTFWarning(self):
        with self._caplog.at_level(logging.WARN):
            # This call will instantiate RelionSource
            # During the starfile parsing we should log warning
            #  regarding incomplete CTF params.
            self.setUpStarFile("sample_incomplete_ctf_params.star")
            assert "partially populated CTF Params" in self._caplog.text


def test_symmetry_group(caplog):
    starfile_with_symmetry = os.path.join(
        DATA_DIR, "sample_relion_symmetry_group_D4.star"
    )
    starfile_without_symmetry = os.path.join(DATA_DIR, "sample_particles_relion30.star")

    # Check default symmetry_group.
    src = RelionSource(starfile_without_symmetry)
    assert isinstance(src.symmetry_group, SymmetryGroup)
    assert str(src.symmetry_group) == "C1"

    # Check symmetry_group attribute.
    src_sym = RelionSource(starfile_with_symmetry)
    assert isinstance(src_sym.symmetry_group, SymmetryGroup)
    assert str(src_sym.symmetry_group) == "D4"

    # Check overriding symmetry_group with RelionSource argument.
    caplog.clear()
    msg = "Overriding metadata with supplied symmetry group"
    caplog.set_level(logging.WARN)
    assert msg not in caplog.text
    src_override_sym = RelionSource(starfile_with_symmetry, symmetry_group="C6")
    assert msg in caplog.text

    assert isinstance(src_override_sym.symmetry_group, SymmetryGroup)
    assert str(src_override_sym.symmetry_group) == "C6"
