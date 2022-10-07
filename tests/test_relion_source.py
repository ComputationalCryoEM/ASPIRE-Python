import logging
import os

import pytest

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
