import logging
from unittest import TestCase

import numpy as np
from pytest import raises

from aspire.operators import wembed, wemd

logger = logging.getLogger(__name__)


class WEMDTestCase(TestCase):
    def setUp(self):
        self.x = np.random.random((100, 100))
        self.y = np.random.random((100, 100))

    def tearDown(self):
        pass

    def testSmokeTest(self):
        """
        Remove me later after adding real tests.
        """

        _ = wembed(self.x, "coif3", 5)

        d = wemd(self.x, self.y, "coif3", 5)
        logger.info(f"wemd {d}")

    def testDimMistmatch(self):
        """
        Intentionally pass something not 2d or 3d.
        """

        # 1d
        with raises(ValueError):
            _ = wembed(self.x.flatten(), "coif3", 5)

        # 4d
        with raises(ValueError):
            _ = wembed(self.x.reshape((2, 5, 5, 2)), "coif3", 5)

    def test_wembed(self):
        pass

    def test_wemd(self):
        pass
