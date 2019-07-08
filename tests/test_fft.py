from unittest import TestCase
import numpy as np
from aspyre.utils.numeric import xp


class ConfigTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFft2(self):
        a = xp.random.random((100, 100))
        b = xp.fft2(a)
        c = xp.ifft2(b)

        self.assertTrue(np.allclose(a, c))
