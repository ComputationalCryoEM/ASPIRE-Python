from unittest import TestCase

from aspire.utils.numeric import xp
from aspire.utils.numeric import fft


class ConfigTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFft(self):
        a = xp.random.random((100))
        b = fft.fft(a)
        c = fft.ifft(b)

        self.assertTrue(xp.allclose(a, c))

    def testFft2(self):
        a = xp.random.random((100, 100))
        b = fft.fft2(a)
        c = fft.ifft2(b)

        self.assertTrue(xp.allclose(a, c))

    def testFftn(self):
        a = xp.random.random((50, 50, 50))
        b = fft.fftn(a, axes=(0, 1, 2))
        c = fft.ifftn(b, axes=(0, 1, 2))

        self.assertTrue(xp.allclose(a, c))

    def testShift(self):
        a = xp.random.random((100))
        b = fft.ifftshift(a)
        c = fft.fftshift(b)

        self.assertTrue(xp.allclose(a, c))
