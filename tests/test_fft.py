from unittest import TestCase

from aspire.utils.numeric import xp
from aspire.utils.numeric import custom_fft


class ConfigTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFft(self):
        a = xp.random.random((100))
        b = custom_fft.fft(a)
        c = custom_fft.ifft(b)

        self.assertTrue(xp.allclose(a, c))

    def testFft2(self):
        a = xp.random.random((100, 100))
        b = custom_fft.fft2(a)
        c = custom_fft.ifft2(b)

        self.assertTrue(xp.allclose(a, c))

    def testFftn(self):
        a = xp.random.random((50, 50, 50))
        b = custom_fft.fftn(a, axes=(0, 1, 2))
        c = custom_fft.ifftn(b, axes=(0, 1, 2))

        self.assertTrue(xp.allclose(a, c))

    def testShift(self):
        a = xp.random.random((100))
        b = custom_fft.ifftshift(a)
        c = custom_fft.fftshift(b)

        self.assertTrue(xp.allclose(a, c))
