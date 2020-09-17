from unittest import TestCase

from aspire.utils.numeric import xp


class ConfigTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFft(self):
        a = xp.random.random((100))
        b = xp.fft(a)
        c = xp.ifft(b)

        self.assertTrue(xp.allclose(a, c))

    def testFft2(self):
        a = xp.random.random((100, 100))
        b = xp.fft2(a)
        c = xp.ifft2(b)

        self.assertTrue(xp.allclose(a, c))

    def testFftn(self):
        a = xp.random.random((50, 50, 50))
        b = xp.fftn(a, axes=(0, 1, 2))
        c = xp.ifftn(b, axes=(0, 1, 2))

        self.assertTrue(xp.allclose(a, c))

    def testShift(self):
        a = xp.random.random((100))
        b = xp.ifftshift(a)
        c = xp.fftshift(b)

        self.assertTrue(xp.allclose(a, c))
