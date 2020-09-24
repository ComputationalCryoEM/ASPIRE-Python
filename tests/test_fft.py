"""
In these tests, we iterate through some valid fft backends in for [common.fft] attribute in config.ini
using the `config_override` mechanism for `Config` objects, and check for valid results.

Because the determination of which fft backend to use happens at import time, this is somewhat tricky to do.
Here we employ the `importlib.reload` function while we're iterating through these fft backends to explicitly
trigger a reload.
"""

from importlib import reload
from unittest import TestCase

import aspire.utils.numeric
from aspire.utils.config import config_override

fft_backends = ('scipy', 'pyfftw')


class ConfigTest(TestCase):
    def testFft(self):
        for fft_backend in fft_backends:
            with config_override({'common.fft': fft_backend}):
                reload(aspire.utils.numeric)
                xp, fft = aspire.utils.numeric.xp, aspire.utils.numeric.fft
                a = xp.random.random(100)
                b = fft.fft(a)
                c = fft.ifft(b)

                self.assertTrue(xp.allclose(a, c))

    def testFft2(self):
        for fft_backend in fft_backends:
            with config_override({'common.fft': fft_backend}):
                reload(aspire.utils.numeric)
                xp, fft = aspire.utils.numeric.xp, aspire.utils.numeric.fft
                a = xp.random.random((100, 100))
                b = fft.fft2(a)
                c = fft.ifft2(b)

                self.assertTrue(xp.allclose(a, c))

    def testFftn(self):
        for fft_backend in fft_backends:
            with config_override({'common.fft': fft_backend}):
                reload(aspire.utils.numeric)
                xp, fft = aspire.utils.numeric.xp, aspire.utils.numeric.fft
                a = xp.random.random((50, 50, 50))
                b = fft.fftn(a, axes=(0, 1, 2))
                c = fft.ifftn(b, axes=(0, 1, 2))

                self.assertTrue(xp.allclose(a, c))

    def testShift(self):
        for fft_backend in fft_backends:
            with config_override({'common.fft': fft_backend}):
                reload(aspire.utils.numeric)
                xp, fft = aspire.utils.numeric.xp, aspire.utils.numeric.fft
                a = xp.random.random(100)
                b = fft.ifftshift(a)
                c = fft.fftshift(b)

                self.assertTrue(xp.allclose(a, c))
