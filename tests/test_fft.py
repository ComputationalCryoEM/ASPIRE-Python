from unittest import TestCase
import numpy as np

from aspire import config
from aspire.utils.numeric import fft_class

fft_backends = ["scipy", "pyfftw"]
numeric_classes = [np, np]

# Create Cupy fft backend if Cupy is enabled and lib exits.
if config.common.cupy:
    try:
        import cupy as cp

        fft_backends.append("cupy")
        numeric_classes.append(cp)
    except ImportError:
        pass

fft_classes = [fft_class(backend) for backend in fft_backends]


class ConfigTest(TestCase):
    def testFft(self):
        for fft, xp in zip(fft_classes, numeric_classes):
            for nworkers in (-1, 1, 2):
                a = xp.random.random(100)
                b = fft.fft(a, workers=nworkers)
                c = fft.ifft(b, workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testFft2(self):
        for fft, xp in zip(fft_classes, numeric_classes):
            for nworkers in (-1, 1, 2):
                a = xp.random.random((100, 100))
                b = fft.fft2(a, workers=nworkers)
                c = fft.ifft2(b, workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testFftn(self):
        for fft, xp in zip(fft_classes, numeric_classes):
            for nworkers in (-1, 1, 2):
                a = xp.random.random((50, 50, 50))
                b = fft.fftn(a, axes=(0, 1, 2), workers=nworkers)
                c = fft.ifftn(b, axes=(0, 1, 2), workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testShift(self):
        for fft, xp in zip(fft_classes, numeric_classes):
            a = xp.random.random(100)
            b = fft.ifftshift(a)
            c = fft.fftshift(b)
            self.assertTrue(xp.allclose(a, c))
