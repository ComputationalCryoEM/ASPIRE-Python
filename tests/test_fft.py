from unittest import TestCase

import numpy as np

from aspire import config
from aspire.numeric import fft_object, numeric_object

# Create test option combinations between numerical modules and FFT libs
test_backends = [("numpy", "scipy"), ("numpy", "pyfftw")]

# Create Cupy fft backend if Cupy module is enabled and lib exits.
if config.common.numeric == "cupy":
    test_backends.append(("cupy", "cupy"))

# Create test objects from option combinations
test_objects = [
    (numeric_object(backend[0]), fft_object(backend[1])) for backend in test_backends
]


class ConfigTest(TestCase):
    def testFft(self):
        for backend in test_objects:
            xp, fft = backend
            for nworkers in (-1, 1, 2):
                a = xp.random.random(100)
                b = fft.fft(a, workers=nworkers)
                c = fft.ifft(b, workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testFft2(self):
        for backend in test_objects:
            xp, fft = backend
            for nworkers in (-1, 1, 2):
                a = xp.random.random((100, 100))
                b = fft.fft2(a, workers=nworkers)
                c = fft.ifft2(b, workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testFftn(self):
        for backend in test_objects:
            xp, fft = backend
            for nworkers in (-1, 1, 2):
                a = xp.random.random((50, 50, 50))
                b = fft.fftn(a, axes=(0, 1, 2), workers=nworkers)
                c = fft.ifftn(b, axes=(0, 1, 2), workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testShift(self):
        for backend in test_objects:
            xp, fft = backend
            a = xp.random.random(100)
            b = fft.ifftshift(a)
            c = fft.fftshift(b)
            self.assertTrue(xp.allclose(a, c))

    def testCenteredFft(self):
        for backend in test_objects:
            xp, fft = backend
            for nworkers in (-1, 1, 2):
                a = xp.random.random(100)
                b = fft.centered_fft(a, workers=nworkers)
                self.assertTrue(b[50], np.sum(a))
                c = fft.centered_ifft(b, workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testCenteredFft2(self):
        for backend in test_objects:
            xp, fft = backend
            for nworkers in (-1, 1, 2):
                a = xp.random.random((100, 100))
                b = fft.centered_fft2(a, workers=nworkers)
                self.assertTrue(b[50][50], np.sum(a))
                c = fft.centered_ifft2(b, workers=nworkers)
                self.assertTrue(xp.allclose(a, c))

    def testCenteredFftn(self):
        for backend in test_objects:
            xp, fft = backend
            for nworkers in (-1, 1, 2):
                a = xp.random.random((50, 50, 50))
                b = fft.centered_fftn(a, axes=(0, 1, 2), workers=nworkers)
                self.assertTrue(b[25][25][25], np.sum(a))
                c = fft.centered_ifftn(b, axes=(0, 1, 2), workers=nworkers)
                self.assertTrue(xp.allclose(a, c))
