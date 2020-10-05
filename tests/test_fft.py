from unittest import TestCase
import numpy as np
from aspire import config
from aspire.utils.numeric import fft_class

fft_backends = ('scipy', 'pyfftw')
fft_classes = [fft_class(backend) for backend in fft_backends]


class ConfigTest(TestCase):
    def testFft(self):
        for fft in fft_classes:
            for nworkers in (-1, 1, 2):
                a = np.random.random(100)
                b = fft.fft(a, workers=nworkers)
                c = fft.ifft(b, workers=nworkers)
                self.assertTrue(np.allclose(a, c))

        # Cupy is not default install and need test separately
        if config.common.cupy:
            import cupy as cp
            fft = fft_class('cupy')
            a = cp.random.random(100)
            # workers is not used for cupy
            b = fft.fft(a, workers=-1)
            c = fft.ifft(b, workers=-1)
            self.assertTrue(cp.allclose(a, c))

    def testFft2(self):
        for fft in fft_classes:
            for nworkers in (-1, 1, 2):
                a = np.random.random((100, 100))
                b = fft.fft2(a, workers=nworkers)
                c = fft.ifft2(b, workers=nworkers)
                self.assertTrue(np.allclose(a, c))

        # Cupy is not default install and need test separately
        if config.common.cupy:
            import cupy as cp
            fft = fft_class('cupy')
            a = cp.random.random((100, 100))
            # workers is not used for cupy
            b = fft.fft2(a, workers=-1)
            c = fft.ifft2(b, workers=-1)
            self.assertTrue(cp.allclose(a, c))

    def testFftn(self):
        for fft in fft_classes:
            for nworkers in (-1, 1, 2):
                a = np.random.random((50, 50, 50))
                b = fft.fftn(a, axes=(0, 1, 2), workers=nworkers)
                c = fft.ifftn(b, axes=(0, 1, 2), workers=nworkers)
                self.assertTrue(np.allclose(a, c))

        # Cupy is not default install and need test separately
        if config.common.cupy:
            import cupy as cp
            fft = fft_class('cupy')
            a = cp.random.random((50, 50, 50))
            # workers is not used for cupy
            b = fft.fftn(a, axes=(0, 1, 2), workers=-1)
            c = fft.ifftn(b, axes=(0, 1, 2), workers=-1)
            self.assertTrue(cp.allclose(a, c))

    def testShift(self):
        for fft in fft_classes:
            a = np.random.random(100)
            b = fft.ifftshift(a)
            c = fft.fftshift(b)
            self.assertTrue(np.allclose(a, c))

        # Cupy is not default install and need test separately
        if config.common.cupy:
            import cupy as cp
            fft = fft_class('cupy')
            a = cp.random.random(100)
            b = fft.ifftshift(a)
            c = fft.fftshift(b)
            self.assertTrue(cp.allclose(a, c))
