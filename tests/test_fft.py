from unittest import TestCase
import numpy as np
from aspire.utils.numeric import fft_class

fft_backends = ('scipy', 'pyfftw')
fft_classes = [fft_class(backend) for backend in fft_backends]


class ConfigTest(TestCase):
    def testFft(self):
        for fft in fft_classes:
            a = np.random.random(100)
            b = fft.fft(a, workers=2)
            c = fft.ifft(b, workers=-1)

            self.assertTrue(np.allclose(a, c))

    def testFft2(self):
        for fft in fft_classes:
            a = np.random.random((100, 100))
            b = fft.fft2(a, workers=None)
            c = fft.ifft2(b)

            self.assertTrue(np.allclose(a, c))

    def testFftn(self):
        for fft in fft_classes:
            a = np.random.random((50, 50, 50))
            b = fft.fftn(a, axes=(0, 1, 2), workers=2)
            c = fft.ifftn(b, axes=(0, 1, 2), workers=3)

            self.assertTrue(np.allclose(a, c))

    def testShift(self):
        for fft in fft_classes:
            a = np.random.random(100)
            b = fft.ifftshift(a)
            c = fft.fftshift(b)

            self.assertTrue(np.allclose(a, c))
