from unittest.case import SkipTest

import numpy as np

from aspire.image import Image
from aspire.utils import gaussian_2d, utest_tolerance
from aspire.utils.coor_trans import grid_2d
from aspire.utils.random import randn
from aspire.volume import Volume


class Steerable2DMixin:
    def testIndices(self):
        ell_max = self.basis.ell_max
        k_max = self.basis.k_max

        indices = self.basis.indices()

        i = 0

        for ell in range(ell_max + 1):
            if ell == 0:
                sgns = [1]
            else:
                sgns = [1, -1]

            for sgn in sgns:
                for k in range(k_max[ell]):
                    self.assertTrue(indices["ells"][i] == ell)
                    self.assertTrue(indices["sgns"][i] == sgn)
                    self.assertTrue(indices["ks"][i] == k)

                    i += 1

    def testGaussianExpand(self):
        # Offset slightly
        x0 = 0.50
        y0 = 0.75

        # Want sigma to be as large as possible without the Gaussian
        # spilling too much outside the central disk.
        sigma = self.L / 8
        im1 = gaussian_2d(self.L, mu=(x0, y0), sigma=sigma, dtype=self.dtype)

        coef = self.basis.expand(im1)
        im2 = self.basis.evaluate(coef)

        if isinstance(im2, Image):
            im2 = im2.asnumpy()
        im2 = im2[0]

        # For small L there's too much clipping at high freqs to get 1e-3
        # accuracy.
        if self.L < 32:
            atol = 1e-2
        else:
            atol = 1e-3

        self.assertTrue(im1.shape == im2.shape)
        self.assertTrue(np.allclose(im1, im2, atol=atol))

    def testIsotropic(self):
        sigma = self.L / 8
        im = gaussian_2d(self.L, sigma=sigma, dtype=self.dtype)

        coef = self.basis.expand(im)

        ells = self.basis.indices()["ells"]

        energy_outside = np.sum(np.abs(coef[ells != 0]) ** 2)
        energy_total = np.sum(np.abs(coef) ** 2)

        energy_ratio = energy_outside / energy_total

        self.assertTrue(energy_ratio < 0.01)

    def testModulated(self):
        if self.L < 32:
            raise SkipTest

        ell = 1

        sigma = self.L / 8
        im = gaussian_2d(self.L, sigma=sigma, dtype=self.dtype)

        g2d = grid_2d(self.L)

        for trig_fun in (np.sin, np.cos):
            im1 = im * trig_fun(ell * g2d["phi"])

            coef = self.basis.expand(im1)

            ells = self.basis.indices()["ells"]

            energy_outside = np.sum(np.abs(coef[ells != ell]) ** 2)
            energy_total = np.sum(np.abs(coef) ** 2)

            energy_ratio = energy_outside / energy_total

            self.assertTrue(energy_ratio < 0.10)

    def testEvaluateExpand(self):
        coef1 = randn(self.basis.count, seed=self.seed)
        coef1 = coef1.astype(self.dtype)

        im = self.basis.evaluate(coef1)
        if isinstance(im, Image):
            im = im.asnumpy()
        coef2 = self.basis.expand(im)[0]

        self.assertTrue(coef1.shape == coef2.shape)
        self.assertTrue(np.allclose(coef1, coef2, atol=utest_tolerance(self.dtype)))

    def testAdjoint(self):
        u = randn(self.basis.count, seed=self.seed)
        u = u.astype(self.dtype)

        Au = self.basis.evaluate(u)
        if isinstance(Au, Image):
            Au = Au.asnumpy()

        x = Image(randn(*self.basis.sz, seed=self.seed), dtype=self.dtype)

        ATx = self.basis.evaluate_t(x)

        Au_dot_x = np.sum(Au * x.asnumpy())
        u_dot_ATx = np.sum(u * ATx)

        self.assertTrue(Au_dot_x.shape == u_dot_ATx.shape)
        self.assertTrue(np.isclose(Au_dot_x, u_dot_ATx))


class UniversalBasisMixin:
    def getClass(self):
        if self.basis.ndim == 2:
            return Image
        elif self.basis.ndim == 3:
            return Volume

    def testEvaluate(self):
        # evaluate should take a NumPy array and return an Image/Volume
        _class = self.getClass()
        result = self.basis.evaluate(np.zeros((self.basis.count), dtype=self.dtype))
        self.assertTrue(isinstance(result, _class))

    def testEvaluate_t(self):
        # evaluate_t should take an Image/Volume and return a NumPy array
        _class = self.getClass()
        result = self.basis.evaluate_t(
            _class(np.zeros((self.L,) * self.basis.ndim, dtype=self.dtype))
        )
        self.assertTrue(isinstance(result, np.ndarray))

    def testExpand(self):
        _class = self.getClass()
        # expand should take an Image/Volume and return a NumPy array
        result = self.basis.expand(
            _class(np.zeros((self.L,) * self.basis.ndim, dtype=self.dtype))
        )
        self.assertTrue(isinstance(result, np.ndarray))

    def testInitWithIntSize(self):
        # make sure we can instantiate with just an int as a shortcut
        self.assertEqual((self.L,) * self.basis.ndim, self.basis.__class__(self.L).sz)
