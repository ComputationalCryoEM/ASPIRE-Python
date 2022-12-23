import numpy as np

import pytest

from aspire.image import Image
from aspire.utils import gaussian_2d, utest_tolerance
from aspire.utils.coor_trans import grid_2d
from aspire.utils.random import randn
from aspire.volume import Volume

seed = 0
class Steerable2DMixin:
    def testIndices(self, L, dtype):
        basis = self.getBasis(L, dtype)
        ell_max = basis.ell_max
        k_max = basis.k_max

        indices = basis.indices()

        i = 0

        for ell in range(ell_max + 1):
            if ell == 0:
                sgns = [1]
            else:
                sgns = [1, -1]

            for sgn in sgns:
                for k in range(k_max[ell]):
                    assert indices["ells"][i] == ell
                    assert indices["sgns"][i] == sgn
                    assert indices["ks"][i] == k

                    i += 1

    def testGaussianExpand(self, L, dtype):
        basis = self.getBasis(L, dtype)
        # Offset slightly
        x0 = 0.50
        y0 = 0.75

        # Want sigma to be as large as possible without the Gaussian
        # spilling too much outside the central disk.
        sigma = L / 8
        im1 = gaussian_2d(L, mu=(x0, y0), sigma=sigma, dtype=dtype)

        coef = basis.expand(im1)
        im2 = basis.evaluate(coef)

        if isinstance(im2, Image):
            im2 = im2.asnumpy()
        im2 = im2[0]

        # For small L there's too much clipping at high freqs to get 1e-3
        # accuracy.
        if L < 32:
            atol = 1e-2
        else:
            atol = 1e-3

        assert im1.shape == im2.shape
        assert np.allclose(im1, im2, atol=atol)

    def testIsotropic(self, L, dtype):
        basis = self.getBasis(L, dtype)
        sigma = L / 8
        im = gaussian_2d(L, sigma=sigma, dtype=dtype)

        coef = basis.expand(im)

        ells = basis.indices()["ells"]

        energy_outside = np.sum(np.abs(coef[ells != 0]) ** 2)
        energy_total = np.sum(np.abs(coef) ** 2)

        energy_ratio = energy_outside / energy_total

        assert energy_ratio < 0.01

    def testModulated(self, L, dtype):
        basis = self.getBasis(L, dtype)
        if L < 32:
            pytest.skip()

        ell = 1

        sigma = L / 8
        im = gaussian_2d(L, sigma=sigma, dtype=dtype)

        g2d = grid_2d(L)

        for trig_fun in (np.sin, np.cos):
            im1 = im * trig_fun(ell * g2d["phi"])

            coef = basis.expand(im1)

            ells = basis.indices()["ells"]

            energy_outside = np.sum(np.abs(coef[ells != ell]) ** 2)
            energy_total = np.sum(np.abs(coef) ** 2)

            energy_ratio = energy_outside / energy_total

            assert energy_ratio < 0.10

    def testEvaluateExpand(self, L, dtype):
        basis = self.getBasis(L, dtype)
        coef1 = randn(basis.count, seed=seed)
        coef1 = coef1.astype(dtype)

        im = basis.evaluate(coef1)
        if isinstance(im, Image):
            im = im.asnumpy()
        coef2 = basis.expand(im)[0]

        assert coef1.shape == coef2.shape
        assert np.allclose(coef1, coef2, atol=utest_tolerance(dtype))

    def testAdjoint(self, L, dtype):
        basis = self.getBasis(L, dtype)
        u = randn(basis.count, seed=seed)
        u = u.astype(dtype)

        Au = basis.evaluate(u)
        if isinstance(Au, Image):
            Au = Au.asnumpy()

        x = Image(randn(*basis.sz, seed=seed), dtype=dtype)

        ATx = basis.evaluate_t(x)

        Au_dot_x = np.sum(Au * x.asnumpy())
        u_dot_ATx = np.sum(u * ATx)

        assert Au_dot_x.shape == u_dot_ATx.shape
        assert np.isclose(Au_dot_x, u_dot_ATx)


class UniversalBasisMixin:
    """
    Each function must take L and dtype as parameters
    """
    def getClass(self, L, dtype):
        basis = self.getBasis(L, dtype)
        if basis.ndim == 2:
            return Image
        elif basis.ndim == 3:
            return Volume

    def testEvaluate(self, L, dtype):
        # evaluate should take a NumPy array of type basis.coefficient_dtype
        # and return an Image/Volume
        _class = self.getClass(L, dtype)
        basis = self.getBasis(L, dtype)
        result = basis.evaluate(
            np.zeros((basis.count), dtype=basis.coefficient_dtype)
        )
        assert isinstance(result, _class)

    def testEvaluate_t(self, L, dtype):
        # evaluate_t should take an Image/Volume and return a NumPy array of type
        # basis.coefficient_dtype
        _class = self.getClass(L, dtype)
        basis = self.getBasis(L, dtype)
        result = basis.evaluate_t(
            _class(np.zeros((L,) * basis.ndim, dtype=dtype))
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == basis.coefficient_dtype

    def testExpand(self, L, dtype):
        _class = self.getClass(L, dtype)
        basis = self.getBasis(L, dtype)
        # expand should take an Image/Volume and return a NumPy array of type
        # basis.coefficient_dtype
        result = basis.expand(
            _class(np.zeros((L,) * basis.ndim, dtype=dtype))
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == basis.coefficient_dtype

    def testInitWithIntSize(self, L, dtype):
        basis = self.getBasis(L, dtype)
        # make sure we can instantiate with just an int as a shortcut
        assert (L,) * basis.ndim == basis.__class__(L).sz
