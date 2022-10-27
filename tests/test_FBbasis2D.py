import os.path
from unittest import TestCase

import numpy as np
from parameterized import parameterized_class
from pytest import raises
from scipy.special import jv

from aspire.basis import FBBasis2D
from aspire.image import Image
from aspire.utils import complex_type, real_type
from aspire.utils.coor_trans import grid_2d
from aspire.utils.random import randn

from ._basis_util import Steerable2DMixin, UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


# NOTE: Class with default values is already present, so don't list it below.
@parameterized_class(
    ("L", "dtype"),
    [
        (8, np.float64),
        (16, np.float32),
        (16, np.float64),
        (32, np.float32),
        (32, np.float64),
    ],
)
class FBBasis2DTestCase(TestCase, Steerable2DMixin, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FBBasis2D((self.L, self.L), dtype=self.dtype)
        self.seed = 9161341

    def tearDown(self):
        pass

    def _testElement(self, ell, k, sgn):
        # This is covered by the isotropic test.
        assert ell > 0

        indices = self.basis.indices()
        ells = indices["ells"]
        sgns = indices["sgns"]
        ks = indices["ks"]

        g2d = grid_2d(self.L, dtype=self.dtype)
        mask = g2d["r"] < 1

        r0 = self.basis.r0[k, ell]

        im = np.zeros((self.L, self.L), dtype=self.dtype)
        im[mask] = jv(ell, g2d["r"][mask] * r0)
        im *= np.sqrt(2**2 / self.L**2)
        im *= 1 / (np.sqrt(np.pi) * np.abs(jv(ell + 1, r0)))

        if sgn == 1:
            im *= np.sqrt(2) * np.cos(ell * g2d["phi"])
        else:
            im *= np.sqrt(2) * np.sin(ell * g2d["phi"])

        coef_ref = np.zeros(self.basis.count, dtype=self.dtype)
        coef_ref[(ells == ell) & (sgns == sgn) & (ks == k)] = 1

        im_ref = self.basis.evaluate(coef_ref)

        coef = self.basis.expand(im)

        # TODO: These tolerances should be tighter.
        self.assertTrue(np.allclose(im, im_ref.asnumpy(), atol=1e-4))
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-4))

    def testElements(self):
        ells = [1, 1, 1, 1]
        ks = [1, 2, 1, 2]
        sgns = [-1, -1, 1, 1]

        for ell, k, sgn in zip(ells, ks, sgns):
            self._testElement(ell, k, sgn)

    def testComplexCoversion(self):
        x = Image(randn(*self.basis.sz, seed=self.seed), dtype=self.dtype)

        # Express in an FB basis
        v1 = self.basis.expand(x)

        # Convert real FB coef to complex coef,
        cv = self.basis.to_complex(v1)
        # then convert back to real coef representation.
        v2 = self.basis.to_real(cv)

        # The round trip should be equivalent up to machine precision
        self.assertTrue(np.allclose(v1, v2))

    def testComplexCoversionErrorsToComplex(self):
        x = randn(*self.basis.sz, seed=self.seed)

        # Express in an FB basis
        v1 = self.basis.expand(x.astype(self.dtype))

        # Test catching Errors
        with raises(TypeError):
            # Pass complex into `to_complex`
            _ = self.basis.to_complex(v1.astype(np.complex64))

        # Test casting case, where basis and coef don't match
        if self.basis.dtype == np.float32:
            test_dtype = np.float64
        elif self.basis.dtype == np.float64:
            test_dtype = np.float32
        # Result should be same precision as coef input, just complex.
        result_dtype = complex_type(test_dtype)

        v3 = self.basis.to_complex(v1.astype(test_dtype))
        self.assertTrue(v3.dtype == result_dtype)

        # Try 0d vector, should not crash.
        _ = self.basis.to_complex(v1.reshape(-1))

    def testComplexCoversionErrorsToReal(self):
        x = randn(*self.basis.sz, seed=self.seed)

        # Express in an FB basis
        cv1 = self.basis.to_complex(self.basis.expand(x.astype(self.dtype)))

        # Test catching Errors
        with raises(TypeError):
            # Pass real into `to_real`
            _ = self.basis.to_real(cv1.real.astype(np.float32))

        # Test casting case, where basis and coef precision don't match
        if self.basis.dtype == np.float32:
            test_dtype = np.complex128
        elif self.basis.dtype == np.float64:
            test_dtype = np.complex64
        # Result should be same precision as coef input, just real.
        result_dtype = real_type(test_dtype)

        v3 = self.basis.to_real(cv1.astype(test_dtype))
        self.assertTrue(v3.dtype == result_dtype)

        # Try a 0d vector, should not crash.
        _ = self.basis.to_real(cv1.reshape(-1))
