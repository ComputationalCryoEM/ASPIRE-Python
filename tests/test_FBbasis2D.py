import os.path

import numpy as np
import pytest
from pytest import raises
from scipy.special import jv

from aspire.basis import FBBasis2D
from aspire.image import Image
from aspire.utils import complex_type, real_type
from aspire.utils.coor_trans import grid_2d
from aspire.utils.random import randn

from ._basis_util import Steerable2DMixin, UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


params = [
    (8, np.float32),
    (8, np.float64),
    (16, np.float32),
    (16, np.float64),
    (32, np.float32),
    (32, np.float64),
]

test_bases = [FBBasis2D(L, dtype=dtype) for L, dtype in params]


def show_basis_params(basis):
    # print descriptive test name for parametrized test
    # run pytest with option -rA to see explicitly
    return f"{basis.nres}-{basis.dtype}"


@pytest.mark.parametrize("basis", test_bases, ids=show_basis_params)
class TestFBBasis2D(UniversalBasisMixin, Steerable2DMixin):
    seed = 9161341

    def _testElement(self, basis, ell, k, sgn):
        # This is covered by the isotropic test.
        assert ell > 0

        indices = basis.indices()
        ells = indices["ells"]
        sgns = indices["sgns"]
        ks = indices["ks"]

        g2d = grid_2d(basis.nres, dtype=basis.dtype)
        mask = g2d["r"] < 1

        r0 = basis.r0[ell][k]

        im = np.zeros((basis.nres, basis.nres), dtype=basis.dtype)
        im[mask] = jv(ell, g2d["r"][mask] * r0)
        im *= np.sqrt(2**2 / basis.nres**2)
        im *= 1 / (np.sqrt(np.pi) * np.abs(jv(ell + 1, r0)))

        if sgn == 1:
            im *= np.sqrt(2) * np.cos(ell * g2d["phi"])
        else:
            im *= np.sqrt(2) * np.sin(ell * g2d["phi"])

        coef_ref = np.zeros(basis.count, dtype=basis.dtype)
        coef_ref[(ells == ell) & (sgns == sgn) & (ks == k)] = 1

        im_ref = basis.evaluate(coef_ref)

        coef = basis.expand(im)

        # TODO: These tolerances should be tighter.
        assert np.allclose(im, im_ref.asnumpy(), atol=1e-4)
        assert np.allclose(coef, coef_ref, atol=1e-4)

    def testElements(self, basis):
        ells = [1, 1, 1, 1]
        ks = [1, 2, 1, 2]
        sgns = [-1, -1, 1, 1]

        for ell, k, sgn in zip(ells, ks, sgns):
            self._testElement(basis, ell, k, sgn)

    def testComplexCoversion(self, basis):
        x = Image(randn(*basis.sz, seed=self.seed), dtype=basis.dtype)

        # Express in an FB basis
        v1 = basis.expand(x)

        # Convert real FB coef to complex coef,
        cv = basis.to_complex(v1)
        # then convert back to real coef representation.
        v2 = basis.to_real(cv)

        # The round trip should be equivalent up to machine precision
        assert np.allclose(v1, v2)

    def testComplexCoversionErrorsToComplex(self, basis):
        x = randn(*basis.sz, seed=self.seed)

        # Express in an FB basis
        v1 = basis.expand(x.astype(basis.dtype))

        # Test catching Errors
        with raises(TypeError):
            # Pass complex into `to_complex`
            _ = basis.to_complex(v1.astype(np.complex64))

        # Test casting case, where basis and coef don't match
        if basis.dtype == np.float32:
            test_dtype = np.float64
        elif basis.dtype == np.float64:
            test_dtype = np.float32
        # Result should be same precision as coef input, just complex.
        result_dtype = complex_type(test_dtype)

        v3 = basis.to_complex(v1.astype(test_dtype))
        assert v3.dtype == result_dtype

        # Try 0d vector, should not crash.
        _ = basis.to_complex(v1.reshape(-1))

    def testComplexCoversionErrorsToReal(self, basis):
        x = randn(*basis.sz, seed=self.seed)

        # Express in an FB basis
        cv1 = basis.to_complex(basis.expand(x.astype(basis.dtype)))

        # Test catching Errors
        with raises(TypeError):
            # Pass real into `to_real`
            _ = basis.to_real(cv1.real.astype(np.float32))

        # Test casting case, where basis and coef precision don't match
        if basis.dtype == np.float32:
            test_dtype = np.complex128
        elif basis.dtype == np.float64:
            test_dtype = np.complex64
        # Result should be same precision as coef input, just real.
        result_dtype = real_type(test_dtype)

        v3 = basis.to_real(cv1.astype(test_dtype))
        assert v3.dtype == result_dtype

        # Try a 0d vector, should not crash.
        _ = basis.to_real(cv1.reshape(-1))
