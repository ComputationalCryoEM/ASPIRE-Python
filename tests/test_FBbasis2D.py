import os.path

import numpy as np
import pytest
from pytest import raises
from scipy.special import jv

from aspire.basis import Coef, ComplexCoef, FBBasis2D
from aspire.image import Image
from aspire.source import Simulation
from aspire.utils import complex_type, real_type
from aspire.utils.coor_trans import grid_2d
from aspire.utils.random import randn

from ._basis_util import (
    Steerable2DMixin,
    UniversalBasisMixin,
    basis_params_2d,
    show_basis_params,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

# Create a test Basis object for each combination of parameters we want to test
test_bases = [FBBasis2D(L, dtype=dtype) for L, dtype in basis_params_2d]


@pytest.mark.parametrize("basis", test_bases, ids=show_basis_params)
class TestFBBasis2D(UniversalBasisMixin, Steerable2DMixin):
    seed = 9161341

    def _testElement(self, basis, ell, k, sgn):
        # This is covered by the isotropic test.
        assert ell > 0

        ells = basis.angular_indices
        sgns = basis.signs_indices
        ks = basis.radial_indices

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

        im_ref = basis.evaluate(Coef(basis, coef_ref))

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

        # Convert real FB coef to complex coef using Coef class
        cv = v1.to_complex()
        # then convert back to real coef representation.
        v2 = cv.to_real()

        # The round trip should be equivalent up to machine precision
        assert np.allclose(v1, v2)

    def testComplexCoversionErrorsToComplex(self, basis):
        x = randn(*basis.sz, seed=self.seed).astype(basis.dtype)

        # Express in an FB basis, cast to array.
        v1 = basis.expand(x).asnumpy()

        # Test catching Errors
        with raises(TypeError):
            # Pass complex into `to_complex`
            v1_cpx = Coef(basis, v1, dtype=np.complex64)
            _ = basis.to_complex(v1_cpx)

        # Test catching Errors
        with raises(TypeError):
            # Pass complex into `to_complex`
            v1_cpx = Coef(basis, v1, dtype=np.complex64)

        with raises(TypeError):
            # Pass complex into `to_complex`
            v1_cpx = Coef(basis, v1).to_complex()
            _ = v1_cpx.to_complex()

        # Test casting case, where basis and coef don't match
        if basis.dtype == np.float32:
            test_dtype = np.float64
        elif basis.dtype == np.float64:
            test_dtype = np.float32
        # Result should be same precision as coef input, just complex.
        result_dtype = complex_type(test_dtype)

        v3 = basis.to_complex(Coef(basis, v1, dtype=test_dtype))
        assert v3.dtype == result_dtype

    def testComplexCoversionErrorsToReal(self, basis):
        x = randn(*basis.sz, seed=self.seed)

        # Express in an FB basis
        cv = basis.expand(x.astype(basis.dtype))
        ccv = cv.to_complex()

        # Test catching Errors
        with raises(TypeError):
            # Pass real into `to_real`
            _ = basis.to_real(cv)

        # Test catching Errors
        with raises(TypeError):
            # Pass real into `to_real`
            _ = cv.to_real()

        # Test casting case, where basis and coef precision don't match
        if basis.dtype == np.float32:
            test_dtype = np.complex128
        elif basis.dtype == np.float64:
            test_dtype = np.complex64
        # Result should be same precision as coef input, just real.
        result_dtype = real_type(test_dtype)

        x = ComplexCoef(basis, ccv.asnumpy().astype(test_dtype))
        v3 = x.to_real()
        assert v3.dtype == result_dtype


params = [pytest.param(256, np.float32, marks=pytest.mark.expensive)]


@pytest.mark.parametrize(
    "L, dtype",
    params,
)
def testHighResFBBasis2D(L, dtype):
    seed = 42
    basis = FBBasis2D(L, dtype=dtype)
    sim = Simulation(
        n=1,
        L=L,
        dtype=dtype,
        amplitudes=1,
        offsets=0,
        seed=seed,
    )
    im = sim.images[0]

    # Round trip
    coef = basis.expand(im)
    im_fb = basis.evaluate(coef)

    # Mask to compare inside disk of radius 1.
    mask = grid_2d(L, normalized=True)["r"] < 1
    assert np.allclose(im_fb.asnumpy()[0][mask], im.asnumpy()[0][mask], atol=2e-5)
