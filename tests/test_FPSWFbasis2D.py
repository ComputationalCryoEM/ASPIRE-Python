import os.path

import numpy as np
import pytest

from aspire.basis import FPSWFBasis2D
from aspire.image import Image

from ._basis_util import UniversalBasisMixin, pswf_params_2d, show_basis_params

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

test_bases = [FPSWFBasis2D(L, dtype=dtype) for L, dtype in pswf_params_2d]


@pytest.mark.parametrize("basis", test_bases, ids=show_basis_params)
class TestFPSWFBasis2D(UniversalBasisMixin):
    def testFPSWFBasis2DEvaluate_t(self, basis):
        img_ary = np.load(
            os.path.join(DATA_DIR, "ffbbasis2d_xcoeff_in_8_8.npy")
        ).T  # RCOPT
        images = Image(img_ary)

        result = basis.evaluate_t(images)
        # Historically, FPSWF returned complex values.
        # Load and convert them for this hard coded test.
        ccoeffs = np.load(
            os.path.join(DATA_DIR, "pswf2d_vcoeffs_out_8_8.npy")
        ).T  # RCOPT
        coeffs = basis.to_real(ccoeffs)

        np.testing.assert_allclose(result, coeffs, atol=utest_tolerance(basis.dtype))

    def testFPSWFBasis2DEvaluate(self, basis):
        # Historically, FPSWF returned complex values.
        # Load and convert them for this hard coded test.
        ccoeffs = np.load(
            os.path.join(DATA_DIR, "pswf2d_vcoeffs_out_8_8.npy")
        ).T  # RCOPT
        coeffs = basis.to_real(ccoeffs)
        result = coeffs.evaluate()

        result = basis.evaluate(coeffs)
        images = np.load(os.path.join(DATA_DIR, "pswf2d_xcoeff_out_8_8.npy")).T  # RCOPT

        np.testing.assert_allclose(result, images)
