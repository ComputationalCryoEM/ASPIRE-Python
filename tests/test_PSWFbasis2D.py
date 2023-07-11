import os.path

import numpy as np
import pytest

from aspire.basis import PSWFBasis2D
from aspire.image import Image

from ._basis_util import UniversalBasisMixin, pswf_params_2d, show_basis_params

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

test_bases = [PSWFBasis2D(L, dtype=dtype) for L, dtype in pswf_params_2d]


@pytest.mark.parametrize("basis", test_bases, ids=show_basis_params)
class TestPSWFBasis2D(UniversalBasisMixin):
    def testPSWFBasis2DEvaluate_t(self, basis):
        img_ary = np.load(
            os.path.join(DATA_DIR, "ffbbasis2d_xcoeff_in_8_8.npy")
        ).T  # RCOPT
        images = Image(img_ary)

        result = basis.evaluate_t(images)

        ccoeffs = np.load(
            os.path.join(DATA_DIR, "pswf2d_vcoeffs_out_8_8.npy")
        ).T  # RCOPT
        coeffs = basis.to_real(ccoeffs)

        # make sure both real and imaginary parts are consistent.
        assert np.allclose(result, coeffs)

    def testPSWFBasis2DEvaluate(self, basis):
        ccoeffs = np.load(
            os.path.join(DATA_DIR, "pswf2d_vcoeffs_out_8_8.npy")
        ).T  # RCOPT
        coeffs = basis.to_real(ccoeffs)

        result = coeffs.evaluate()
        images = np.load(os.path.join(DATA_DIR, "pswf2d_xcoeff_out_8_8.npy")).T  # RCOPT
        assert np.allclose(result, images)
