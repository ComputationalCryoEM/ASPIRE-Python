import os.path

import numpy as np
import pytest

from aspire.basis import ComplexCoef, PSWFBasis2D
from aspire.image import Image

from ._basis_util import UniversalBasisMixin, pswf_params_2d, show_basis_params

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

test_bases = [PSWFBasis2D(L, dtype=dtype) for L, dtype in pswf_params_2d]


@pytest.mark.parametrize("basis", test_bases, ids=show_basis_params)
class TestPSWFBasis2D(UniversalBasisMixin):
    def testPSWFBasis2DEvaluate_t(self, basis):
        img_ary = np.load(os.path.join(DATA_DIR, "ffbbasis2d_xcoef_in_8_8.npy"))
        images = Image(img_ary)

        result = basis.evaluate_t(images)

        # Historically, PSWF returned complex values.
        # Load and convert them for this hard coded test.
        ccoefs = np.load(os.path.join(DATA_DIR, "pswf2d_vcoefs_out_8_8.npy")).T  # RCOPT
        coefs = ComplexCoef(basis, ccoefs).to_real()

        np.testing.assert_allclose(result, coefs, rtol=1e-05, atol=1e-08)

    def testPSWFBasis2DEvaluate(self, basis):
        # Historically, PSWF returned complex values.
        # Load and convert them for this hard coded test.
        ccoefs = np.load(os.path.join(DATA_DIR, "pswf2d_vcoefs_out_8_8.npy")).T  # RCOPT
        coefs = ComplexCoef(basis, ccoefs).to_real()

        result = coefs.evaluate()

        # This hardcoded reference result requires transposing the stack axis.
        images = np.transpose(
            np.load(os.path.join(DATA_DIR, "pswf2d_xcoef_out_8_8.npy")), (2, 0, 1)
        )

        np.testing.assert_allclose(result.asnumpy(), images, rtol=1e-05, atol=1e-08)
