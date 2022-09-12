from unittest import TestCase

import numpy as np

from aspire.basis import FLEBasis2D

from ._basis_util import UniversalBasisMixin


class FLEBasis2DTestCase(TestCase, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FLEBasis2D((self.L, self.L), dtype=self.dtype)
