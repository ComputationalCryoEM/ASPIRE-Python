import os.path
from unittest import TestCase

import numpy as np
from parameterized import parameterized_class

from aspire.basis import FFBBasis3D

from ._basis_util import Steerable3DMixin, UniversalBasisMixin

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
class FFBBasis3DTestCase(TestCase, Steerable3DMixin, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FFBBasis3D((self.L, self.L, self.L), dtype=self.dtype)
        self.seed = 9161341

    def tearDown(self):
        pass
