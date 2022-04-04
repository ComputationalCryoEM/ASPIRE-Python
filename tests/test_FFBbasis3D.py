import os.path
from unittest import TestCase

import numpy as np

from aspire.basis import FFBBasis3D

from ._basis_util import Steerable3DMixin, UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FFBBasis3DTestCase(TestCase, Steerable3DMixin, UniversalBasisMixin):
    def setUp(self):
        self.L = 8
        self.dtype = np.float32
        self.basis = FFBBasis3D((self.L, self.L, self.L), dtype=self.dtype)
        self.seed = 9161341

    def tearDown(self):
        pass
