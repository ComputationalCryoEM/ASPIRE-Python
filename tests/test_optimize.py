from unittest import TestCase

import numpy as np
from numpy.random import random

from aspire.optimization import conj_grad


class OptimizeTestCase(TestCase):
    def setUp(self):
        # Generate real and complex A matrices from random numbers
        rand_mat = random((4, 4))
        self.A = rand_mat @ rand_mat.T
        rand_mat_comp = random((4, 4)) + random((4, 4)) * 1j
        self.A_comp = rand_mat_comp @ rand_mat_comp.T.conjugate()

        # Generate real and complex x vectors from random numbers
        self.b = random(4)
        self.b_comp = random(4) + random(4) * 1j

    def testConjGradReal(self):
        x_est, _, _ = conj_grad(lambda x: self.A @ x, self.b)
        self.assertTrue(np.allclose(self.A @ x_est, self.b))

    def testConjGradComplex(self):
        x_comp_est, _, _ = conj_grad(lambda x: self.A_comp @ x, self.b_comp)
        self.assertTrue(np.allclose(self.A_comp @ x_comp_est, self.b_comp))
