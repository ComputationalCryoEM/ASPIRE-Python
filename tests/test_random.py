from unittest import TestCase
from aspyre.utils.matlab_compat import randi


class UtilsRandomTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testRandi(self):
        l = list(randi(10, 10, seed=0))
        # This should produce identical results to MATLAB `randi(10, 1, 10)` with the same random seed (0)
        self.assertListEqual(l, [9, 10, 2, 10, 7, 1, 3, 6, 10, 10])

