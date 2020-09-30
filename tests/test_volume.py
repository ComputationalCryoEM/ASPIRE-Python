from unittest import TestCase

import numpy as np
import os

from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class VolumeTestCase(TestCase):

    def setUp(self):
        self.n = n = 3
        self.res = res = 42
        self.data_1 = np.arange(n * res ** 3, dtype=np.float64).reshape(
            n, res, res, res)
        self.data_2 = 123 * self.data_1.copy()
        self.vols_1 = Volume(self.data_1)
        self.vols_2 = Volume(self.data_2)
        self.random_data = np.random.randn(res, res, res)
        self.vec = self.data_1.reshape(n, res ** 3)

    def tearDown(self):
        pass

    def testAsNumpy(self):
        self.assertTrue(np.all(self.data_1 == self.vols_1.asnumpy()))

    def testGetter(self):
        k = np.random.randint(self.n)
        self.assertTrue(np.all(self.vols_1[k] == self.data_1[k]))

    def testSetter(self):
        k = np.random.randint(self.n)
        ref = self.vols_1.asnumpy().copy()
        # Set one entry in the stack with new data
        self.vols_1[k] = self.random_data

        # Assert we have updated the kth volume
        self.assertTrue(np.all(self.vols_1[k] == self.random_data))

        # Assert the other volumes are not updated.
        inds = np.arange(self.n) != k
        self.assertTrue(np.all(self.vols_1[inds] == ref[inds]))

    def testLen(self):
        self.assertTrue(len(self.vols_1) == self.n)

        # Also test a single volume
        self.assertTrue(len(Volume(self.random_data)) == 1)

    def testAdd(self):
        result = self.vols_1 + self.vols_2
        self.assertTrue(np.all(result == self.data_1 + self.data_2))
        self.assertTrue(isinstance(result, Volume))

    def testScalarAdd(self):
        result = self.vols_1 + 42
        self.assertTrue(np.all(result == self.data_1 + 42))
        self.assertTrue(isinstance(result, Volume))

    def testScalarRAdd(self):
        result = 42 + self.vols_1
        self.assertTrue(np.all(result == self.data_1 + 42))
        self.assertTrue(isinstance(result, Volume))

    def testSub(self):
        result = self.vols_1 - self.vols_2
        self.assertTrue(np.all(result == self.data_1 - self.data_2))
        self.assertTrue(isinstance(result, Volume))

    def testScalarSub(self):
        result = self.vols_1 - 42
        self.assertTrue(np.all(result == self.data_1 - 42))
        self.assertTrue(isinstance(result, Volume))

    def testScalarRSub(self):
        result = 42 - self.vols_1
        self.assertTrue(np.all(result == 42 - self.data_1))
        self.assertTrue(isinstance(result, Volume))

    def testScalarMul(self):
        result = self.vols_1 * 123
        self.assertTrue(np.all(result == self.data_2))
        self.assertTrue(isinstance(result, Volume))

    def testScalarRMul(self):
        result = 123 * self.vols_1
        self.assertTrue(np.all(result == self.data_2))
        self.assertTrue(isinstance(result, Volume))

    def testProject(self):
        pass

    def to_vec(self):
        """ Compute the to_vec method and compare. """
        result = self.vols_1.to_vec()
        self.assertTrue(result == self.vec)
        self.assertTrue(isinstance(result, np.ndarray))

    def testFromVec(self):
        """ Compute Volume from_vec method and compare. """
        vol = Volume.from_vec(self.vec)
        self.assertTrue(np.allclose(vol, self.vols_1))
        self.assertTrue(isinstance(vol, Volume))

    def testVecId1(self):
        """ Test composition of from_vec(to_vec). """
        # Construct vec
        vec = self.vols_1.to_vec()

        # Convert back to Volume and compare
        self.assertTrue(np.allclose(Volume.from_vec(vec), self.vols_1))

    def testVecId2(self):
        """ Test composition of to_vec(from_vec). """
        # Construct Volume
        vol = Volume.from_vec(self.vec)

        # # Convert back to vec and compare
        self.assertTrue(np.all(vol.to_vec() == self.vec))

    def testTranspose(self):
        data_t = np.transpose(self.data_1, (0, 3, 2, 1))

        result = self.vols_1.transpose()
        self.assertTrue(np.all(result == data_t))
        self.assertTrue(isinstance(result, Volume))

        result = self.vols_1.T
        self.assertTrue(np.all(result == data_t))
        self.assertTrue(isinstance(result, Volume))

    def testFlatten(self):
        result = self.vols_1.flatten()
        self.assertTrue(np.all(result == self.data_1.flatten()))
        self.assertTrue(isinstance(result, np.ndarray))

    def testDownsample(self):
        # Data files re-used from test_preprocess
        vols = Volume(
            np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy')))

        resv = Volume(
            np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol_down8.npy')))

        result = vols.downsample((8, 8, 8))
        self.assertTrue(np.allclose(result, resv))
        self.assertTrue(isinstance(result, Volume))
