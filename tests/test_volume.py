import os
import tempfile
from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation

from aspire.utils import powerset
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class VolumeTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.n = n = 3
        self.res = res = 42
        self.data_1 = np.arange(n * res ** 3, dtype=self.dtype).reshape(
            n, res, res, res
        )
        self.data_2 = 123 * self.data_1.copy()
        self.vols_1 = Volume(self.data_1)
        self.vols_2 = Volume(self.data_2)
        self.random_data = np.random.randn(res, res, res).astype(self.dtype)
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
        self.assertTrue(np.allclose(self.vols_1[k], self.random_data))

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

    def testSaveLoad(self):
        # Create a tmpdir in a context. It will be cleaned up on exit.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the Volume object into an MRC files
            mrcs_filepath = os.path.join(tmpdir, "test.mrc")
            self.vols_1.save(mrcs_filepath)

            # Load saved MRC file as a Volume of dtypes single and double.
            vols_loaded_single = Volume.load(mrcs_filepath, dtype=np.float32)
            vols_loaded_double = Volume.load(mrcs_filepath, dtype=np.float64)

            # Check that loaded data are Volume instances and compare to original volume.
            self.assertTrue(isinstance(vols_loaded_single, Volume))
            self.assertTrue(np.allclose(self.vols_1, vols_loaded_single))
            self.assertTrue(isinstance(vols_loaded_double, Volume))
            self.assertTrue(np.allclose(self.vols_1, vols_loaded_double))

    def testProject(self):
        # Create a stack of rotations to test.
        r_stack = np.empty((12, 3, 3), dtype=self.dtype)
        for r, ax in enumerate(["x", "y", "z"]):
            r_stack[r] = Rotation.from_euler(ax, 0).as_matrix()
            # We'll consider the multiples of pi/2.
            r_stack[r + 3] = Rotation.from_euler(ax, np.pi / 2).as_matrix()
            r_stack[r + 6] = Rotation.from_euler(ax, np.pi).as_matrix()
            r_stack[r + 9] = Rotation.from_euler(ax, 3 * np.pi / 2).as_matrix()

        # Project a Volume with all the test rotations
        vol_id = 1  # select a volume from Volume stack
        img_stack = self.vols_1.project(vol_id, r_stack)

        for r in range(len(r_stack)):
            # Get result of test projection at center of Image.
            prj_along_axis = img_stack[r][21, 21]

            # For Volume, take mean along the axis of rotation.
            vol_along_axis = np.mean(self.vols_1[vol_id], axis=r % 3)
            # Volume is uncentered, take the mean of a 2x2 window.
            vol_along_axis = np.mean(vol_along_axis[20:22, 20:22])

            # The projection and Volume should be equivalent
            #  centered along the rotation axis for multiples of pi/2.
            self.assertTrue(np.allclose(vol_along_axis, prj_along_axis))

    def to_vec(self):
        """Compute the to_vec method and compare."""
        result = self.vols_1.to_vec()
        self.assertTrue(result == self.vec)
        self.assertTrue(isinstance(result, np.ndarray))

    def testFromVec(self):
        """Compute Volume from_vec method and compare."""
        vol = Volume.from_vec(self.vec)
        self.assertTrue(np.allclose(vol, self.vols_1))
        self.assertTrue(isinstance(vol, Volume))

    def testVecId1(self):
        """Test composition of from_vec(to_vec)."""
        # Construct vec
        vec = self.vols_1.to_vec()

        # Convert back to Volume and compare
        self.assertTrue(np.allclose(Volume.from_vec(vec), self.vols_1))

    def testVecId2(self):
        """Test composition of to_vec(from_vec)."""
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

    def testFlip(self):
        # Test over all sane axis.
        for axis in powerset(range(4)):
            if not axis:
                # test default
                result = self.vols_1.flip()
                axis = 0
            else:
                result = self.vols_1.flip(axis)
            self.assertTrue(np.all(result == np.flip(self.data_1, axis)))
            self.assertTrue(isinstance(result, Volume))

    def testDownsample(self):
        # Data files re-used from test_preprocess
        vols = Volume(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")))

        resv = Volume(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy")))

        result = vols.downsample((8, 8, 8))
        self.assertTrue(np.allclose(result, resv))
        self.assertTrue(isinstance(result, Volume))
