import os

import tempfile
from itertools import product
from unittest import TestCase

import numpy as np
from parameterized import parameterized
from pytest import raises
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils import Rotation, powerset
from aspire.utils.coor_trans import grid_3d
from aspire.utils.types import utest_tolerance
from aspire.volume import Volume, gaussian_blob_vols

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class VolumeTestCase(TestCase):
    # res is at this scope to be picked up by parameterization in testRotate.
    res = 42

    def setUp(self):
        self.dtype = np.float32
        self.n = n = 3
        self.data_1 = np.arange(n * self.res ** 3, dtype=self.dtype).reshape(
            n, self.res, self.res, self.res
        )
        self.data_2 = 123 * self.data_1.copy()
        self.vols_1 = Volume(self.data_1)
        self.vols_2 = Volume(self.data_2)
        self.random_data = np.random.randn(self.res, self.res, self.res).astype(
            self.dtype
        )
        self.vec = self.data_1.reshape(n, self.res ** 3)

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
            r_stack[r] = sp_rot.from_euler(ax, 0).as_matrix()
            # We'll consider the multiples of pi/2.
            r_stack[r + 3] = sp_rot.from_euler(ax, np.pi / 2).as_matrix()
            r_stack[r + 6] = sp_rot.from_euler(ax, np.pi).as_matrix()
            r_stack[r + 9] = sp_rot.from_euler(ax, 3 * np.pi / 2).as_matrix()

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

    # Parameterize over even and odd resolutions
    @parameterized.expand([(res,), (res - 1,)])
    def testRotate(self, L):
        # Create a Volume instance to rotate.
        # This "basis" volume has a 1 placed along each positive axis, zeros elsewhere.
        data = np.zeros((L, L, L), dtype=self.dtype)
        data[L // 2 + 1, L // 2, L // 2] = 1
        data[L // 2, L // 2 + 1, L // 2] = 1
        data[L // 2, L // 2, L // 2 + 1] = 1
        vol = Volume(data)

        # Create a stack of rotations to test. We will rotate vol by multiples of pi/2 from each axis.
        _rot_mat = np.empty((12, 3, 3), dtype=self.dtype)
        axes = ["x", "y", "z"]
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for k, (axis, angle) in enumerate(product(axes, angles)):
            _rot_mat[k] = sp_rot.from_euler(axis, angle).as_matrix()
        rot_mat = Rotation(_rot_mat)

        # Rotate the basis volume by rotation matrices rot_mat.
        # For even resolution we keep the Nyquist frequency.
        # This reduces error on the rotations.
        rot_vols = vol.rotate(0, rot_mat, zero_nyquist=False)

        # Create reference volumes.
        # All possible orientations of rotating the basis volume about each axis by multiples of pi/2.
        # For ref_vol[i,j,k], a 1 is place on the positive axis if i,j, or k is 0,
        # and a 1 is placed on the negative axis if i,j, or k is 1.
        # For example, ref_vol[0,1,1] has a 1 along the positive x-axis, and 1's on the negative y and z axes.
        ref_vol = np.zeros((2, 2, 2, L, L, L))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ref_vol[i, j, k, L // 2 + (-1) ** i, L // 2, L // 2] = 1
                    ref_vol[i, j, k, L // 2, L // 2 + (-1) ** j, L // 2] = 1
                    ref_vol[i, j, k, L // 2, L // 2, L // 2 + (-1) ** k] = 1

        # Compare rotated volumes with appropriate reference volumes.
        atol = utest_tolerance(self.dtype)
        # 3 equivalent cases of rotating by zero degrees about each axis.
        for k in range(3):
            self.assertTrue(np.allclose(ref_vol[0, 0, 0], rot_vols[4 * k], atol=atol))

        # Equivalent cases of rotation by pi/2 about y-axis and 3*pi/2 about x-axis.
        for k in range(2):
            self.assertTrue(
                np.allclose(ref_vol[0, 0, 1], rot_vols[6 * k + 1], atol=atol)
            )

        # Equivalent cases of rotation by pi/2 about x-axis and 3*pi/2 about z-axis.
        for k in range(2):
            self.assertTrue(
                np.allclose(ref_vol[0, 1, 0], rot_vols[6 * k + 3], atol=atol)
            )

        # Equivalent cases of rotation by pi/2 about z-axis and 3*pi/2 about y-axis.
        for k in range(2):
            self.assertTrue(
                np.allclose(ref_vol[1, 0, 0], rot_vols[6 * k + 5], atol=atol)
            )

        # Rotation by pi about x-axis, pi about y-axis, and pi about z-axis, respectively.
        self.assertTrue(np.allclose(ref_vol[0, 1, 1], rot_vols[2], atol=atol))
        self.assertTrue(np.allclose(ref_vol[1, 0, 1], rot_vols[6], atol=atol))
        self.assertTrue(np.allclose(ref_vol[1, 1, 0], rot_vols[10], atol=atol))

    def testCnSymmetricVolume(self):
        # We create volumes with Cn symmetry and check that they align when rotated by multiples of 2pi/n.
        L = self.res
        sym_type = {2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6"}

        for k, s in sym_type.items():
            # Build rotation matrices that rotate by multiples of 2pi/k about the z axis
            rot_mat = np.zeros((k, 3, 3), dtype=self.dtype)
            for i in range(k):
                rot_mat[i, :, :] = [
                    [np.cos(2 * i * np.pi / k), -np.sin(2 * i * np.pi / k), 0],
                    [np.sin(2 * i * np.pi / k), np.cos(2 * i * np.pi / k), 0],
                    [0, 0, 1],
                ]

            # To build a reference volume we rotate a volume instance.
            # ref_vol[0] is a volume rotated by zero degrees.
            vol = gaussian_blob_vols(
                L=L, C=1, symmetry_type=s, seed=0, dtype=self.dtype
            )
            ref_vol = vol.rotate(0, rot_mat, zero_nyquist=False)

            # We rotate ref_vol[0] by the stack of rotation matrices
            # rot_vol is a stack of rotated ref_vol[0]
            rot_vol = ref_vol.rotate(0, rot_mat, zero_nyquist=False)

            # Compare rotated volumes to reference volume within the shpere of radius L/4.
            # Check that rotated volumes are within 1% of reference volume.
            selection = grid_3d(L, dtype=self.dtype)["r"] <= 1 / 2
            for i in range(k):
                ref = ref_vol[0, selection]
                rot = rot_vol[i, selection]
                self.assertTrue(np.amax(abs(rot - ref) / ref) < 0.01)

        # Test we raise with expected error message when volume is instantiated with unsupported C-type symmetry.
        with raises(NotImplementedError, match=r"CH2 symmetry not supported.*"):
            _ = gaussian_blob_vols(symmetry_type="Ch2")

        # Test we raise with expected message for junk symmetry.
        with raises(NotImplementedError, match=r"J type symmetry.*"):
            _ = gaussian_blob_vols(symmetry_type="junk")

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
