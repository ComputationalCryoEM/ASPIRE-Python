import os
from itertools import product
from unittest import TestCase

import numpy as np
from pytest import raises
from scipy.spatial.transform import Rotation

from aspire.source.simulation import Simulation
from aspire.utils import powerset
from aspire.utils.coor_trans import grid_3d
from aspire.utils.types import utest_tolerance
from aspire.volume import Volume, parseSymmetry

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
        self.data_3 = np.zeros((res, res, res), dtype=self.dtype)
        self.data_3[res // 2 + 1, res // 2, res // 2] = 1
        self.data_3[res // 2, res // 2 + 1, res // 2] = 1
        self.data_3[res // 2, res // 2, res // 2 + 1] = 1
        self.vols_1 = Volume(self.data_1)
        self.vols_2 = Volume(self.data_2)
        self.vols_3 = Volume(self.data_3)
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

    def testRotate(self):
        # Create a stack of rotations to test. We will rotate by multiples of pi/2 from each axis.
        rot_mat = np.empty((12, 3, 3), dtype=self.dtype)
        axes = ["x", "y", "z"]
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pairs = list(product(axes, angles))
        for k in range(len(pairs)):
            rot_mat[k] = Rotation.from_euler(pairs[k][0], pairs[k][1]).as_matrix()

        # Rotate the basis volume (contains a 1 on each positive axis, zero elsewhere) by rotation matrices rot_mat.
        # Nyquist frequencies are not set to zero for even resolution to improve error.
        rot_vols = self.vols_3.rotate(0, rot_mat, nyquist=False)

        # Create reference volumes
        L = self.res
        ref_vol = np.zeros((8, L, L, L))
        index = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ref_vol[index, L // 2 + (-1) ** i, L // 2, L // 2] = 1
                    ref_vol[index, L // 2, L // 2 + (-1) ** j, L // 2] = 1
                    ref_vol[index, L // 2, L // 2, L // 2 + (-1) ** k] = 1
                    index += 1
        ref_vol = Volume(ref_vol)

        # Compare rotated volumes with appropriate reference volumes
        atol = utest_tolerance(self.dtype)
        for k in range(3):
            self.assertTrue(np.allclose(ref_vol[0], rot_vols[4 * k], atol=atol))
        for k in range(2):
            self.assertTrue(np.allclose(ref_vol[1], rot_vols[6 * k + 1], atol=atol))
        for k in range(2):
            self.assertTrue(np.allclose(ref_vol[2], rot_vols[6 * k + 3], atol=atol))
        for k in range(2):
            self.assertTrue(np.allclose(ref_vol[4], rot_vols[6 * k + 5], atol=atol))
        self.assertTrue(np.allclose(ref_vol[3], rot_vols[2], atol=atol))
        self.assertTrue(np.allclose(ref_vol[5], rot_vols[6], atol=atol))
        self.assertTrue(np.allclose(ref_vol[6], rot_vols[10], atol=atol))

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
            sim = Simulation(L=L, C=1, symmetry_type=s, dtype=np.float64)
            vol = sim.vols
            ref_vol = vol.rotate(0, rot_mat, nyquist=False)

            # We rotate ref_vol[0] by the stack of rotation matrices
            # rot_vol is a stack of rotated ref_vol[0]
            rot_vol = ref_vol.rotate(0, rot_mat, nyquist=False)

            # Compare rotated volumes to reference volume within the shpere of radius L/4.
            # Check that rotated volumes are within 0.4% of reference volume.
            selection = grid_3d(L, ...)["r"] <= 1 / 2
            for i in range(k):
                ref = ref_vol[0, selection]
                rot = rot_vol[i, selection]
                self.assertTrue(np.amax(abs(rot - ref) / ref) < 0.004)

    def testParseSymmetryError(self):
        # Test we raise with expected message from parseSymmetry
        with raises(NotImplementedError, match=r"J type symmetry.*"):
            _ = parseSymmetry("junk")

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
