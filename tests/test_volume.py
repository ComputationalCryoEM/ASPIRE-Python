import os
import tempfile
from itertools import product
from unittest import TestCase

import numpy as np
from numpy import pi
from parameterized import parameterized
from pytest import raises, skip

from aspire.utils import Rotation, grid_3d, powerset
from aspire.utils.matrix import anorm
from aspire.utils.types import utest_tolerance
from aspire.volume import Volume, gaussian_blob_vols

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class VolumeTestCase(TestCase):
    # res is at this scope to be picked up by parameterization in testRotate.
    res = 42

    def setUp(self):
        self.dtype = np.float32
        self.n = n = 3
        self.data_1 = np.arange(n * self.res**3, dtype=self.dtype).reshape(
            n, self.res, self.res, self.res
        )
        self.data_2 = 123 * self.data_1.copy()
        self.vols_1 = Volume(self.data_1)
        self.vols_2 = Volume(self.data_2)
        self.random_data = np.random.randn(self.res, self.res, self.res).astype(
            self.dtype
        )
        self.vec = self.data_1.reshape(n, self.res**3)

    def tearDown(self):
        pass

    def testAsNumpy(self):
        self.assertTrue(np.all(self.data_1 == self.vols_1.asnumpy()))

    def testAsType(self):
        if self.dtype == np.float64:
            new_dtype = np.float32
        elif self.dtype == np.float32:
            new_dtype = np.float64
        else:
            skip("Skip numerically comparing non float types.")

        v2 = self.vols_1.astype(new_dtype)
        self.assertTrue(isinstance(v2, Volume))
        self.assertTrue(np.allclose(v2.asnumpy(), self.vols_1.asnumpy()))
        self.assertTrue(v2.dtype == new_dtype)

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
            r_stack[r] = Rotation.about_axis(ax, 0).matrices
            # We'll consider the multiples of pi/2.
            r_stack[r + 3] = Rotation.about_axis(ax, pi / 2).matrices
            r_stack[r + 6] = Rotation.about_axis(ax, pi).matrices
            r_stack[r + 9] = Rotation.about_axis(ax, 3 * pi / 2).matrices

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
        # In this test we instantiate Volume instance `vol`, containing a single nonzero
        # voxel in the first octant, and rotate it by multiples of pi/2 about each axis.
        # We then compare to reference volumes containing appropriately located nonzero voxel.

        # Create a Volume instance to rotate.
        # This volume has a value of 1 in the first octant at (1, 1, 1) and zeros elsewhere.
        data = np.zeros((L, L, L), dtype=self.dtype)
        data[L // 2 + 1, L // 2 + 1, L // 2 + 1] = 1
        vol = Volume(data)

        # Create a dict with map from axis and angle of rotation to new location of nonzero voxel.
        ref_pts = {
            ("x", 0): (1, 1, 1),
            ("x", pi / 2): (1, 1, -1),
            ("x", pi): (1, -1, -1),
            ("x", 3 * pi / 2): (1, -1, 1),
            ("y", 0): (1, 1, 1),
            ("y", pi / 2): (-1, 1, 1),
            ("y", pi): (-1, 1, -1),
            ("y", 3 * pi / 2): (1, 1, -1),
            ("z", 0): (1, 1, 1),
            ("z", pi / 2): (1, -1, 1),
            ("z", pi): (-1, -1, 1),
            ("z", 3 * pi / 2): (-1, 1, 1),
        }

        center = np.array([L // 2] * 3)

        # Rotate Volume 'vol' and test against reference volumes.
        axes = ["x", "y", "z"]
        angles = [0, pi / 2, pi, 3 * pi / 2]
        for axis, angle in product(axes, angles):
            # Build rotation matrices
            rot_mat = Rotation.about_axis(axis, angle)

            # Rotate Volume 'vol' by rotations 'rot_mat'
            rot_vol = vol.rotate(rot_mat, zero_nyquist=False)

            # Build reference volumes using dict 'ref_pts'
            ref_vol = np.zeros((L, L, L), dtype=np.float32)
            # Assign the location of non zero voxel
            loc = center + np.array(ref_pts[axis, angle])
            ref_vol[tuple(loc)] = 1

            # Test that rotated volumes align with reference volumes
            self.assertTrue(
                np.allclose(ref_vol, rot_vol, atol=utest_tolerance(self.dtype))
            )

    def testRotateBroadcastUnicast(self):
        # Build `Rotation` objects. A singleton for broadcasting and a stack for unicasting.
        # The stack consists of copies of the singleton.
        angles = np.array([pi, pi / 2, 0])
        angles = np.tile(angles, (3, 1))
        rot_mat = Rotation.from_euler(angles).matrices
        rot = Rotation(rot_mat[0])
        rots = Rotation(rot_mat)

        # Broadcast the singleton `Rotation` across the `Volume` stack.
        vols_broadcast = self.vols_1.rotate(rot)

        # Unicast the `Rotation` stack across the `Volume` stack.
        vols_unicast = self.vols_1.rotate(rots)

        for i in range(self.n):
            self.assertTrue(np.allclose(vols_broadcast[i], vols_unicast[i]))

    def testCnSymmetricVolume(self):
        # We create volumes with Cn symmetry and check that they align when rotated by multiples of 2pi/n.
        L = self.res
        sym_type = {2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6"}

        for k, s in sym_type.items():

            # Build `Volume` instance with symmetry type s.
            vol = gaussian_blob_vols(L=L, C=1, symmetry=s, seed=0, dtype=self.dtype)

            # Build rotation matrices that rotate by multiples of 2pi/k about the z axis.
            angles = np.zeros(shape=(k, 3))
            angles[:, 2] = 2 * np.pi * np.arange(k) / k
            rot_mat = Rotation.from_euler(angles).matrices

            # Create mask to compare volumes on.
            selection = grid_3d(L, dtype=self.dtype)["r"] <= 1 / 2

            for i in range(k):
                # Rotate volume.
                rot = Rotation(rot_mat[i])
                rot_vol = vol.rotate(rot, zero_nyquist=False)

                # Restrict volumes to mask for comparison.
                ref = vol[0, selection]
                rot = rot_vol[0, selection]

                # Assert that rotated volume is within .5% of original volume.
                self.assertTrue(np.amax(abs(rot - ref) / ref) < 0.005)

        # Test we raise with expected error message when volume is instantiated with unsupported symmetry.
        with raises(NotImplementedError, match=r"CH2 symmetry not supported.*"):
            _ = gaussian_blob_vols(symmetry="Ch2")

        # Test we raise with expected message for junk symmetry.
        with raises(NotImplementedError, match=r"J type symmetry.*"):
            _ = gaussian_blob_vols(symmetry="junk")

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
        for axis in powerset(range(1, 4)):
            if not axis:
                # test default
                result = self.vols_1.flip()
                axis = 1
            else:
                result = self.vols_1.flip(axis)
            self.assertTrue(np.all(result == np.flip(self.data_1, axis)))
            self.assertTrue(isinstance(result, Volume))

        # Test axis 0 raises
        msg = r"Cannot flip Axis 0, stack axis."
        with raises(ValueError, match=msg):
            _ = self.vols_1.flip(axis=0)

        with raises(ValueError, match=msg):
            _ = self.vols_1.flip(axis=(0, 1))

    def testDownsample(self):
        vols = Volume(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")))
        result = vols.downsample(8)
        res = vols.resolution
        ds_res = result.resolution

        # check signal energy
        self.assertTrue(
            np.allclose(
                anorm(vols.asnumpy(), axes=(1, 2, 3)) / res,
                anorm(result.asnumpy(), axes=(1, 2, 3)) / ds_res,
                atol=1e-3,
            )
        )

        # check gridpoints
        self.assertTrue(
            np.allclose(
                vols[:, res // 2, res // 2, res // 2],
                result[:, ds_res // 2, ds_res // 2, ds_res // 2],
                atol=1e-4,
            )
        )
