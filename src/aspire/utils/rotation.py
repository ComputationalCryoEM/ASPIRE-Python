"""
Define a Rotation Class for customized rotation operations used by ASPIRE.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils import ensure
from aspire.utils.random import Random


class Rotation:
    def __init__(
        self, num_rots, seed=None, angles=None, matrices=None, dtype=np.float32
    ):
        """
         Initialize a Rotation object

         :param num_rots: Total number of rotation sets
         :param seed: Seed for initializing random rotations.
             If None, the random generator is not re-seeded.
         :param angles: Euler angles (in radian) to generate rotation matrices.
             If None, Predefined rotation matrices or uniformly
             distributed angles will be used.
        :param matrices: Rotation matrices to initialize Rotation object.
             If None, Predefined or uniformly distributed angles
             will be used.
         :param dtype: data type for angles and rotation matrices
        """
        self.dtype = np.dtype(dtype)
        self.num_rots = None
        self.angles = None
        self.matrices = None
        self.shape = None
        if angles is not None:
            ensure(
                num_rots == angles.shape[0],
                "Number of rotation matrices should equal to "
                "number of sets of Euler angles.",
            )
            self.from_euler(angles, dtype=dtype)
        elif matrices is not None:
            ensure(
                num_rots == matrices.shape[0],
                "Number of rotation matrices should equal to "
                "number of input sets of matrices.",
            )
            self.from_matrix(matrices, dtype=dtype)
        else:
            self.num_rots = num_rots
            angles = Rotation.uniform_random_angles(num_rots, seed=seed, dtype=dtype)
            self.from_euler(angles, dtype=dtype)

    def __str__(self):
        """
        String representation.
        """
        return f"Rotation stack consisting of {self.num_rots} elements of {self.dtype} type"

    def __len__(self):
        return self.num_rots

    def __getitem__(self, item):
        return self.matrices[item]

    def __setitem__(self, key, value):
        self.matrices[key] = value

    def __mul__(self, other):
        if isinstance(other, Rotation):
            output = self.matrices @ other.matrices
        else:
            output = self.matrices @ other
        return output

    @property
    def T(self):
        """
        Shortcut to get full set of transposed rotation matrices
        """
        return self.invert()

    def invert(self):
        """
        Apply transpose operation to all rotation matrices

        :return: The set of transposed matrices
        """
        return np.transpose(self.matrices, axes=(0, 2, 1))

    def asnumpy(self):
        """
        :return: Rotation matrices as a n x 3 x 3 array
        """
        return self.matrices

    def from_euler(self, values, dtype=np.float32):
        """
        build rotation object from Euler angles in degrees

        :param dtype:  data type for rotational angles and matrices
        :param values: Rotation angles in radians, as a n x 3 array
        :return: None
        """
        self.dtype = np.dtype(dtype)
        self.num_rots = values.shape[0]
        rotations = sp_rot.from_euler("ZYZ", values.astype(dtype), degrees=False)
        self.matrices = rotations.as_matrix().astype(dtype)
        self.angles = rotations.as_euler("ZYZ", degrees=False).astype(dtype)
        self.shape = (self.num_rots, 3, 3)

    def from_matrix(self, values, dtype=np.float32):
        """
        build rotation object from rotational matrices

        :param dtype:  data type for rotational angles and matrices
        :param values: Rotation matrices, as a n x 3 x 3 array
        :return: None
        """
        self.dtype = np.dtype(dtype)
        self.num_rots = values.shape[0]
        rotations = sp_rot.from_matrix(values.astype(dtype))
        self.angles = rotations.as_euler("ZYZ", degrees=False).astype(dtype)
        self.matrices = rotations.as_matrix().astype(dtype)
        self.shape = (self.num_rots, 3, 3)

    def find_registration(self, rots_ref):
        """
        Register estimated orientations to reference ones.

        Finds the orthogonal transformation that best aligns the estimated rotations
        to the reference rotations.

        :param rots_ref: The reference Rotation object to which we would like to align
            with data matrices in the form of a n-by-3-by-3 array.
        :return: o_mat, optimal orthogonal 3x3 matrix to align the two sets;
                flag, flag==1 then J conjugacy is required and 0 is not.
        """
        rots = self.matrices
        rots_ref = rots_ref.matrices.astype(self.dtype)
        ensure(
            rots.shape == rots_ref.shape,
            "Two sets of rotations must have same dimensions.",
        )
        K = rots.shape[0]

        # Reflection matrix
        J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        Q1 = np.zeros((3, 3), dtype=rots.dtype)
        Q2 = np.zeros((3, 3), dtype=rots.dtype)

        for k in range(K):
            R = rots[k, :, :]
            Rref = rots_ref[k, :, :]
            Q1 = Q1 + R @ Rref.T
            Q2 = Q2 + (J @ R @ J) @ Rref.T

        # Compute the two possible orthogonal matrices which register the
        # estimated rotations to the true ones.
        Q1 = Q1 / K
        Q2 = Q2 / K

        # We are registering one set of rotations (the estimated ones) to
        # another set of rotations (the true ones). Thus, the transformation
        # matrix between the two sets of rotations should be orthogonal. This
        # matrix is either Q1 if we recover the non-reflected solution, or Q2,
        # if we got the reflected one. In any case, one of them should be
        # orthogonal.

        err1 = norm(Q1 @ Q1.T - np.eye(3), ord="fro")
        err2 = norm(Q2 @ Q2.T - np.eye(3), ord="fro")

        # In any case, enforce the registering matrix O to be a rotation.
        if err1 < err2:
            # Use Q1 as the registering matrix
            U, _, V = svd(Q1)
            flag = 0
        else:
            # Use Q2 as the registering matrix
            U, _, V = svd(Q2)
            flag = 1

        Q_mat = U @ V

        return Q_mat, flag

    def apply_registration(self, Q_mat, flag):
        """
        Get aligned Rotation object to reference ones.

        Calculated aligned rotation matrices from the orthogonal transformation
        that best aligns the estimated rotations to the reference rotations.

        :param Q_mat:  optimal orthogonal 3x3 transformation matrix
        :param flag:  flag==1 then J conjugacy is required and 0 is not
        :return: regrot, aligned Rotation object
        """
        rots = self.matrices
        K = rots.shape[0]

        # Reflection matrix
        J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        regrot = np.zeros_like(rots)
        for k in range(K):
            R = rots[k, :, :]
            if flag == 1:
                R = J @ R @ J
            regrot[k, :, :] = Q_mat.T @ R
        aligned_rots = Rotation(K, matrices=regrot)
        return aligned_rots

    def register(self, rots_ref):
        """
         Estimate global orientation and return an aligned Rotation object.

        :param rots_ref: The reference Rotation object to which we would like
            to align with data matrices in the form of a n-by-3-by-3 array.
        :return: an aligned Rotation object
        """
        Q_mat, flag = self.find_registration(rots_ref)
        return self.apply_registration(Q_mat, flag)

    def mse(self, rots_ref):
        """
        Calculate MSE between the estimated orientations to reference ones.

        :param rots_reg: The estimated Rotation object after alignment
             with data matrices in the form of a n-by-3-by-3 array.
        :param rots_ref: The reference Rotation object.
        :return: The MSE value between two sets of rotations.
        """
        aligned_rots = self.register(rots_ref)
        rots_reg = aligned_rots.matrices
        rots_ref = rots_ref.matrices
        ensure(
            rots_reg.shape == rots_ref.shape,
            "Two sets of rotations must have same dimensions.",
        )
        K = rots_reg.shape[0]

        diff = np.zeros(K)
        mse = 0
        for k in range(K):
            diff[k] = norm(rots_reg[k, :, :] - rots_ref[k, :, :], ord="fro")
            mse += diff[k] ** 2
        mse = mse / K
        return mse

    def common_lines(self, i, j, ell):
        """
        Compute the common line induced by rotation matrices i and j.

        :param i: The index of first rotation matrix of 3-by-3 array.
        :param j: The index of second rotation matrix of 3-by-3 array.
        :param ell: The total number of common lines.
        :return: The common line indices for both first and second rotations.
        """
        r1 = self.matrices[i]
        r2 = self.matrices[j]
        ut = np.dot(r2, r1.T)
        alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
        alpha_ji = np.arctan2(ut[0, 2], -ut[1, 2]) + np.pi

        ell_ij = alpha_ij * ell / (2 * np.pi)
        ell_ji = alpha_ji * ell / (2 * np.pi)

        ell_ij = int(np.mod(np.round(ell_ij), ell))
        ell_ji = int(np.mod(np.round(ell_ji), ell))

        return ell_ij, ell_ji

    @staticmethod
    def uniform_random_angles(
        n,
        seed=None,
        dtype=np.float32,
    ):
        """
        Generate random 3D rotation angles in radians

        :param n: The number of rotation angles to generate
        :param seed: Random integer seed to use. If None,
            the current random state is used.
        :return: A n-by-3 ndarray of rotation angles in radians
        """
        # Generate random rotation angles, in radians
        angles = np.zeros((n, 3), dtype=dtype)
        with Random(seed):
            angles = np.column_stack(
                (
                    np.random.random(n) * 2 * np.pi,
                    np.arccos(2 * np.random.random(n) - 1),
                    np.random.random(n) * 2 * np.pi,
                )
            )
        return angles
