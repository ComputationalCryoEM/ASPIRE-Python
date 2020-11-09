"""
Define a Rotation Class for customized rotation operations used by ASPIRE.
"""

import copy
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils import ensure
from aspire.utils.matlab_compat import Random


class Rotation:

    def __init__(self, num_rots, seed=None, seq='ZYZ',
                 angles=None, dtype=np.float32):
        """
        Initialize a Rotation object

        :param num_rots: Total number of rotation sets
        :param seed: Seed for initializing random rotations.
            If None, numpy.random will continue previous state.
        :param seq:  Sequence of order applying Euler angles
        :param angles: Euler angles (in degrees) to generate rotation matrices.
            If None, uniformly distributed angles will be generated.
        :param dtype: data type for angles and rotation matrices
        """
        self.rot_seq = seq
        self.dtype = np.dtype(dtype)
        if angles:
            ensure(num_rots == angles.shape[0],
                   'Number of rotation matrices should equal to '
                   'number of sets of Euler angles.')
            self.angles = angles.astype(self.dtype)
        else:
            self.angles = self._uniform_random_angles(
                num_rots, seed=seed)
        self.num_rots = num_rots

        self.data = self.rot_matrices
        self.shape = (num_rots, 3, 3)

    def _uniform_random_angles(self, n, seed=None):
        """
        Generate random 3D rotation angles in degrees

        :param n: The number of rotation angles to generate
        :param seed: Random integer seed to use. If None,
            the current random state is used.
        :return: A n-by-3 ndarray of rotation angles in degrees
        """
        # Generate random rotation angles, in radians
        angles = np.zeros((n, 3), dtype=self.dtype)
        with Random(seed):
            angles = np.column_stack((
                np.random.random(n) * 2 * np.pi,
                np.arccos(2 * np.random.random(n) - 1),
                np.random.random(n) * 2 * np.pi
            ))
        # Return random rotation angles in degrees
        return np.rad2deg(angles).astype(self.dtype)

    @property
    def rot_matrices(self):
        """
        :return: Rotation matrices as a n x 3 x 3 array
        """
        return self._rotations.as_matrix().astype(self.dtype)

    @rot_matrices.setter
    def rot_matrices(self, values):
        """
        Set rotation matrices

        :param values: Rotation matrices as a n x 3 x 3 array
        :return: None
        """
        self._rotations = sp_rot.from_matrix(values.astype(self.dtype))
        self.data = values.astype(self.dtype)

    @property
    def angles(self):
        """
        :return: Rotation angles in degrees, as a n x 3 array
        """
        return self._rotations.as_euler(self.rot_seq, degrees=True
                                        ).astype(self.dtype)

    @angles.setter
    def angles(self, values):
        """
        Set rotation angles in degrees

        :param values: Rotation angles in degrees, as a n x 3 array
        :return: None
        """
        self._rotations = sp_rot.from_euler(
            self.rot_seq, values.astype(self.dtype), degrees=True)

    @staticmethod
    def register_rotations(rots, rots_ref):
        """
        Register estimated orientations to reference ones.

        Finds the orthogonal transformation that best aligns the estimated rotations
        to the reference rotations.

        :param rots: The rotations to be aligned in the form of a n-by-3-by-3 array.
        :param rots_ref: The reference rotations to which we would like to align in
            the form of a n-by-3-by-3 array.
        :return: o_mat, optimal orthogonal 3x3 matrix to align the two sets;
                flag, flag==1 then J conjugacy is required and 0 is not.
        """

        ensure(rots.shape == rots_ref.shape,
               'Two sets of rotations must have same dimensions.')
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

        err1 = norm(Q1 @ Q1.T - np.eye(3), ord='fro')
        err2 = norm(Q2 @ Q2.T - np.eye(3), ord='fro')

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

    @staticmethod
    def get_aligned_rotations(rots, Q_mat, flag):
        """
        Get aligned rotation matrices to reference ones.

        Calculated aligned rotation matrices from the orthogonal transformation
        that best aligns the estimated rotations to the reference rotations.


        :param rots: The reference rotations to which we would like to align in
            the form of a n-by-3-by-3 array.
        :param Q_mat:  optimal orthogonal 3x3 transformation matrix
        :param flag:  flag==1 then J conjugacy is required and 0 is not
        :return: regrot, aligned rotation matrices
        """

        K = rots.shape[0]

        # Reflection matrix
        J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        regrot = np.zeros_like(rots)
        for k in range(K):
            R = rots[k, :, :]
            if flag == 1:
                R = J @ R @ J
            regrot[k, :, :] = Q_mat.T @ R

        return regrot

    @staticmethod
    def get_rots_mse(rots_reg, rots_ref):
        """
        Calculate MSE between the estimated orientations to reference ones.

        :param rots_reg: The estimated rotations after alignment in the form of
            a n-by-3-by-3 array.
        :param rots_ref: The reference rotations.
        :return: The MSE value between two sets of rotations.
        """
        ensure(rots_reg.shape == rots_ref.shape,
               'Two sets of rotations must have same dimensions.')
        K = rots_reg.shape[0]

        diff = np.zeros(K)
        mse = 0
        for k in range(K):
            diff[k] = norm(rots_reg[k, :, :] - rots_ref[k, :, :], ord='fro')
            mse += diff[k] ** 2
        mse = mse / K
        return mse

    @staticmethod
    def common_line_from_rots(r1, r2, ell):
        """
        Compute the common line induced by rotation matrices r1 and r2.

        :param r1: The first rotation matrix of 3-by-3 array.
        :param r2: The second rotation matrix of 3-by-3 array.
        :param ell: The total number of common lines.
        :return: The common line indices for both first and second rotations.
        """
        ut = np.dot(r2, r1.T)
        alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
        alpha_ji = np.arctan2(ut[0, 2], -ut[1, 2]) + np.pi

        ell_ij = alpha_ij * ell / (2 * np.pi)
        ell_ji = alpha_ji * ell / (2 * np.pi)

        ell_ij = int(np.mod(np.round(ell_ij), ell))
        ell_ji = int(np.mod(np.round(ell_ji), ell))

        return ell_ij, ell_ji

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __mul__(self, other):
        if isinstance(other, Rotation):
            other = other.data
            output = np.zeros_like(other)
            for i in range(self.num_rots):
                output[i] = np.matmul(self.data[i], other[i])
        else:
            output = self.data*other
        return output

    @property
    def T(self):
        """
        Shortcut to get full set of transposed rotation matrices
        """
        return self.transpose()

    def transpose(self):
        """
        Apply transpose operation to all rotation matrices

        :return: The set of transposed matrices
        """
        T = np.zeros((self.num_rots, 3, 3), dtype=self.dtype)
        for i in range(self.num_rots):
            T[i] = self.data[i].T
        return T

    def __str__(self):
        """
        String representation.
        """
        return f'Rotation object with matrices[{self.num_rots}, 3, 3] of {self.dtype} type'

    def copy(self):
        """
        Returns a copy of `self`.

        :return Rotation object like self
        """

        return copy.copy(self)

    def __len__(self):
        return self.num_rots

    def asnumpy(self):
        return self.data
