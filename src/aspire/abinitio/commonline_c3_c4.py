import logging

import numpy as np
from numpy import linalg

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CLSymmetryC3C4(CLOrient3D):
    """
    Define a class to estimate 3D orientations using common lines methods for molecules with
    C3 and C4 cyclical symmetry.

    The related publications are listed below:
    G. Pragier and Y. Shkolnisky,
    A Common Lines Approach for Abinitio Modeling of Cyclically Symmetric Molecules,
    Inverse Problems, 35, 124005, (2019).

    Y. Shkolnisky, and A. Singer,
    Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

    A. Singer, R. R. Coifman, F. J. Sigworth, D. W. Chester, Y. Shkolnisky,
    Detecting Consistent Common Lines in Cryo-EM by Voting,
    Journal of Structural Biology, 169, 312-322 (2010).
    """

    def __init__(self, src, n_symm=None, n_rad=None, n_theta=None):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_symm: The symmetry order of the molecule. 3 or 4.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """

        super().__init__(src, n_rad=n_rad, n_theta=n_theta)

        self.n_symm = n_symm

    def orientation_estimation(self):
        """
        Estimate rotation matrices for symmetric molecules.
        """
        pass

    ###########################################
    # Primary Methods                         #
    ###########################################

    def compute_third_row_outer_prod_c34(self, max_shift_1d):
        """
        Compute the outer products of the third rows of the rotation matrices Rij and Rii.
        """

        pf = self.pf
        n_symm = self.n_symm
        max_shift = self.max_shift
        shift_step = self.shift_step
        n_theta = self.n_theta

        # Step 1: Detect a single pair of common-lines between each pair of images

        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        if n_symm == 3:
            is_handle_equator_ims = False
        else:
            is_handle_equator_ims = True

        sclmatrix = self.self_clmatrix_c3_c4(
            pf, n_symm, max_shift, shift_step, is_handle_equator_ims
        )

        # Step 3: Calculate self-relative-rotations
        Riis = self.estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta)

        # Step 4: Calculate relative rotations
        Rijs = self.estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta)

        # Step 5: Inner J-synchronization
        vijs, viis = self.local_sync_J_c3_c4(n_symm, Rijs, Riis)

        return vijs, viis

    @staticmethod
    def global_sync_J(vijs, viis):
        """
        Global J-synchronization of all third row outer products. Given 3x3 matrices vijs and viis, each
        of which might contain a spurious J, we return vijs and viis that all have either a spurious J
        or not.

        :param vijs: An nchoose2x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of matrices Ri and Rj. Each estimate might have a
        spurious J independently of other estimates.

        :param viis: An nx3x3 array where the ith slice holds an estimate for the outer product vi*vi^T
        between the third row of matrix Ri and itself. Each estimate might have a spurious J independently
        of other estimates.

        :return: vijs, viis all of which have a spurious J or not.
        """

        n_ims = viis.shape[0]
        n_vijs = vijs.shape[0]
        nchoose2 = int(n_ims * (n_ims - 1) / 2)
        assert viis.shape[1:] == (3, 3), "viis must be 3x3 matrices."
        assert vijs.shape[1:] == (3, 3), "vijs must be 3x3 matrices."
        assert n_vijs == nchoose2, "There must be n_ims-choose-2 vijs."
        pass

    @staticmethod
    def estimate_third_rows(vijs, viis):
        """
        Find the third row of each rotation matrix given third row outer products.

        :param vijs: An n-choose-2x3x3 array where each 3x3 slice holds the third rows
        outer product of the corresponding pair of matrices.

        :param viis: An nx3x3 array where the i-th 3x3 slice holds the outer product of
        the third row of Ri with itself.

        :param n_symm: The underlying molecular symmetry.

        :return: vis, An n_imagesx3 matrix whose i-th row is the third row of the rotation matrix Ri.
        """

        n_ims = viis.shape[0]
        n_vijs = vijs.shape[0]
        nchoose2 = int(n_ims * (n_ims - 1) / 2)
        assert viis.shape[1:] == (3, 3), "viis must be 3x3 matrices."
        assert vijs.shape[1:] == (3, 3), "vijs must be 3x3 matrices."
        assert n_vijs == nchoose2, "There must be n_ims-choose-2 vijs."

        # Build 3nx3n matrix V whose (i,j)-th block of size 3x3 holds the outer product vij
        V = np.zeros((3 * n_ims, 3 * n_ims), dtype=vijs.dtype)

        # All pairs (i,j) where i<j
        indices = np.arange(n_ims)
        pairs = [(i, j) for idx, i in enumerate(indices) for j in indices[idx + 1 :]]

        # Populate upper triangle of V with vijs
        for idx, (i, j) in enumerate(pairs):
            V[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = vijs[idx]

        # Populate lower triangle of V with vjis, where vji = vij^T
        V = V + V.T

        # Populate diagonal of V with viis
        for i in range(n_ims):
            V[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = viis[i]

        # In a clean setting V is of rank 1 and its eigenvector is the concatenation
        # of the third rows of all rotation matrices.
        # In the noisy setting we use the eigenvector corresponding to the leading eigenvalue
        val, vec = linalg.eig(V)
        lead_idx = np.argsort(val)[-1]
        lead_vec = vec[:, lead_idx]

        vis = lead_vec.reshape((n_ims, 3))
        for i in range(n_ims):
            vis[i] = vis[i] / linalg.norm(vis[i])

        return vis

    def estimate_inplane_rotations(
        self, pf, vis, inplane_rot_res, max_shift, shift_step
    ):
        # return rots
        pass

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def self_clmatrix_c3_c4(
        self, pf, n_symm, max_shift, shift_step, is_handle_equator_ims
    ):
        # return sclmatrix
        pass

    def estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta):
        # return Riis
        pass

    def estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta):
        # return Rijs
        pass

    def local_sync_J_c3_c4(n_symm, Rijs, Riis):
        # return vijs, viis
        pass
