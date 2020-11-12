import logging

import numpy as np
from scipy.spatial.transform import Rotation

from aspire.abinitio import CLOrient3D
from aspire.utils import ensure
from aspire.utils.matlab_compat import stable_eigsh

logger = logging.getLogger(__name__)


class CLSyncVoting(CLOrient3D):
    """
    Define a class to estimate 3D orientations using synchronization matrix and voting method.

    The related publications are listed as below:
    Y. Shkolnisky, and A. Singer,
    Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

    A. Singer, R. R. Coifman, F. J. Sigworth, D. W. Chester, Y. Shkolnisky,
    Detecting Consistent Common Lines in Cryo-EM by Voting,
    Journal of Structural Biology, 169, 312-322 (2010).
    """

    def __init__(self, src, n_rad=None, n_theta=None):
        """
        Initialize an object for estimating 3D orientations using synchronization matrix

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """
        super().__init__(src, n_rad=n_rad, n_theta=n_theta)
        self.syncmatrix = None

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images using synchronization matrix
        """
        if self.syncmatrix is None:
            self.syncmatrix_vote()

        S = self.syncmatrix
        sz = S.shape
        ensure(sz[0] == sz[1], "syncmatrix must be a square matrix.")
        ensure(sz[0] % 2 == 0, "syncmatrix must be a square matrix of size 2Kx2K.")

        n_img = sz[0] // 2

        # S is a 2Kx2K matrix (K=n_img), containing KxK blocks of size 2x2.
        # The [i,j] block is given by [r11 r12; r12 r22], where
        # r_{kl}=<R_{i}^{k},R_{j}^{l}>, k,l=1,2, namely, the dot product of
        # column k of R_{i} and columns l of R_{j}. Thus, given the true
        # rotations R_{1},...,R_{K}, S is decomposed as S=W^{T}W where
        # W=(R_{1}^{1},R_{1}^{2},...,R_{K}^{1},R_{K}^{2}), where R_{j}^{k}
        # the k column of R_{j}. Therefore, S is a rank-3 matrix, and thus, it
        # three eigenvectors that correspond to non-zero eigenvalues, are linear
        # combinations of the column space of S, namely, W^{T}.

        # Extract three eigenvectors corresponding to non-zero eigenvalues.
        d, v = stable_eigsh(S, 10)
        sort_idx = np.argsort(-d)
        logger.info(
            f"Top 10 eigenvalues from synchronization voting matrix: {d[sort_idx]}"
        )

        # Only need the top 3 eigen-vectors.
        v = v[:, sort_idx[:3]]
        # According to the structure of W^{T} above, the odd rows of V, denoted V1,
        # are a linear combination of the vectors R_{i}^{1}, i=1,...,K, that is of
        # column 1 of all rotation matrices. Similarly, the even rows of V,
        # denoted, V2, are linear combinations of R_{i}^{1}, i=1,...,K.
        v1 = v[: 2 * n_img : 2].T.copy()
        v2 = v[1 : 2 * n_img : 2].T.copy()

        # We look for a linear transformation (3 x 3 matrix) A such that
        # A*V1'=R1 and A*V2=R2 are the columns of the rotations matrices.
        # Therefore:
        # V1 * A'*A V1' = 1
        # V2 * A'*A V2' = 1
        # V1 * A'*A V2' = 0
        # These are 3*K linear equations for 9 matrix entries of A'*A
        # Actually, there are only 6 unknown variables, because A'*A is symmetric.
        # So we will truncate from 9 variables to 6 variables corresponding
        # to the upper half of the matrix A'*A
        truncated_equations = np.zeros((3 * n_img, 6))
        k = 0
        for i in range(3):
            for j in range(i, 3):
                truncated_equations[0::3, k] = v1[i] * v1[j]
                truncated_equations[1::3, k] = v2[i] * v2[j]
                truncated_equations[2::3, k] = v1[i] * v2[j]
                k += 1

        # b = [1 1 0 1 1 0 ...]' is the right hand side vector
        b = np.ones(3 * n_img)
        b[2::3] = 0

        # Find the least squares approximation of A'*A in vector form
        ATA_vec = np.linalg.lstsq(truncated_equations, b, rcond=None)[0]

        # Construct the matrix A'*A from the vectorized matrix.
        ATA = np.zeros((3, 3))
        upper_mask = np.triu_indices(3)
        ATA[upper_mask] = ATA_vec
        lower_mask = np.tril_indices(3)
        ATA[lower_mask] = ATA.T[lower_mask]

        # The Cholesky decomposition of A'*A gives A
        # numpy returns lower, matlab upper
        a = np.linalg.cholesky(ATA)

        # Recover the rotations. The first two columns of all rotation
        # matrices are given by unmixing V1 and V2 using A. The third
        # column is the cross product of the first two.
        r1 = np.dot(a, v1)
        r2 = np.dot(a, v2)
        r3 = np.cross(r1, r2, axis=0)

        rotations = np.empty((n_img, 3, 3))
        rotations[:, :, 0] = r1.T
        rotations[:, :, 1] = r2.T
        rotations[:, :, 2] = r3.T
        # Make sure that we got rotations by enforcing R to be
        # a rotation (in case the error is large)
        u, _, v = np.linalg.svd(rotations)
        np.einsum("ijk, ikl -> ijl", u, v, out=rotations)

        self.rotations = rotations

    def syncmatrix_vote(self):
        """
        Construct the synchronization matrix using voting method

        A pre-computed common line matrix is required as input.
        """
        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        sz = clmatrix.shape
        n_theta = self.n_theta

        ensure(sz[0] == sz[1], "clmatrix must be a square matrix.")

        n_img = sz[0]
        S = np.eye(2 * n_img, dtype=self.dtype).reshape(n_img, 2, n_img, 2)

        # Build Synchronization matrix from the rotation blocks in X and Y
        for i in range(n_img - 1):
            for j in range(i + 1, n_img):
                rot_block = self._syncmatrix_ij_vote(
                    clmatrix, i, j, np.arange(n_img), n_theta
                )
                S[i, :, j, :] = rot_block
                S[j, :, i, :] = rot_block.T

        self.syncmatrix = S.reshape(2 * n_img, 2 * n_img)

    def _syncmatrix_ij_vote(self, clmatrix, i, j, k_list, n_theta):
        """
        Compute the (i,j) rotation block of the synchronization matrix using voting method

        Given the common lines matrix `clmatrix`, a list of images specified in k_list
        and the number of common lines n_theta, find the (i, j) rotation block (in X and Y)
        of the synchronization matrix.
        :param clmatrix: The common lines matrix
        :param i: The i image
        :param j: The j image
        :param k_list: The list of images for the third image for voting algorithm
        :param n_theta: The number of points in the theta direction (common lines)
        :return: The (i,j) rotation block of the synchronization matrix
        """

        good_k = self._vote_ij(clmatrix, n_theta, i, j, k_list)

        rots = self._rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta)

        if rots.shape[0] > 0:
            rot_mean = np.mean(rots, 0)
            # The error to mean value can be calculated as
            #    rot_block = rots[:2, :2]
            #    diff = rot_block - rot_mean[np.newaxis, :2, :2]
            #    err = np.linalg.norm(diff) / np.linalg.norm(rot_block)
            # if err > tol, this means that images i and j have inconsistent
            # rotations. The original Matlab code tried to print out the information
            # on inconsistent rotations but was commented out and do nothing,
            # probably due to the fact that it will print out a lot of
            # inconsistent rotations if the resolution or number of images
            # are not enough. We choose to pass it as Matlab code.

        else:
            # This for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # Simply put to zero as Matlab code.
            rot_mean = np.zeros((3, 3))

        # return the rotation matrix in X and Y
        r22 = rot_mean[:2, :2]
        return r22

    def _rotratio_eulerangle_vec(self, clmatrix, i, j, good_k, n_theta):
        """
        Compute the rotation that takes image i to image j

        Given a common lines matrix, where the index of each common line
        is in the range of n_theta and a list of good image k from voting results.
        :param clmatrix: The common lines matrix
        :param i: The i image
        :param j: The j image
        :param good_k: The list of good images k from voting algorithm
        :param n_theta: The number of points in the theta direction (common lines)
        :return: The rotation matrix that takes image i to image j for good index of k.
        """

        if i == j:
            return []

        # Prepare the theta values from the differences of common line indices
        # C1, C2, and C3 are unit circles of image i, j, and k
        # cl_diff1 is for the angle on C1 created by its intersection with C3 and C2.
        # cl_diff2 is for the angle on C2 created by its intersection with C1 and C3.
        # cl_diff3 is for the angle on C3 created by its intersection with C2 and C1.
        cl_diff1 = clmatrix[i, good_k] - clmatrix[i, j]  # for theta1
        cl_diff2 = clmatrix[j, good_k] - clmatrix[j, i]  # for - theta2
        cl_diff3 = clmatrix[good_k, j] - clmatrix[good_k, i]  # for theta3

        # Calculate the cos values of rotation angles between i an j images for good k images
        c_alpha, good_idx = self._get_cos_phis(cl_diff1, cl_diff2, cl_diff3, n_theta)
        alpha = np.arccos(c_alpha)

        # Convert the Euler angles with ZXZ conversion to rotation matrices
        angles = np.zeros((alpha.shape[0], 3))
        angles[:, 0] = clmatrix[i, j] * 2 * np.pi / n_theta - np.pi
        angles[:, 1] = alpha
        angles[:, 2] = np.pi - clmatrix[j, i] * 2 * np.pi / n_theta
        r = Rotation.from_euler("ZXZ", angles).as_matrix()

        return r[good_idx, :, :]

    def _vote_ij(self, clmatrix, n_theta, i, j, k_list):
        """
        Apply the voting algorithm for images i and j.

        clmatrix is the common lines matrix, constructed using angular resolution,
        n_theta. k_list are the images to be used for voting of the pair of images
        (i ,j).
        :param clmatrix: The common lines matrix
        :param n_theta: The number of points in the theta direction (common lines)
        :param i: The i image
        :param j: The j image
        :param k_list: The list of images for the third image for voting algorithm
        :return:  good_k, the list of all third images in the peak of the histogram
            corresponding to the pair of images (i,j)
        """

        if i == j or clmatrix[i, j] == -1:
            return []

        # Some of the entries in clmatrix may be zero if we cleared
        # them due to small correlation, or if for each image
        # we compute intersections with only some of the other images.
        #
        # Note that as long as the diagonal of the common lines matrix is
        # -1, the conditions (i != j) && (j != k) are not needed, since
        # if i == j then clmatrix[i, k] == -1 and similarly for i == k or
        # j == k. Thus, the previous voting code (from the JSB paper) is
        # correct even though it seems that we should test also that
        # (i != j) && (i != k) && (j != k), and only (i != j) && (i != k)
        #  as tested there.
        cl_idx12 = clmatrix[i, j]
        cl_idx21 = clmatrix[j, i]
        k_list = k_list[
            (k_list != i) & (clmatrix[i, k_list] != -1) & (clmatrix[j, k_list] != -1)
        ]
        cl_idx13 = clmatrix[i, k_list]
        cl_idx31 = clmatrix[k_list, i]
        cl_idx23 = clmatrix[j, k_list]
        cl_idx32 = clmatrix[k_list, j]

        # Prepare the theta values from the differences of common line indices
        # C1, C2, and C3 are unit circles of image i, j, and k
        # cl_diff1 is for the angle on C1 created by its intersection with C3 and C2.
        # cl_diff2 is for the angle on C2 created by its intersection with C1 and C3.
        # cl_diff3 is for the angle on C3 created by its intersection with C2 and C1.
        cl_diff1 = cl_idx13 - cl_idx12
        cl_diff2 = cl_idx21 - cl_idx23
        cl_diff3 = cl_idx32 - cl_idx31
        # Calculate the cos values of rotation angles between i an j images for good k images
        cos_phi2, good_idx = self._get_cos_phis(cl_diff1, cl_diff2, cl_diff3, n_theta)

        if np.any(np.abs(cos_phi2) - 1 > 1e-12):
            logger.warning(
                f"Globally Consistent Angular Reconstruction (GCAR) exists"
                f" numerical problem: abs(cos_phi2) > 1, with the"
                f" difference of {np.abs(cos_phi2)-1}."
            )
        cos_phi2 = np.clip(cos_phi2, -1, 1)

        # Store angles between i and j induced by each third image k.
        phis = cos_phi2
        # Sore good indices of l in k_list of the image that creates that angle.
        inds = k_list[good_idx]

        if phis.shape[0] == 0:
            return []

        # Parameters used to compute the smoothed angle histogram.
        ntics = 60
        angles_grid = np.linspace(0, 180, ntics, True)
        # Get angles between images i and j for computing the histogram
        angles = np.arccos(phis[:]) * 180 / np.pi
        # Angles that are up to 10 degrees apart are considered
        # similar. This sigma ensures that the width of the density
        # estimation kernel is roughly 10 degrees. For 15 degrees, the
        # value of the kernel is negligible.
        sigma = 3.0

        # Compute the histogram of the angles between images i and j
        squared_values = np.add.outer(np.square(angles), np.square(angles_grid))
        angles_hist = np.sum(
            np.exp(
                (2 * np.multiply.outer(angles, angles_grid) - squared_values)
                / (2 * sigma ** 2)
            ),
            0,
        )

        # We assume that at the location of the peak we get the true angle
        # between images i and j. Find all third images k, that induce an
        # angle between i and j that is at most 10 off the true angle.
        # Even for debugging, don't put a value that is smaller than two
        # tics, since the peak might move a little bit due to wrong k images
        # that accidentally fall near the peak.
        peak_idx = angles_hist.argmax()
        idx = np.abs(angles - angles_grid[peak_idx]) < 360 / ntics
        good_k = inds[idx]
        return good_k.astype("int")

    def _get_cos_phis(self, cl_diff1, cl_diff2, cl_diff3, n_theta):
        """
        Calculate cos values of rotation angles between i and j images

        Given C1, C2, and C3 are unit circles of image i, j, and k, compute
        resulting cos values of rotation angles between i an j images when both
        of them are intersecting with k.

        To ensure that the smallest singular value is big enough, controlled by
        the determinant of the matrix,
           C=[  1  c1  c2 ;
               c1   1  c3 ;
               c2  c3   1 ],
        we therefore use the condition below
               1+2*c1*c2*c3-(c1^2+c2^2+c3^2) > 1.0e-5,
        so the matrix is far from singular.

        :param cl_diff1: Difference of common line indices on C1 created by
            its intersection with C3 and C2
        :param cl_diff2: Difference of common line indices on C2 created by
            its intersection with C1 and C3
        :param cl_diff3: Difference of common line indices on C3 created by
            its intersection with C2 and C1
        :param n_theta: The number of points in the theta direction (common lines)
        :return: cos values of rotation angles between i and j images
            and indices for good k
        """

        # Calculate the theta values from the differences of common line indices
        # C1, C2, and C3 are unit circles of image i, j, and k
        # theta1 is the angle on C1 created by its intersection with C3 and C2.
        # theta2 is the angle on C2 created by its intersection with C1 and C3.
        # theta3 is the angle on C3 created by its intersection with C2 and C1.
        theta1 = cl_diff1 * 2 * np.pi / n_theta
        theta2 = cl_diff2 * 2 * np.pi / n_theta
        theta3 = cl_diff3 * 2 * np.pi / n_theta

        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c3 = np.cos(theta3)

        # Each common-line corresponds to a point on the unit sphere. Denote the
        # coordinates of these points by (Pix, Piy Piz), and put them in the matrix
        #   M=[ P1x  P2x  P3x ;
        #       P1y  P2y  P3y ;
        #       P1z  P2z  P3z ].
        #
        # Then the matrix
        #   C=[  1  c1  c2 ;
        #       c1   1  c3 ;
        #       c2  c3   1 ],
        # where c1, c2, c3 are given above, is given by C = M.T @ M.
        # For the points P1, P2, and P3 to form a triangle on the unit sphere, a
        # necessary and sufficient condition is for C to be positive definite. This
        # is equivalent to
        #       1+2*c1*c2*c3-(c1^2+c2^2+c3^2) > 0.
        # However, this may result in a triangle that is too flat, that is, the
        # angle between the projections is very close to zero. We therefore use the
        # condition below
        #       1+2*c1*c2*c3-(c1^2+c2^2+c3^2) > 1.0e-5.
        # This ensures that the smallest singular value (which is actually
        # controlled by the determinant of C) is big enough, so the matrix is far
        # from singular. This condition is equivalent to computing the singular
        # values of C, followed by checking that the smallest one is big enough.

        cond = 1 + 2 * c1 * c2 * c3 - (np.square(c1) + np.square(c2) + np.square(c3))
        good_idx = np.nonzero(cond > 1e-5)[0]

        # Calculated cos values of angle between i and j images
        cos_phi2 = (c3[good_idx] - c1[good_idx] * c2[good_idx]) / (
            np.sin(theta1[good_idx]) * np.sin(theta2[good_idx])
        )
        return cos_phi2, good_idx
