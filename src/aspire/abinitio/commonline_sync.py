import logging

import numpy as np

from aspire.abinitio import CLOrient3D, SyncVotingMixin
from aspire.utils import nearest_rotations
from aspire.utils.matlab_compat import stable_eigsh

logger = logging.getLogger(__name__)


class CLSyncVoting(CLOrient3D, SyncVotingMixin):
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

    def __init__(
        self,
        src,
        n_rad=None,
        n_theta=360,
        max_shift=0.15,
        shift_step=1,
        hist_bin_width=3,
        full_width=6,
        mask=True,
    ):
        """
        Initialize an object for estimating 3D orientations using synchronization matrix

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction.
            Default is 360.
        :param max_shift: Determines maximum range for shifts as a proportion
            of the resolution. Default is 0.15.
        :param shift_step: Resolution for shift estimation in pixels. Default is 1 pixel.
        :param hist_bin_width: Bin width in smoothing histogram (degrees).
        :param full_width: Selection width around smoothed histogram peak (degrees).
            `adaptive` will attempt to automatically find the smallest number of
            `hist_bin_width`s required to find at least one valid image index.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """
        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            hist_bin_width=hist_bin_width,
            full_width=full_width,
            mask=mask,
        )
        self.syncmatrix = None

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images using synchronization matrix
        """
        if self.syncmatrix is None:
            self.syncmatrix_vote()

        S = self.syncmatrix
        sz = S.shape
        assert sz[0] == sz[1], "syncmatrix must be a square matrix."
        assert sz[0] % 2 == 0, "syncmatrix must be a square matrix of size 2Kx2K."

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
        truncated_equations = np.zeros((3 * n_img, 6), dtype=self.dtype)
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
        ATA = np.zeros((3, 3), dtype=self.dtype)
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

        rotations = np.empty((n_img, 3, 3), dtype=self.dtype)
        rotations[:, :, 0] = r1.T
        rotations[:, :, 1] = r2.T
        rotations[:, :, 2] = r3.T
        # Make sure that we got rotations by enforcing R to be
        # a rotation (in case the error is large)
        rotations = nearest_rotations(rotations)

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

        assert sz[0] == sz[1], "clmatrix must be a square matrix."

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

        _, good_k = self._vote_ij(clmatrix, n_theta, i, j, k_list)

        rots = self._rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta)

        if rots is not None:
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
