import logging

import numpy as np

from aspire.abinitio import CLOrient3D
from aspire.abinitio.sync_voting import (
    _rotratio_eulerangle_vec,
    _syncrotations,
    _vote_ij,
)

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
        disable_gpu=False,
        **kwargs,
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
        :param disable_gpu: Disables GPU acceleration;
            forces CPU only code for this module.
            Defaults to automatically using GPU when available.
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
            disable_gpu=disable_gpu,
            **kwargs,
        )
        self.syncmatrix = None

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images using synchronization matrix
        """
        if self.syncmatrix is None:
            self.syncmatrix_vote()

        self.rotations = _syncrotations(self.syncmatrix)

    def syncmatrix_vote(self):
        """
        Construct the synchronization matrix using voting method

        A pre-computed common line matrix is required as input.
        """

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

        _, good_k = _vote_ij(
            clmatrix, n_theta, i, j, k_list, self.hist_bin_width, self.full_width
        )

        rots = _rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta)

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
