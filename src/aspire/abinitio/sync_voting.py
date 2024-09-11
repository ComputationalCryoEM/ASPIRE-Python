import logging

import numpy as np

from aspire.utils import Rotation

logger = logging.getLogger(__name__)


class SyncVotingMixin(object):
    """
    SyncVotingMixin is a mixin implementing methods for the synchronization voting algorithm
    which are shared by CLSynVoting and CLSymmetryC3C4
    """

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
        cl_diff2 = clmatrix[j, good_k] - clmatrix[j, i]  # for theta2
        cl_diff3 = clmatrix[good_k, j] - clmatrix[good_k, i]  # for theta3

        # Calculate the cos values of rotation angles between i an j images for good k images
        c_alpha, good_idx = self._get_cos_phis(
            cl_diff1, cl_diff2, cl_diff3, n_theta, sync=False
        )

        if len(c_alpha) == 0:
            return None
        alpha = np.arccos(c_alpha)

        # Convert the Euler angles with ZYZ conversion to rotation matrices
        angles = np.zeros((alpha.shape[0], 3))
        angles[:, 0] = clmatrix[i, j] * 2 * np.pi / n_theta + np.pi / 2
        angles[:, 1] = alpha
        angles[:, 2] = -np.pi / 2 - clmatrix[j, i] * 2 * np.pi / n_theta
        r = Rotation.from_euler(angles).matrices

        return r[good_idx, :, :]

    def _vote_ij(self, clmatrix, n_theta, i, j, k_list, sync=False):
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
        :param sync: Perform 180 degree ambiguity synchronization.
        :return: (alpha, good_k), angles and list of all third images
            in the peak of the histogram corresponding to the pair of
            images (i,j)
        """

        if i == j or clmatrix[i, j] == -1:
            return None, []

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
        cl_diff2 = cl_idx23 - cl_idx21
        cl_diff3 = cl_idx32 - cl_idx31

        # Calculate the cos values of rotation angles between i an j images for good k images
        cos_phi2, good_idx = self._get_cos_phis(
            cl_diff1, cl_diff2, cl_diff3, n_theta, sync=sync
        )

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
            return None, []

        # Parameters used to compute the smoothed angle histogram.
        ntics = int(180 / self.hist_bin_width)
        angles_grid = np.linspace(0, 180, ntics + 1, True)

        # Get angles between images i and j for computing the histogram
        angles = np.arccos(phis[:]) * 180 / np.pi

        # Angles that are up to 10 degrees apart are considered
        # similar. This sigma ensures that the width of the density
        # estimation kernel is roughly 10 degrees. For 15 degrees, the
        # value of the kernel is negligible.
        sigma = 3.0

        # Compute the histogram of the angles between images i and j
        angles_distances = angles_grid[None, :] - angles[:, None]
        angles_hist = np.sum(np.exp(-(angles_distances**2) / (2 * sigma**2)), axis=0)

        # We assume that at the location of the peak we get the true angle
        # between images i and j. Find all third images k, that induce an
        # angle between i and j that is at most 10 off the true angle.
        # Even for debugging, don't put a value that is smaller than two
        # tics, since the peak might move a little bit due to wrong k images
        # that accidentally fall near the peak.
        peak_idx = angles_hist.argmax()

        if str(self.full_width).lower() == "adaptive":
            # Adaptive width  (MATLAB)
            # Look for the estimations in the peak of the histogram
            w_theta_needed = 0
            idx = []
            while sum(idx) == 0:
                w_theta_needed += self.hist_bin_width  # widen peak as needed
                idx = np.abs(angles - angles_grid[peak_idx]) < w_theta_needed
            if w_theta_needed > self.hist_bin_width:
                logger.info(
                    f"Adaptive width {w_theta_needed} required for ({i},{j}), found {sum(idx)} indices."
                )
        else:
            # Fixed width
            idx = np.abs(angles - angles_grid[peak_idx]) < self.full_width

        good_k = inds[idx]
        alpha = np.arccos(phis[idx])

        return alpha, good_k.astype("int")

    def _get_cos_phis(self, cl_diff1, cl_diff2, cl_diff3, n_theta, sync=False):
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
        :param sync: Perform 180 degree ambiguity synchronization.
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
        if sync:
            # MATLAB
            cos_phi2 = (c3[good_idx] - c1[good_idx] * c2[good_idx]) / (
                np.sqrt(1 - c1[good_idx] ** 2) * np.sqrt(1 - c2[good_idx] ** 2)
            )

            #  Some synchronization must be applied when common line is
            #  out by 180 degrees.
            #  Here fix the angles between c_ij(c_ji) and c_ik(c_jk) to be smaller than pi/2,
            #  otherwise there will be an ambiguity between alpha and pi-alpha.
            TOL_idx = 1e-12

            # Select only good_idx
            theta1 = theta1[good_idx]
            theta2 = theta2[good_idx]
            theta3 = theta3[good_idx]

            # Check sync conditions
            ind1 = (theta1 > (np.pi + TOL_idx)) | (
                (theta1 < -TOL_idx) & (theta1 > -np.pi)
            )
            ind2 = (theta2 > (np.pi + TOL_idx)) | (
                (theta2 < -TOL_idx) & (theta2 > -np.pi)
            )
            align180 = (ind1 & ~ind2) | (~ind1 & ind2)

            # Apply sync
            cos_phi2[align180] = -cos_phi2[align180]
        else:
            # Python
            cos_phi2 = (c3[good_idx] - c1[good_idx] * c2[good_idx]) / (
                np.sin(theta1[good_idx]) * np.sin(theta2[good_idx])
            )

        return cos_phi2, good_idx
