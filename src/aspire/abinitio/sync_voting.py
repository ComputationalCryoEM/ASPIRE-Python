import logging

import numpy as np

from aspire.utils import J_conjugate, Rotation, nearest_rotations
from aspire.utils.matlab_compat import stable_eigsh

logger = logging.getLogger(__name__)


def _syncmatrix_ij_vote_3n(
    clmatrix, i, j, k_list, n_theta, hist_bin_width, full_width, sigma=3.0
):
    """
    Compute the (i,j) rotation block of the synchronization matrix using voting method

    Given the common lines matrix `clmatrix`, a list of images specified in k_list
    and the number of common lines n_theta, find the (i, j) rotation block Rij.

    :param clmatrix: The common lines matrix
    :param i: The i image
    :param j: The j image
    :param k_list: The list of images for the third image for voting algorithm
    :param n_theta: The number of points in the theta direction (common lines)
    :param hist_bin_width: Bin width in smoothing histogram (degrees).
    :param full_width: Selection width around smoothed histogram peak (degrees).
        `adaptive` will attempt to automatically find the smallest number of
        `hist_bin_width`s required to find at least one valid image index.
    :param sigma: Voting contribution smoothing factor. Default is 3.0.

    :return: The (i,j) rotation block of the synchronization matrix
    """
    alphas, good_k = _vote_ij(
        clmatrix,
        n_theta,
        i,
        j,
        k_list,
        hist_bin_width,
        full_width,
        sigma=sigma,
        sync=True,
    )

    angles = np.zeros(3)

    # Note, len(alphas) case covers when no acceptable hist bin was found.
    if (alphas is not None) and len(alphas) > 0:
        angles[0] = clmatrix[i, j] * 2 * np.pi / n_theta + np.pi / 2
        angles[1] = np.mean(alphas)
        angles[2] = -np.pi / 2 - clmatrix[j, i] * 2 * np.pi / n_theta
        rot = Rotation.from_euler(angles).matrices

    else:
        # This is for the case that images i and j correspond to the same
        # viewing direction and differ only by in-plane rotation.
        # We set to zero as in the Matlab code.
        rot = np.zeros((3, 3))

    return rot


def _syncrotations(S):
    """
    Compute the rotations from the syncronization matrix S.

    :param S: A 2Kx2K synchronization matrix.

    :return: Kx3x3 rotations.
    """
    sz = S.shape
    dtype = S.dtype
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
    logger.info(f"Top 10 eigenvalues from synchronization voting matrix: {d[sort_idx]}")

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
    truncated_equations = np.zeros((3 * n_img, 6), dtype=dtype)
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
    ATA = np.zeros((3, 3), dtype=dtype)
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

    rotations = np.empty((n_img, 3, 3), dtype=dtype)
    rotations[:, :, 0] = r1.T
    rotations[:, :, 1] = r2.T
    rotations[:, :, 2] = r3.T

    # Make sure that we got rotations by enforcing R to be
    # a rotation (in case the error is large)
    return nearest_rotations(rotations)


def _rotratio_eulerangle(clmatrix, n_theta):
    """
    Given a 3x3 common lines matrix, where the index of each common line is
    between 1 and n_theta, compute the rotation that takes image 1 to image 2.

    :param clmatrix: A 3-by-3 common lines matrix.
    :param n_theta: The number of points in the theta direction (common lines).

    :return: A 3-by-3 rotation matrix taking the first image to the second,
        or ``None`` if the common lines form a degenerate triangle.
    """

    # Prepare the theta values from the differences of common-line indices.
    #
    # These correspond to idx3, idx2, and idx1, respectively, in the
    # original MATLAB implementation.
    cl_diff1 = np.asarray([clmatrix[0, 2] - clmatrix[0, 1]])
    cl_diff2 = np.asarray([clmatrix[1, 2] - clmatrix[1, 0]])
    cl_diff3 = np.asarray([clmatrix[2, 1] - clmatrix[2, 0]])

    # Calculate the cos values of rotation angles between the two images.
    c_alpha, _ = _get_cos_phis(
        cl_diff1,
        cl_diff2,
        cl_diff3,
        n_theta,
        sync=False,
    )

    if len(c_alpha) == 0:
        return None

    alpha = np.arccos(c_alpha[0])

    # Convert the Euler angles with ZYZ conversion to rotation matrices
    angles = np.zeros(3)
    angles[0] = clmatrix[0, 1] * 2 * np.pi / n_theta + np.pi / 2
    angles[1] = alpha
    angles[2] = -np.pi / 2 - clmatrix[1, 0] * 2 * np.pi / n_theta

    return Rotation.from_euler(angles).matrices


def _syncmatrix(clmatrix, n_theta, dtype):
    """
    Construct the CryoEM synchronization matrix, given a common lines matrix
    clmatrix, that was constructed using angular resolution of n_theta radial
    lines per image.

    :param clmatrix: A commonline matrix
    :param n_theta: The number of points in the theta direction (common lines).
    :param dtype: dtype of the synchronization matrix.

    :return: The 2N-by-2N synchronization matrix, where N is the number of
        images.
    """
    if clmatrix.ndim != 2 or clmatrix.shape[0] != clmatrix.shape[1]:
        raise ValueError("clmatrix must be a square matrix.")

    n_img = clmatrix.shape[0]

    # This tolerance is relevant only if n_theta is very large (1.0e15),
    # to check for accuracy. Otherwise, ignore it and the resulting output.
    tol = 1.0e-12

    # S is initialized with identity 2-by-2 diagonal blocks.
    #
    # Use an n_img-by-2-by-n_img-by-2 view so that synchronization blocks
    # can be addressed directly as S[i, :, j, :].
    syncmatrix = np.eye(
        2 * n_img,
        dtype=dtype,
    ).reshape(n_img, 2, n_img, 2)

    for i in range(n_img - 1):
        for j in range(i + 1, n_img):
            # Each triplet (i, j, k) defines some rotation matrix from
            # image i to image j. Not all images k give rise to such a
            # rotation since, for example, the resulting triangle can be
            # too small.
            #
            # Store all valid rotation matrices. This is not strictly
            # necessary, as the matrices could be processed on the fly.
            rotations = []

            for k in range(n_img):
                if k == i or k == j:
                    continue

                indices = [i, j, k]
                cl_triplet = clmatrix[indices][:, indices]

                rot = _rotratio_eulerangle(cl_triplet, n_theta)

                # _rotratio_eulerangle returns None when the common lines
                # form a degenerate triangle.
                if rot is not None:
                    # Average over the handedness ambiguity.
                    rot = (rot + J_conjugate(rot)) / 2
                    rotations.append(rot)

            # Merge all rotations computed by all triplets (i, j, k) into
            # a single rotation that takes the frame of image i into the
            # frame of image j.
            if rotations:
                rotations = np.stack(rotations)
                rot_mean = np.mean(rotations, axis=0)

                # Preserve the consistency diagnostic from the MATLAB code.
                diff = rotations - rot_mean
                denominator = np.linalg.norm(rotations)

                err = np.linalg.norm(diff) / denominator
                if err > tol:
                    logger.debug(
                        f"Inconsistent rotations: [i={i}, j={j}] "
                        f"err={err:e}, tol={tol:e}"
                    )
            else:
                # Images i and j may correspond to the same viewing
                # direction and differ only by an in-plane rotation. No
                # triangle can then be formed using any third image k.
                #
                # The MATLAB implementation optionally uses reference
                # rotations here. Reference rotations are not available to
                # the estimator, so put zero.
                rot_mean = np.zeros((3, 3), dtype=dtype)

            # S is a 2N-by-2N matrix containing N-by-N blocks of size 2-by-2.
            # The (i, j) block is
            #
            #     [r11 r12]
            #     [r21 r22],
            #
            # where r_kl is the inner product of column k of rotation i and
            # column l of rotation j.
            #
            # To extract the required block, note that the handedness-
            # averaged relative rotation has the form
            #
            #     [r11 r12  0]
            #     [r21 r22  0]
            #     [  0   0 r33].
            rot_block = rot_mean[:2, :2]

            # Put the block in the correct location in S and its transpose
            # in the symmetric location.
            syncmatrix[i, :, j, :] = rot_block
            syncmatrix[j, :, i, :] = rot_block.T

    return syncmatrix.reshape(2 * n_img, 2 * n_img)


def _rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta):
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
    c_alpha, good_idx = _get_cos_phis(cl_diff1, cl_diff2, cl_diff3, n_theta, sync=False)

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


def _vote_ij(
    clmatrix, n_theta, i, j, k_list, hist_bin_width, full_width, sigma=3.0, sync=False
):
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
    :param hist_bin_width: Bin width in smoothing histogram (degrees).
    :param full_width: Selection width around smoothed histogram peak (degrees).
        `adaptive` will attempt to automatically find the smallest number of
        `hist_bin_width`s required to find at least one valid image index.
    :param sigma: Voting contribution smoothing factor. Default is 3.0.
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
    cos_phi2, good_idx = _get_cos_phis(cl_diff1, cl_diff2, cl_diff3, n_theta, sync=sync)

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
    ntics = int(180 / hist_bin_width)
    angles_grid = np.linspace(0, 180, ntics + 1, True)

    # Get angles between images i and j for computing the histogram
    angles = np.arccos(phis[:]) * 180 / np.pi

    # Angles that are up to 10 degrees apart are considered
    # similar. `sigma` ensures that the width of the density
    # estimation kernel is roughly 10 degrees. For 15 degrees, the
    # value of the kernel is negligible.

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

    if full_width == -1:
        # Adaptive width  (MATLAB)
        # Look for the estimations in the peak of the histogram
        w_theta_needed = 0
        idx = []
        while sum(idx) == 0:
            w_theta_needed += hist_bin_width  # widen peak as needed
            idx = np.abs(angles - angles_grid[peak_idx]) < w_theta_needed
        if w_theta_needed > hist_bin_width:
            logger.info(
                f"Adaptive width {w_theta_needed} required for ({i},{j}), found {sum(idx)} indices."
            )
    else:
        # Fixed width
        idx = np.abs(angles - angles_grid[peak_idx]) < full_width

    good_k = inds[idx]
    alpha = np.arccos(phis[idx])

    return alpha, good_k.astype("int")


def _get_cos_phis(cl_diff1, cl_diff2, cl_diff3, n_theta, sync=False):
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
        ind1 = (theta1 > (np.pi + TOL_idx)) | ((theta1 < -TOL_idx) & (theta1 > -np.pi))
        ind2 = (theta2 > (np.pi + TOL_idx)) | ((theta2 < -TOL_idx) & (theta2 > -np.pi))
        align180 = (ind1 & ~ind2) | (~ind1 & ind2)

        # Apply sync
        cos_phi2[align180] = -cos_phi2[align180]
    else:
        # Python
        cos_phi2 = (c3[good_idx] - c1[good_idx] * c2[good_idx]) / (
            np.sin(theta1[good_idx]) * np.sin(theta2[good_idx])
        )

    return cos_phi2, good_idx
