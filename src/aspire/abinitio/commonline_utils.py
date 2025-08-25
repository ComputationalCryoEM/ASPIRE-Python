import logging

import numpy as np
from numpy.linalg import eigh, norm

from aspire.operators import PolarFT
from aspire.utils import Rotation, all_pairs, anorm, tqdm

logger = logging.getLogger(__name__)


def estimate_third_rows(vijs, viis):
    """
    Find the third row of each rotation matrix given a collection of matrices
    representing the outer products of the third rows from each rotation matrix.

    :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds the third rows
    outer product of the rotation matrices Ri and Rj.

    :param viis: An n_imgx3x3 array where the i'th 3x3 slice holds the outer product of
    the third row of Ri with itself.

    :param order: The underlying molecular symmetry.

    :return: vis, An n_imgx3 matrix whose i'th row is the third row of the rotation matrix Ri.
    """

    n_img = viis.shape[0]

    # Build matrix V whose (i,j)-th block of size 3x3 holds the outer product vij
    V = np.zeros((n_img, n_img, 3, 3), dtype=vijs.dtype)

    # All pairs (i,j) where i<j
    pairs = all_pairs(n_img)

    # Populate upper triangle of V with vijs and lower triangle with vjis, where vji = vij^T.
    for idx, (i, j) in enumerate(pairs):
        V[i, j] = vijs[idx]
        V[j, i] = vijs[idx].T

    # Populate diagonal of V with viis
    for i, vii in enumerate(viis):
        V[i, i] = vii

    # Permute axes and reshape to (3 * n_img, 3 * n_img).
    V = np.swapaxes(V, 1, 2).reshape(3 * n_img, 3 * n_img)

    # In a clean setting V is of rank 1 and its eigenvector is the concatenation
    # of the third rows of all rotation matrices.
    # In the noisy setting we use the eigenvector corresponding to the leading eigenvalue
    val, vec = eigh(V)
    lead_idx = np.argmax(val)
    lead_vec = vec[:, lead_idx]

    # We decompose the leading eigenvector and normalize to obtain the third rows, vis.
    vis = lead_vec.reshape((n_img, 3))
    vis /= anorm(vis, axes=(-1,))[:, np.newaxis]

    return vis


def estimate_inplane_rotations(cl_class, vis):
    """
    Estimate the rotation matrices for each image by constructing arbitrary rotation matrices
    populated with the given third rows, vis, and then rotating by an appropriate in-plane rotation.

    :cl_class: A commonlines class instance.
    :param vis: An n_imgx3 array where the i'th row holds the estimate for the third row of
    the i'th rotation matrix.

    :return: Rotation matrices Ris and in-plane rotation matrices R_thetas, both size n_imgx3x3.
    """
    pf = cl_class.pf
    n_img = cl_class.n_img
    n_theta = cl_class.n_theta
    max_shift_1d = cl_class.max_shift
    shift_step = cl_class.shift_step
    order = cl_class.order
    degree_res = cl_class.degree_res

    # Step 1: Construct all rotation matrices Ri_tildes whose third rows are equal to
    # the corresponding third rows vis.
    Ri_tildes = complete_third_row_to_rot(vis)

    # Step 2: Construct all in-plane rotation matrices, R_theta_ijs.
    max_angle = (360 // order) * order
    theta_ijs = np.arange(0, max_angle, degree_res) * np.pi / 180
    R_theta_ijs = Rotation.about_axis("z", theta_ijs, dtype=cl_class.dtype).matrices

    # Step 3: Compute the correlation over all shifts.
    # Generate shifts.
    r_max = pf.shape[-1]
    shifts, shift_phases, _ = cl_class._generate_shift_phase_and_filter(
        r_max, max_shift_1d, shift_step
    )
    n_shifts = len(shifts)

    # Q is the n_img x n_img  Hermitian matrix defined by Q = q*q^H,
    # where q = (exp(i*order*theta_0), ..., exp(i*order*theta_{n_img-1}))^H,
    # and theta_i in [0, 2pi/order) is the in-plane rotation angle for the i'th image.
    Q = np.zeros((n_img, n_img), dtype=complex)

    # Reconstruct the full polar Fourier for use in correlation. cl_class.pf only consists of
    # rays in the range [180, 360), with shape (n_img, n_theta//2, n_rad-1).
    pf = PolarFT.half_to_full(pf)

    # Normalize rays.
    pf /= norm(pf, axis=-1)[..., np.newaxis]

    n_pairs = n_img * (n_img - 1) // 2
    with tqdm(total=n_pairs) as pbar:
        idx = 0
        # Note: the ordering of i and j in these loops should not be changed as
        # they correspond to the ordered tuples (i, j), for i<j.
        for i in range(n_img):
            pf_i = pf[i]

            # Generate shifted versions of images.
            pf_i_shifted = np.array(
                [pf_i * shift_phase for shift_phase in shift_phases]
            )

            Ri_tilde = Ri_tildes[i]

            for j in range(i + 1, n_img):
                pf_j = pf[j]

                Rj_tilde = Ri_tildes[j]

                # Compute all possible rotations between the i'th and j'th images.
                Us = np.array(
                    [Ri_tilde.T @ R_theta_ij @ Rj_tilde for R_theta_ij in R_theta_ijs]
                )

                # Find the angle between common lines induced by the rotations.
                c1s = np.array([[-U[1, 2], U[0, 2]] for U in Us])
                c2s = np.array([[U[2, 1], -U[2, 0]] for U in Us])

                # Convert from angles to indices.
                c1s = cl_angles_to_ind(c1s, n_theta)
                c2s = cl_angles_to_ind(c2s, n_theta)

                # Perform correlation, corrs is shape n_shifts x len(theta_ijs).
                corrs = np.array(
                    [
                        np.dot(pf_i_shift[c1], np.conj(pf_j[c2]))
                        for pf_i_shift in pf_i_shifted
                        for c1, c2 in zip(c1s, c2s)
                    ]
                )

                # Reshape to group by shift and symmetric order.
                corrs = corrs.reshape((n_shifts, order, len(theta_ijs) // order))

                # For each pair of lines we take the maximum correlation over all shifts.
                corrs = np.max(np.real(corrs), axis=0)

                # corrs[i] is the set of correlations for theta_ij in [2pi * i / order, 2pi * (i + 1) / order).
                # Due to symmetry, each corrs[i] represents correlations over identical pairs of lines.
                # With that in mind, we average over corrs[i] and find the max correlation.
                # This produces an index corresponding to theta_ij in the range [0, 2pi/order).
                corrs = np.mean(np.real(corrs), axis=0)
                max_idx_corr = np.argmax(corrs)

                theta_ij = degree_res * max_idx_corr * np.pi / 180

                Q[i, j] = np.exp(-1j * order * theta_ij)

                pbar.update()

                idx += 1

        # Populate the lower triangle and diagonal of Q.
        # Diagonals are 1 since e^{i*0}=1.
        Q += np.conj(Q).T + np.eye(n_img)

        # Q is a rank-1 Hermitian matrix.
        eig_vals, eig_vecs = eigh(Q)
        leading_eig_vec = eig_vecs[:, -1]
        logger.info(f"Top 3 eigenvalues of Q (rank-1) are {str(eig_vals[-3:][::-1])}.")

        # Calculate R_thetas.
        R_thetas = Rotation.about_axis("z", np.angle(leading_eig_vec ** (1 / order)))

        # Form rotation matrices Ris.
        Ris = R_thetas @ Ri_tildes

        return Ris


def complete_third_row_to_rot(r3):
    """
    Construct rotation matrices whose third rows are equal to the given row vectors.
    For vector r3 = [a, b, c], where [a, b, c] != [0, 0, 1], we return the matrix
    with rows r1, r2, r3, given by:

    r1 = 1/sqrt(a^2 + b^2)[b, -a, 0],
    r2 = 1/sqrt(a^2 + b^2)[ac, bc, -(a^2 + b^2)].

    :param r3: A nx3 array where each row vector has norm 1.
    :return: An nx3x3 array of rotation matrices whose third rows are r3.
    """

    # Handle singleton vector.
    singleton = False
    if r3.shape == (3,):
        r3 = np.expand_dims(r3, axis=0)
        singleton = True

    # Initialize output rotation matrices.
    rots = np.zeros((len(r3), 3, 3), dtype=r3.dtype)

    # Populate 3rd rows.
    rots[:, 2] = r3

    # Mask for third rows that do not coincide with the z-axis.
    mask = np.linalg.norm(r3 - [0, 0, 1], axis=1) >= 1e-5

    # If the third row coincides with the z-axis we return the identity matrix.
    rots[~mask] = np.eye(3, dtype=r3.dtype)

    # 'norm_12' is non-zero since r3 does not coincide with the z-axis.
    norm_12 = np.sqrt(r3[mask, 0] ** 2 + r3[mask, 1] ** 2)

    # Populate 1st rows with vector orthogonal to row 3.
    rots[mask, 0, 0] = r3[mask, 1] / norm_12
    rots[mask, 0, 1] = -r3[mask, 0] / norm_12

    # Populate 2nd rows such that r3 = r1 x r2
    rots[mask, 1, 0] = r3[mask, 0] * r3[mask, 2] / norm_12
    rots[mask, 1, 1] = r3[mask, 1] * r3[mask, 2] / norm_12
    rots[mask, 1, 2] = -norm_12

    if singleton:
        rots = rots.reshape(3, 3)

    return rots


def cl_angles_to_ind(cl_angles, n_theta):
    thetas = np.arctan2(cl_angles[:, 1], cl_angles[:, 0])

    # Shift from [-pi,pi] to [0,2*pi).
    thetas = np.mod(thetas, 2 * np.pi)

    # linear scale from [0,2*pi) to [0,n_theta).
    ind = np.mod(np.round(thetas / (2 * np.pi) * n_theta), n_theta).astype(int)

    # Return scalar for single value.
    if ind.size == 1:
        ind = ind.flat[0]

    return ind
