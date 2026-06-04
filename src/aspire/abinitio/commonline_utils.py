import logging

import numpy as np
from numpy.linalg import eigh, norm

from aspire.operators import PolarFT
from aspire.utils import J_conjugate, Rotation, all_pairs, anorm, cyclic_rotations, tqdm
from aspire.volume import CnSymmetryGroup, SymmetryGroup

logger = logging.getLogger(__name__)


def _estimate_third_rows(vijs, viis):
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
    vis /= anorm(vis, axes=(-1,))[:, None]

    return vis


def _generate_shift_phase_and_filter(r_max, max_shift, shift_step, dtype):
    """
    Prepare the shift phases and generate filter for common-line detection

    The shift phases are pre-defined in a range of max_shift that can be
    applied to maximize the common line calculation. The common-line filter
    is also applied to the radial direction for easier detection.

    :param r_max: Maximum index for common line detection.
    :param max_shift: Maximum value of 1D shift (in pixels) to search.
    :param shift_step: Resolution of shift estimation in pixels.
    :param dtype: dtype for shift phases and filter.
    :return: shift phases matrix and common lines filter.
    """

    # Number of shifts to try
    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))

    # only half of ray, excluding the DC component.
    rk = np.arange(1, r_max + 1, dtype=dtype)

    # Generate all shift phases
    shifts = -max_shift + shift_step * np.arange(n_shifts, dtype=dtype)
    shift_phases = np.exp(np.outer(shifts, -2 * np.pi * 1j * rk / (2 * r_max + 1)))
    # Set filter for common-line detection
    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * (r_max / 4) ** 2))

    return shifts, shift_phases, h


def _estimate_inplane_rotations(vis, pf, max_shift, shift_step, order, degree_res):
    """
    Estimate the rotation matrices for each image of a cyclically symmetric molecule by
    constructing arbitrary rotation matrices populated with the given third rows, vis, and
    then rotating by an appropriate in-plane rotation.

    :param vis: An n_imgx3 array where the i'th row holds the estimate for the third row of
        the i'th rotation matrix.
    :param pf: The polar Fourier transform of the source images, shape (n_img, n_theta/2, n_rad).
    :param max_shift: Maximum range for shifts (in pixels) for estimating in-plane rotations.
    :param shift_step: Shift step (in pixels) for estimating in-plane rotations.
    :param order: Cyclic order.
    :param degree_res: Resolution (in degrees) of in-plane rotation to search over.
    :return: Rotation matrices Ris and in-plane rotation matrices R_thetas, both size n_imgx3x3.
    """
    n_img = vis.shape[0]
    dtype = vis.dtype
    n_theta = pf.shape[1] * 2

    # Step 1: Construct all rotation matrices Ri_tildes whose third rows are equal to
    # the corresponding third rows vis.
    Ri_tildes = _complete_third_row_to_rot(vis)

    # Step 2: Construct all in-plane rotation matrices, R_theta_ijs.
    max_angle = (360 // order) * order
    theta_ijs = np.arange(0, max_angle, degree_res) * np.pi / 180
    R_theta_ijs = Rotation.about_axis("z", theta_ijs, dtype=dtype).matrices

    # Step 3: Compute the correlation over all shifts.
    # Generate shifts.
    r_max = pf.shape[-1]
    shifts, shift_phases, _ = _generate_shift_phase_and_filter(
        r_max, max_shift, shift_step, dtype
    )
    n_shifts = len(shifts)

    # Q is the n_img x n_img  Hermitian matrix defined by Q = q*q^H,
    # where q = (exp(i*order*theta_0), ..., exp(i*order*theta_{n_img-1}))^H,
    # and theta_i in [0, 2pi/order) is the in-plane rotation angle for the i'th image.
    Q = np.zeros((n_img, n_img), dtype=complex)

    # Reconstruct the full polar Fourier for use in correlation. pf only consists of
    # rays in the range [180, 360), with shape (n_img, n_theta//2, n_rad-1).
    pf = PolarFT.half_to_full(pf)

    # Normalize rays.
    pf /= norm(pf, axis=-1)[..., None]

    n_pairs = n_img * (n_img - 1) // 2
    pbar = tqdm(total=n_pairs)
    # Note: the ordering of i and j in these loops should not be changed as
    # they correspond to the ordered tuples (i, j), for i<j.
    for i in range(n_img):
        pf_i = pf[i]

        # Generate shifted versions of images.
        pf_i_shifted = pf_i[None] * shift_phases[:, None]

        Ri_tilde = Ri_tildes[i]

        # Compute part of Ri_tilde.T[None] @ R_theta_ijs @ Rj_tilde[None]
        partial_prod = Ri_tilde.T[None] @ R_theta_ijs

        for j in range(i + 1, n_img):
            pf_j = pf[j]

            Rj_tilde = Ri_tildes[j]

            # Compute all possible rotations between the i'th and j'th images.
            Us = partial_prod @ Rj_tilde[None]

            # Find the angle between common lines induced by the rotations.
            c1s = np.array([-Us[:, 1, 2], Us[:, 0, 2]]).T
            c2s = np.array([Us[:, 2, 1], -Us[:, 2, 0]]).T

            # Convert from angles to indices.
            c1s = _cl_angles_to_ind(c1s, n_theta)
            c2s = _cl_angles_to_ind(c2s, n_theta)

            # Perform correlation, corrs is shape n_shifts x len(theta_ijs).
            pf_i_sel = pf_i_shifted[:, c1s, :]  # (n_shifts, n_angles, n_rad)
            pf_j_sel = (pf_j[c2s, :]).conj()  # (n_angles, n_rad)
            corrs = (pf_i_sel * pf_j_sel[None, ...]).sum(axis=-1)

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
    pbar.close()

    # Populate the lower triangle and diagonal of Q.
    # Diagonals are 1 since e^{i*0}=1.
    Q += Q.conj().T + np.eye(n_img)

    # Q is a rank-1 Hermitian matrix.
    eig_vals, eig_vecs = eigh(Q)
    leading_eig_vec = eig_vecs[:, -1]
    logger.info(f"Top 3 eigenvalues of Q (rank-1) are {str(eig_vals[-3:][::-1])}.")

    # Calculate R_thetas.
    R_thetas = Rotation.about_axis("z", np.angle(leading_eig_vec ** (1 / order)))

    # Form rotation matrices Ris.
    Ris = R_thetas @ Ri_tildes

    return Ris


def _complete_third_row_to_rot(r3):
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


def _cl_angles_to_ind(cl_angles, n_theta):
    """
    Map 2D direction vectors to discretized angular indices.

    For each 2D vector [x, y] in `cl_angles`, compute its polar angle
    and find the nearest of `n_theta` polar ray indices.

    :param cl_angles: Array of shape (n, 2) of [x, y] values corresponding
        to the commonline induced by a pair of rotations.
    :param n_theta: Resolution of polar rays.

    :return: int or array of length n of commonline indice in the range [0, n_theta - 1].
    """
    thetas = np.arctan2(cl_angles[:, 1], cl_angles[:, 0])

    # Shift from [-pi,pi] to [0,2*pi).
    thetas = np.mod(thetas, 2 * np.pi)

    # linear scale from [0,2*pi) to [0,n_theta).
    ind = np.mod(np.round(thetas / (2 * np.pi) * n_theta), n_theta).astype(int)

    # Return scalar for single value.
    if ind.size == 1:
        ind = ind.flat[0]

    return ind


def saff_kuijlaars(N):
    """
    Generates N vertices on the unit sphere that are approximately evenly distributed.

    This implements the recommended algorithm in spherical coordinates
    (theta, phi) according to "Distributing many points on a sphere"
    by E.B. Saff and A.B.J. Kuijlaars, Mathematical Intelligencer 19.1
    (1997) 5--11.

    :param N: Number of vertices to generate.

    :return: Nx3 array of vertices in cartesian coordinates.
    """
    k = np.arange(1, N + 1)
    h = -1 + 2 * (k - 1) / (N - 1)
    theta = np.arccos(h)
    phi = np.zeros(N)

    for i in range(1, N - 1):
        phi[i] = (phi[i - 1] + 3.6 / (np.sqrt(N * (1 - h[i] ** 2)))) % (2 * np.pi)

    # Spherical coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    mesh = np.column_stack((x, y, z))

    return mesh


def g_sync(rots, rots_gt, symmetry):
    """
    Given ground truth rotations, synchronize estimated rotations over
    symmetry group elements. This method dispatches to either the faster
    cyclic implementation for Cn symmetry or the generalized version for
    all other symmetries.

    :param rots: Estimated rotation matrices
    :param rots_gt: Ground truth rotation matrices
    :param symmetry: The symmetry of the underlying molecule.

    :return: g-synchronized ground truth rotations.
    """
    sym_grp = SymmetryGroup.parse(symmetry)

    if isinstance(sym_grp, CnSymmetryGroup):
        return g_sync_cyclic(rots, rots_gt, symmetry)
    # else
    return g_sync_finite_group(rots, rots_gt, symmetry)


def g_sync_cyclic(rots, rots_gt, symmetry):
    """
    Given ground truth rotations, synchronize estimated rotations over
    cyclic symmetry group elements.

    Every estimated rotation might be a version of the ground truth rotation
    rotated by g^{s_i}, where s_i = 0, 1, ..., order. This method synchronizes the
    ground truth rotations so that only a single global rotation need be applied
    to all estimates for error analysis.

    :param rots: Estimated rotation matrices
    :param rots_gt: Ground truth rotation matrices
    :param symmetry: The symmetry of the underlying molecule.

    :return: g-synchronized ground truth rotations.
    """
    assert len(rots) == len(
        rots_gt
    ), "Number of estimates not equal to number of references."
    n_img = len(rots)
    dtype = rots.dtype

    rots_symm = SymmetryGroup.parse(symmetry).matrices
    order = len(rots_symm)
    A_g = np.zeros((n_img, n_img), dtype=complex)

    pairs = all_pairs(n_img)

    for i, j in pairs:
        Ri = rots[i]
        Rj = rots[j]
        Rij = Ri.T @ Rj

        Ri_gt = rots_gt[i]
        Rj_gt = rots_gt[j]

        diffs = np.zeros(order)
        for s, g_s in enumerate(rots_symm):
            Rij_gt = Ri_gt.T @ g_s @ Rj_gt
            diffs[s] = min([norm(Rij - Rij_gt), norm(Rij - J_conjugate(Rij_gt))])

        idx = np.argmin(diffs)

        A_g[i, j] = np.exp(-1j * 2 * np.pi / order * idx)

    # A_g(k,l) is exp(-j(-theta_k+theta_l))
    # Diagonal elements correspond to exp(-i*0) so put 1.
    # This is important only for verification purposes that spectrum is (K,0,0,0...,0).
    A_g += A_g.conj().T + np.eye(n_img)

    _, eig_vecs = eigh(A_g)
    leading_eig_vec = eig_vecs[:, -1]

    angles = np.exp(1j * 2 * np.pi / order * np.arange(order))
    rots_gt_sync = np.zeros((n_img, 3, 3), dtype=dtype)

    for i, rot_gt in enumerate(rots_gt):
        # Since the closest ccw or cw rotation are just as good,
        # we take the absolute value of the angle differences.
        angle_dists = np.abs(np.angle(leading_eig_vec[i] / angles))
        power_g_Ri = np.argmin(angle_dists)
        rots_gt_sync[i] = rots_symm[power_g_Ri] @ rot_gt

    return rots_gt_sync


def g_sync_finite_group(rots, rots_gt, symmetry):
    """
    Synchronize ground-truth rotations over a finite symmetry group.

    This is a finite-group generalization of cyclic synchronization.  The
    pairwise matching step estimates relative symmetry elements between image
    pairs.  The spectral synchronization step then recovers one symmetry
    element per image that is globally consistent with those pairwise estimates.

    Unlike the cyclic case, the relative symmetry cannot generally be encoded
    as a scalar complex phase.  For non-commutative groups such as D_n, we instead
    represent each group element by its left-regular permutation matrix.

    :param rots: Estimated rotation matrices
    :param rots_gt: Ground truth rotation matrices
    :param symmetry: The symmetry of the underlying molecule.

    :return: g-synchronized ground truth rotations.
    """
    assert len(rots) == len(
        rots_gt
    ), "Number of estimates not equal to number of references."

    n_img = len(rots)
    dtype = rots.dtype

    # All matrices in the symmetry group.
    G = SymmetryGroup.parse(symmetry).matrices
    n_group = len(G)

    def find_group_index(A):
        """Return the index of the group matrix closest to A."""
        dists = np.linalg.norm(G - A, axis=(1, 2))
        return np.argmin(dists)

    # Build multiplication and inverse tables for the symmetry group.
    # mult[a, b] is the index c such that:
    #     G[a] @ G[b] == G[c]
    # inv[a] is the index b such that:
    #     G[a] @ G[b] == identity
    mult = np.empty((n_group, n_group), dtype=int)
    inv = np.empty(n_group, dtype=int)

    for a in range(n_group):
        inv[a] = find_group_index(G[a].T)

        for b in range(n_group):
            mult[a, b] = find_group_index(G[a] @ G[b])

    # Build the left-regular representation of the group.
    #
    # reps[a] is an n_group x n_group permutation matrix representing left
    # multiplication by group element a:
    #
    #     reps[a] @ e_b = e_{a b}
    #
    # This lets us store arbitrary finite-group relative elements in a block
    # synchronization matrix.  This is the key generalization beyond cyclic
    # scalar phases in g_sync_Cn.
    reps = np.zeros((n_group, n_group, n_group), dtype=float)

    for a in range(n_group):
        for b in range(n_group):
            reps[a, mult[a, b], b] = 1.0

    # Block synchronization matrix.
    # A is made of n_img x n_img blocks, each of size n_group x n_group.
    # Block (i, j) stores the representation of the estimated relative
    # symmetry element between images i and j.
    A = np.zeros((n_img * n_group, n_img * n_group), dtype=float)

    # The relative symmetry from an image to itself is the identity.
    for i in range(n_img):
        sl_i = slice(i * n_group, (i + 1) * n_group)
        A[sl_i, sl_i] = np.eye(n_group)

    for i, j in all_pairs(n_img):
        # Estimated relative rotation.
        Ri = rots[i]
        Rj = rots[j]
        Rij = Ri.T @ Rj

        # Ground-truth rotations for this pair.
        Ri_gt = rots_gt[i]
        Rj_gt = rots_gt[j]

        # Try every symmetry element and find which one makes the
        # ground-truth relative rotation most closely match the estimated
        # relative rotation.
        diffs = np.zeros(n_group, dtype=float)

        for s, g_s in enumerate(G):
            Rij_gt = Ri_gt.T @ g_s @ Rj_gt
            diffs[s] = min(
                np.linalg.norm(Rij - Rij_gt),
                np.linalg.norm(Rij - J_conjugate(Rij_gt)),
            )

        # Estimates relative group element h_i^{-1} h_j.
        idx_ij = np.argmin(diffs)

        sl_i = slice(i * n_group, (i + 1) * n_group)
        sl_j = slice(j * n_group, (j + 1) * n_group)

        # Store the pairwise relative group elements in blocks (i, j)/(j, i).
        A[sl_i, sl_j] = reps[idx_ij]
        A[sl_j, sl_i] = reps[idx_ij].T

    # Spectral synchronization:
    # In the noiseless case, this block matrix has a top eigenspace of
    # dimension n_group. That eigenspace encodes the unknown per-image group
    # elements, up to one global group action.
    _, eig_vecs = np.linalg.eigh(A)
    V = eig_vecs[:, -n_group:]

    # Fix the global gauge using image 0 as a reference:
    # The recovered group elements are only determined up to a common global
    # symmetry. This is fine for error analysis, because a single remaining
    # global rotation/symmetry can still be applied later.
    V0 = V[:n_group, :]
    V0_pinv = np.linalg.pinv(V0)

    rots_gt_sync = np.zeros_like(rots_gt, dtype=dtype)

    for i, rot_gt in enumerate(rots_gt):
        sl_i = slice(i * n_group, (i + 1) * n_group)
        Vi = V[sl_i, :]

        # Compare image i's eigenspace block to the reference block.
        # Ideally, this matrix is close to the representation of one group
        # element. Noise makes it approximate, so we round it to the nearest
        # valid group representation below.
        M_i = Vi @ V0_pinv

        # Round to nearest group representation.
        dists = np.linalg.norm(M_i - reps, axis=(1, 2))
        q_i = np.argmin(dists)

        # Apply the synchronized group element to the ground-truth rotation.
        h_i_sync = inv[q_i]

        rots_gt_sync[i] = G[h_i_sync] @ rot_gt

    return rots_gt_sync


def build_outer_products(n, dtype):
    """
    Builds sets of outer products of 3rd rows of rotation matrices.
    This is a helper function used in commonline testing.

    :param n: Number of 3rd rows to construct outer product from.
    :param dtype: dtype of outputs

    :return: tuple of (vijs, viis, gt_vis), where vijs are the pairwise
        outer products of gt_vis and viis are self outer products of gt_vis.
    """
    # Build random third rows, ground truth vis (unit vectors)
    gt_vis = np.zeros((n, 3), dtype=dtype)
    for i in range(n):
        np.random.seed(i)
        v = np.random.randn(3)
        gt_vis[i] = v / norm(v)

    # Find outer products viis and vijs for i<j
    nchoose2 = int(n * (n - 1) / 2)
    vijs = np.zeros((nchoose2, 3, 3), dtype=dtype)
    viis = np.zeros((n, 3, 3), dtype=dtype)

    # All pairs (i,j) where i<j
    pairs = all_pairs(n)

    for k, (i, j) in enumerate(pairs):
        vijs[k] = np.outer(gt_vis[i], gt_vis[j])

    for i in range(n):
        viis[i] = np.outer(gt_vis[i], gt_vis[i])

    return vijs, viis, gt_vis


def compare_rots_sym(R_est, R_true, sym):
    N = R_true.shape[0]
    sym_euler = SymmetryGroup.parse(sym).matrices
    order = sym_euler.shape[0]
    J = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    error = np.zeros((N, N))
    errorJ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            e = np.zeros(order)
            eJ = np.zeros(order)
            for s in range(order):
                Rs = sym_euler[s]
                e[s] = (
                    np.linalg.norm(R_est[i].T @ R_est[j] - R_true[i].T @ Rs @ R_true[j])
                    ** 2
                )
                eJ[s] = (
                    np.linalg.norm(
                        R_est[i].T @ R_est[j] - J @ R_true[i].T @ Rs @ R_true[j] @ J
                    )
                    ** 2
                )
            error[i, j] = e.min()
            errorJ[i, j] = eJ.min()
    E = min(error.sum(), errorJ.sum()) / N**2
    return E
