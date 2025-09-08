import logging

import numpy as np
from numpy.linalg import eigh, norm

from aspire.operators import PolarFT
from aspire.utils import (
    J_conjugate,
    Rotation,
    all_pairs,
    all_triplets,
    anorm,
    cyclic_rotations,
    tqdm,
)
from aspire.utils.random import randn

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
    vis /= anorm(vis, axes=(-1,))[:, np.newaxis]

    return vis


def _estimate_inplane_rotations(cl_class, vis):
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
    Ri_tildes = _complete_third_row_to_rot(vis)

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
    pbar = tqdm(total=n_pairs)
    # Note: the ordering of i and j in these loops should not be changed as
    # they correspond to the ordered tuples (i, j), for i<j.
    for i in range(n_img):
        pf_i = pf[i]

        # Generate shifted versions of images.
        pf_i_shifted = np.array([pf_i * shift_phase for shift_phase in shift_phases])

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
            c1s = _cl_angles_to_ind(c1s, n_theta)
            c2s = _cl_angles_to_ind(c2s, n_theta)

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
    pbar.close()

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
    thetas = np.arctan2(cl_angles[:, 1], cl_angles[:, 0])

    # Shift from [-pi,pi] to [0,2*pi).
    thetas = np.mod(thetas, 2 * np.pi)

    # linear scale from [0,2*pi) to [0,n_theta).
    ind = np.mod(np.round(thetas / (2 * np.pi) * n_theta), n_theta).astype(int)

    # Return scalar for single value.
    if ind.size == 1:
        ind = ind.flat[0]

    return ind


def g_sync(rots, order, rots_gt):
    """
    Given ground truth rotations, synchronize estimated rotations over
    symmetry group elements.

    Every estimated rotation might be a version of the ground truth rotation
    rotated by g^{s_i}, where s_i = 0, 1, ..., order. This method synchronizes the
    ground truth rotations so that only a single global rotation need be applied
    to all estimates for error analysis.

    :param rots: Estimated rotation matrices
    :param order: The cyclic order asssociated with the symmetry of the underlying molecule.
    :param rots_gt: Ground truth rotation matrices.

    :return: g-synchronized ground truth rotations.
    """
    assert len(rots) == len(
        rots_gt
    ), "Number of estimates not equal to number of references."
    n_img = len(rots)
    dtype = rots.dtype

    rots_symm = cyclic_rotations(order, dtype).matrices

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
    A_g += np.conj(A_g).T + np.eye(n_img)

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


class JSync:
    """
    Class for handling J-synchronization methods.
    """

    def __init__(
        self,
        n,
        epsilon=1e-2,
        max_iters=1000,
        seed=None,
    ):
        """
        Initialize JSync object for estimating global handedness synchronization for a
        set of relative rotations, Rij = Ri @ Rj.T, where i <= j = 0, 1, ..., n.

        :param n: Number of images/rotations.
        :param epsilon: Tolerance for the power method.
        :param max_iters: Maximum iterations for the power method.
        :param seed: Optional seed for power method initial random vector.
        """
        self.n_img = n
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.seed = seed

    def power_method(self, vijs):
        """
        Calculate the leading eigenvector of the J-synchronization matrix
        using the power method.

        As the J-synchronization matrix is of size (n-choose-2)x(n-choose-2), we
        use the power method to compute the eigenvalues and eigenvectors,
        while constructing the matrix on-the-fly.

        :param vijs: (n-choose-2)x3x3 array of estimates of relative orientation matrices.

        :return: An array of length n-choose-2 consisting of 1 or -1, where the sign of the
            i'th entry indicates whether the i'th relative orientation matrix will be J-conjugated.
        """

        # Set power method tolerance and maximum iterations.
        epsilon = self.epsilon
        max_iters = self.max_iters

        # Initialize candidate eigenvectors
        n_vijs = vijs.shape[0]
        vec = randn(n_vijs, seed=self.seed)
        vec = vec / norm(vec)
        residual = 1
        itr = 0

        # Power method iterations
        logger.info(
            "Initiating power method to estimate J-synchronization matrix eigenvector."
        )
        while itr < max_iters and residual > epsilon:
            itr += 1
            # Note, this appears to need double precision for accuracy in the following division.
            vec_new = self._signs_times_v(vijs, vec).astype(np.float64, copy=False)
            vec_new = vec_new / norm(vec_new)
            residual = norm(vec_new - vec)
            vec = vec_new
            logger.info(
                f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
            )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec, dtype=vijs.dtype)

        return J_sync

    def sync_viis(self, vijs, viis):
        """
        Given a set of synchronized pairwise outer products vijs, J-synchronize the set of
        outer products viis.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of the rotation matrices Ri and Rj. Each estimate
        might have a spurious J independently of other estimates.

        :param viis: An n_imgx3x3 array where the i'th slice holds an estimate for the outer product vi*vi^T
        between the third row of matrix Ri and itself. Each estimate might have a spurious J independently
        of other estimates.

        :return: J-synchronized viis.
        """

        # Synchronize viis
        # We use the fact that if v_ii and v_ij are of the same handedness, then v_ii @ v_ij = v_ij.
        # If they are opposite handed then Jv_iiJ @ v_ij = v_ij. We compare each v_ii against all
        # previously synchronized v_ij to get a consensus on the handedness of v_ii.
        _, pairs_to_linear = all_pairs(self.n_img, return_map=True)
        for i in range(self.n_img):
            vii = viis[i]
            vii_J = J_conjugate(vii)
            J_consensus = 0
            for j in range(self.n_img):
                if j < i:
                    idx = pairs_to_linear[j, i]
                    vji = vijs[idx]

                    err1 = norm(vji @ vii - vji)
                    err2 = norm(vji @ vii_J - vji)

                elif j > i:
                    idx = pairs_to_linear[i, j]
                    vij = vijs[idx]

                    err1 = norm(vii @ vij - vij)
                    err2 = norm(vii_J @ vij - vij)

                else:
                    continue

                # Accumulate J consensus
                if err1 < err2:
                    J_consensus -= 1
                else:
                    J_consensus += 1

            if J_consensus > 0:
                viis[i] = vii_J
        return viis

    def _signs_times_v(self, vijs, vec):
        """
        Multiplication of the J-synchronization matrix by a candidate eigenvector.

        The J-synchronization matrix is a matrix representation of the handedness graph, Gamma, whose set of
        nodes consists of the estimates vijs and whose set of edges consists of the undirected edges between
        all triplets of estimates vij, vjk, and vik, where i<j<k. The weight of an edge is set to +1 if its
        incident nodes agree in handednes and -1 if not.

        The J-synchronization matrix is of size (n-choose-2)x(n-choose-2), where each entry corresponds to
        the relative handedness of vij and vjk. The entry (ij, jk), where ij and jk are retrieved from the
        all_pairs indexing, is 1 if vij and vjk are of the same handedness and -1 if not. All other entries
        (ij, kl) hold a zero.

        Due to the large size of the J-synchronization matrix we construct it on the fly as follows.
        For each triplet of outer products vij, vjk, and vik, the associated elements of the J-synchronization
        matrix are populated with +1 or -1 and multiplied by the corresponding elements of
        the current candidate eigenvector supplied by the power method. The new candidate eigenvector
        is updated for each triplet.

        :param vijs: (n-choose-2)x3x3 array, where each 3x3 slice holds the outer product of vi and vj.

        :param vec: The current candidate eigenvector of length n-choose-2 from the power method.

        :return: New candidate eigenvector of length n-choose-2. The product of the J-sync matrix and vec.
        """

        # All pairs (i,j) and triplets (i,j,k) where i<j<k
        n_img = self.n_img
        triplets = all_triplets(n_img)
        pairs, pairs_to_linear = all_pairs(n_img, return_map=True)

        # There are 4 possible configurations of relative handedness for each triplet (vij, vjk, vik).
        # 'conjugate' expresses which node of the triplet must be conjugated (True) to achieve synchronization.
        conjugate = np.empty((4, 3), bool)
        conjugate[0] = [False, False, False]
        conjugate[1] = [True, False, False]
        conjugate[2] = [False, True, False]
        conjugate[3] = [False, False, True]

        # 'edges' corresponds to whether conjugation agrees between the pairs (vij, vjk), (vjk, vik),
        # and (vik, vij). True if the pairs are in agreement, False otherwise.
        edges = np.empty((4, 3), bool)
        edges[:, 0] = conjugate[:, 0] == conjugate[:, 1]
        edges[:, 1] = conjugate[:, 1] == conjugate[:, 2]
        edges[:, 2] = conjugate[:, 2] == conjugate[:, 0]

        # The corresponding entries in the J-synchronization matrix are +1 if the pair of nodes agree, -1 if not.
        edge_signs = np.where(edges, 1, -1)

        # For each triplet of nodes we apply the 4 configurations of conjugation and determine the
        # relative handedness based on the condition that vij @ vjk - vik = 0 for synchronized nodes.
        # We then construct the corresponding entries of the J-synchronization matrix with 'edge_signs'
        # corresponding to the conjugation configuration producing the smallest residual for the above
        # condition. Finally, we the multiply the 'edge_signs' by the cooresponding entries of 'vec'.
        v = vijs
        new_vec = np.zeros_like(vec)
        pbar = tqdm(desc="Computing signs_times_v", total=len(triplets))
        for i, j, k in triplets:
            ij = pairs_to_linear[i, j]
            jk = pairs_to_linear[j, k]
            ik = pairs_to_linear[i, k]
            vij, vjk, vik = v[ij], v[jk], v[ik]
            vij_J = J_conjugate(vij)
            vjk_J = J_conjugate(vjk)
            vik_J = J_conjugate(vik)

            conjugated_pairs = np.where(
                conjugate[..., np.newaxis, np.newaxis],
                [vij_J, vjk_J, vik_J],
                [vij, vjk, vik],
            )
            residual = np.stack([norm(x @ y - z) for x, y, z in conjugated_pairs])

            min_residual = np.argmin(residual)

            # Assign edge weights
            s_ij_jk, s_ik_jk, s_ij_ik = edge_signs[min_residual]

            # Update multiplication of signs times vec
            new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
            new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
            new_vec[ik] += s_ij_ik * vec[ij] + s_ik_jk * vec[jk]
            pbar.update()
        pbar.close()

        return new_vec
