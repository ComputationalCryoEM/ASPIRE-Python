import logging

import numpy as np
from numpy.linalg import eigh, norm, svd

from aspire.abinitio import CLOrient3D, SyncVotingMixin
from aspire.operators import PolarFT
from aspire.utils import (
    J_conjugate,
    Rotation,
    all_pairs,
    all_triplets,
    anorm,
    cyclic_rotations,
    tqdm,
    trange,
)
from aspire.utils.random import randn

logger = logging.getLogger(__name__)


class CLSymmetryC3C4(CLOrient3D, SyncVotingMixin):
    """
    Define a class to estimate 3D orientations using common lines methods for molecules with
    C3 and C4 cyclic symmetry.

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

    def __init__(
        self,
        src,
        symmetry=None,
        n_rad=None,
        n_theta=None,
        max_shift=0.15,
        shift_step=1,
        epsilon=1e-2,
        max_iters=1000,
        degree_res=1,
        seed=None,
        mask=True,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param symmetry: A string, 'C3' or 'C4', indicating the symmetry type.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param max_shift: Maximum range for shifts as a proportion of resolution. Default = 0.15.
        :param shift_step: Resolution of shift estimation in pixels. Default = 1 pixel.
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        :param degree_res: Degree resolution for estimating in-plane rotations.
        :param seed: Optional seed for RNG.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """

        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            mask=mask,
        )

        self._check_symmetry(symmetry)
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed

    def _check_symmetry(self, symmetry):
        if symmetry is None:
            raise NotImplementedError(
                "Symmetry type not supplied. Please indicate C3 or C4 symmetry."
            )
        else:
            symmetry = symmetry.upper()
            if symmetry not in ["C3", "C4"]:
                raise NotImplementedError(
                    f"Only C3 and C4 symmetry supported. {symmetry} was supplied."
                )
            self.order = int(symmetry[1])

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with C3 or C4 symmetry.

        :return: Array of rotation matrices, size n_imgx3x3.
        """
        vijs, viis = self._estimate_relative_viewing_directions()

        logger.info("Performing global handedness synchronization.")
        vijs, viis = self._global_J_sync(vijs, viis)

        logger.info("Estimating third rows of rotation matrices.")
        vis = self._estimate_third_rows(vijs, viis)

        logger.info("Estimating in-plane rotations and rotations matrices.")
        Ris = self._estimate_inplane_rotations(vis)

        self.rotations = Ris

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _estimate_relative_viewing_directions(self):
        """
        Estimate the relative viewing directions vij = vi*vj^T, i<j, and vii = vi*vi^T, where
        vi is the third row of the i'th rotation matrix Ri.
        """
        logger.info(f"Estimating relative viewing directions for {self.n_img} images.")
        # Step 1: Detect a single pair of common-lines between each pair of images
        self.build_clmatrix()
        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        sclmatrix = self._self_clmatrix_c3_c4()

        # Step 3: Calculate self-relative-rotations
        Riis = self._estimate_all_Riis_c3_c4(sclmatrix)

        # Step 4: Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4(clmatrix)

        # Step 5: Inner J-synchronization
        vijs, viis = self._local_J_sync_c3_c4(Rijs, Riis)

        return vijs, viis

    def _global_J_sync(self, vijs, viis):
        """
        Global J-synchronization of all third row outer products. Given 3x3 matrices vijs and viis, each
        of which might contain a spurious J (ie. vij = J*vi*vj^T*J instead of vij = vi*vj^T),
        we return vijs and viis that all have either a spurious J or not.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of the rotation matrices Ri and Rj. Each estimate
        might have a spurious J independently of other estimates.

        :param viis: An n_imgx3x3 array where the i'th slice holds an estimate for the outer product vi*vi^T
        between the third row of matrix Ri and itself. Each estimate might have a spurious J independently
        of other estimates.

        :return: vijs, viis all of which have a spurious J or not.
        """
        n_img = self.n_img

        # Determine relative handedness of vijs.
        sign_ij_J = self._J_sync_power_method(vijs)

        # Synchronize vijs
        for i, sign in enumerate(sign_ij_J):
            if sign == -1:
                vijs[i] = J_conjugate(vijs[i])

        # Synchronize viis
        # We use the fact that if v_ii and v_ij are of the same handedness, then v_ii @ v_ij = v_ij.
        # If they are opposite handed then Jv_iiJ @ v_ij = v_ij. We compare each v_ii against all
        # previously synchronized v_ij to get a consensus on the handedness of v_ii.
        _, pairs_to_linear = all_pairs(n_img, return_map=True)
        for i in range(n_img):
            vii = viis[i]
            vii_J = J_conjugate(vii)
            J_consensus = 0
            for j in range(n_img):
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
        return vijs, viis

    def _estimate_third_rows(self, vijs, viis):
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

        n_img = self.n_img

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

    def _estimate_inplane_rotations(self, vis):
        """
        Estimate the rotation matrices for each image by constructing arbitrary rotation matrices
        populated with the given third rows, vis, and then rotating by an appropriate in-plane rotation.

        :param vis: An n_imgx3 array where the i'th row holds the estimate for the third row of
        the i'th rotation matrix.

        :return: Rotation matrices Ris and in-plane rotation matrices R_thetas, both size n_imgx3x3.
        """
        pf = self.pf
        n_img = self.n_img
        n_theta = self.n_theta
        max_shift_1d = self.max_shift
        shift_step = self.shift_step
        order = self.order
        degree_res = self.degree_res

        # Step 1: Construct all rotation matrices Ri_tildes whose third rows are equal to
        # the corresponding third rows vis.
        Ri_tildes = self._complete_third_row_to_rot(vis)

        # Step 2: Construct all in-plane rotation matrices, R_theta_ijs.
        max_angle = (360 // order) * order
        theta_ijs = np.arange(0, max_angle, degree_res) * np.pi / 180
        R_theta_ijs = Rotation.about_axis("z", theta_ijs, dtype=self.dtype).matrices

        # Step 3: Compute the correlation over all shifts.
        # Generate shifts.
        r_max = pf.shape[-1]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step
        )
        n_shifts = len(shifts)

        # Q is the n_img x n_img  Hermitian matrix defined by Q = q*q^H,
        # where q = (exp(i*order*theta_0), ..., exp(i*order*theta_{n_img-1}))^H,
        # and theta_i in [0, 2pi/order) is the in-plane rotation angle for the i'th image.
        Q = np.zeros((n_img, n_img), dtype=complex)

        # Reconstruct the full polar Fourier for use in correlation. self.pf only consists of
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
                        [
                            Ri_tilde.T @ R_theta_ij @ Rj_tilde
                            for R_theta_ij in R_theta_ijs
                        ]
                    )

                    # Find the angle between common lines induced by the rotations.
                    c1s = np.array([[-U[1, 2], U[0, 2]] for U in Us])
                    c2s = np.array([[U[2, 1], -U[2, 0]] for U in Us])

                    # Convert from angles to indices.
                    c1s = self.cl_angles_to_ind(c1s, n_theta)
                    c2s = self.cl_angles_to_ind(c2s, n_theta)

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
            logger.info(
                f"Top 3 eigenvalues of Q (rank-1) are {str(eig_vals[-3:][::-1])}."
            )

            # Calculate R_thetas.
            R_thetas = Rotation.about_axis(
                "z", np.angle(leading_eig_vec ** (1 / order))
            )

            # Form rotation matrices Ris.
            Ris = R_thetas @ Ri_tildes

            return Ris

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def _self_clmatrix_c3_c4(self):
        """
        Find the single pair of self-common-lines in each image assuming that the underlying
        symmetry is C3 or C4.
        """
        pf = self.pf
        n_img = self.n_img
        L = self.src.L
        n_theta = self.n_theta
        max_shift_1d = self.max_shift
        shift_step = self.shift_step
        order = self.order

        # The angle between two self-common-lines is constrained by the underlying symmetry
        # of the molecule (See Lemma A.2 in the listed publication for further details).
        # This angle is in the range [60, 180] for C3 symmetry and [90, 180] for C4 symmetry.
        # Since antipodal lines are perfectly correlated we search for common lines in a smaller window.
        # Note: matlab code used [60, 165] for C3 and [90, 160] for C4.
        # We set the upper bound of this window to be within a factor of 1/L of 180.
        if order == 3:
            min_angle_diff = 60 * np.pi / 180
        else:
            min_angle_diff = 90 * np.pi / 180

        res = 2 * (360 // L)
        max_angle_diff = np.pi - res * np.pi / 180

        # We create a mask associated with angle differences that fall in the
        # range [min_angle_diff, max_angle_diff].

        # `theta_full` and `theta_half` are grids of theta values associate with the polar Fourier transforms
        # `pf_full` and `pf`, respectively.
        theta_vals = np.linspace(0, 360, n_theta)
        theta_vals_half = theta_vals[: len(theta_vals) // 2]
        theta_full, theta_half = np.meshgrid(theta_vals, theta_vals_half)

        # `diff` is the unsigned angle differences between all pairs of polar Fourier rays.
        diff = abs(theta_half - theta_full)
        diff[diff > 180] = 360 - diff[diff > 180]
        diff = np.deg2rad(diff)

        # Build mask.
        good_diffs = (min_angle_diff < diff) & (diff < max_angle_diff)

        # Compute the correlation over all shifts.
        # Generate Shifts.
        r_max = pf.shape[-1]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step
        )
        n_shifts = len(shifts)

        # Reconstruct the full polar Fourier for use in correlation. self.pf only consists of
        # rays in the range [180, 360), with shape (n_img, n_theta//2, n_rad-1).
        pf_full = PolarFT.half_to_full(pf)

        # The self-common-lines matrix holds two indices per image that represent
        # the two self common-lines in the image.
        sclmatrix = np.zeros((n_img, 2))

        for i in trange(n_img):
            pf_i = pf[i]
            pf_full_i = pf_full[i]

            # Generate shifted versions of images.
            pf_i_shifted = np.array(
                [pf_i * shift_phase for shift_phase in shift_phases]
            )
            pf_i_shifted = np.reshape(pf_i_shifted, (n_shifts * n_theta // 2, r_max))

            # # Normalize each ray.
            pf_full_i /= norm(pf_full_i, axis=1)[..., np.newaxis]
            pf_i_shifted /= norm(pf_i_shifted, axis=1)[..., np.newaxis]

            # Compute correlation.
            corrs = pf_i_shifted @ pf_full_i.T
            corrs = np.reshape(corrs, (n_shifts, n_theta // 2, n_theta))

            # Mask with allowed combinations.
            corrs *= good_diffs[np.newaxis, ...]

            # Find maximum correlation.
            shift, scl1, scl2 = np.unravel_index(np.argmax(np.real(corrs)), corrs.shape)
            sclmatrix[i] = [scl1, scl2]

        return sclmatrix

    def _estimate_all_Riis_c3_c4(self, sclmatrix):
        """
        Compute estimates for the self relative rotations Rii for every rotation matrix Ri.
        """

        order = self.order
        n_theta = self.n_theta

        # Calculate the cosine of angle between self-common-lines.
        cos_diff = np.cos((sclmatrix[:, 1] - sclmatrix[:, 0]) * 2 * np.pi / n_theta)

        # Calculate Euler angles `euler_y2` (gamma_ii in publication) corresponding to Y in ZYZ convention.
        if order == 3:
            # cos_diff should be <= 0.5, but due to discretization that might be violated.
            if np.max(cos_diff) > 0.5:
                bad_diffs = np.count_nonzero(cos_diff > 0.5)
                logger.debug(
                    "cos(angular_diff) should be < 0.5."
                    f"Found {bad_diffs} estimates exceeding 0.5, with maximum {np.max(cos_diff)}."
                    " Setting all bad estimates to 0.5."
                )
                cos_diff = np.minimum(cos_diff, 0.5)
            euler_y2 = np.arccos(cos_diff / (1 - cos_diff))
        else:
            # cos_diff should be <= 0, but due to discretization that might be violated.
            if np.max(cos_diff) > 0:
                bad_diffs = np.count_nonzero(cos_diff > 0)
                logger.debug(
                    "cos(angular_diff) should be < 0."
                    f"Found {bad_diffs} estimates exceeding 0, with maximum {np.max(cos_diff)}."
                    " Setting all bad estimates to 0."
                )
                cos_diff = np.minimum(cos_diff, 0)
            euler_y2 = np.arccos((1 + cos_diff) / (1 - cos_diff))

        # Calculate remaining Euler angles in ZYZ convention.
        # Note: Publication uses ZXZ convention. Using the notation of the
        # publication the Euler parameterization in ZXZ convention is given by
        # (alpha_ii^(1), gamma_ii, -alpha_ii^(n-1) - pi).
        # Converting to ZYZ convention gives us the parameteriztion
        # (alpha_ii^(1) - pi/2, gamma_ii, -alpha_ii^(n-1) - pi/2).
        euler_z1 = sclmatrix[:, 0] * 2 * np.pi / n_theta - np.pi / 2
        euler_z3 = -sclmatrix[:, 1] * 2 * np.pi / n_theta - np.pi / 2

        # Compute Riis from Euler angles.
        angles = np.array((euler_z1, euler_y2, euler_z3), dtype=self.dtype).T
        Riis = Rotation.from_euler(angles, dtype=self.dtype).matrices

        return Riis

    def _estimate_all_Rijs_c3_c4(self, clmatrix):
        """
        Estimate Rijs using the voting method.
        """
        n_img = self.n_img
        n_theta = self.n_theta
        pairs = all_pairs(n_img)
        Rijs = np.zeros((len(pairs), 3, 3))
        for idx, (i, j) in enumerate(pairs):
            Rijs[idx] = self._syncmatrix_ij_vote_3n(
                clmatrix, i, j, np.arange(n_img), n_theta
            )

        return Rijs

    def _syncmatrix_ij_vote_3n(self, clmatrix, i, j, k_list, n_theta):
        """
        Compute the (i,j) rotation block of the synchronization matrix using voting method

        Given the common lines matrix `clmatrix`, a list of images specified in k_list
        and the number of common lines n_theta, find the (i, j) rotation block Rij.

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

        else:
            # This is for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # We set to zero as in the Matlab code.
            rot_mean = np.zeros((3, 3))

        return rot_mean

    def _local_J_sync_c3_c4(self, Rijs, Riis):
        """
        Estimate viis and vijs. In order to estimate vij = vi @ vj.T, it is necessary for Rii, Rjj,
        and Rij to be of the same handedness. We perform a local handedness synchronization and
        set vij = 1/order * sum(Rii^s @ Rij @ Rjj^s) for s = 0, 1, ..., order. To estimate
        vii = vi @ vi.T we set vii = 1/order * sum(Rii^s) for s = 0, 1, ..., order.

        :param Rijs: An n-choose-2x3x3 array of estimates of relative rotations
            (each pair of images induces two estimates).
        :param Riis: A nx3x3 array of estimates of self-relative rotations.
        :return: vijs, viis
        """
        order = self.order
        n_img = self.n_img
        pairs = all_pairs(n_img)

        # Estimate viis from Riis. vii = 1/order * sum(Rii^s) for s = 0, 1, ..., order.
        viis = np.zeros((n_img, 3, 3))
        for i, Rii in enumerate(Riis):
            viis[i] = np.mean(
                [np.linalg.matrix_power(Rii, s) for s in np.arange(order)], axis=0
            )

        # Estimate vijs via local handedness synchronization.
        vijs = np.zeros((len(pairs), 3, 3))

        for idx, (i, j) in enumerate(pairs):
            opts = np.zeros((2, 2, 2, 3, 3))
            scores_rank1 = np.zeros(8)
            Rii = Riis[i]
            Rjj = Riis[j]
            Rij = Rijs[idx]

            # At this stage, the estimates Rii and Rjj are in the sets {Ri.T @ g^si @ Ri, JRi.T @ g^si @ RiJ}
            # and {Rj.T @ g^sj @ Rj, JRj.T @ g^sj @ RjJ}, respectively, where g^si (and g^sj) is a rotation
            # about the axis of symmetry by (2pi * si)/order, with si = 1 or order-1. Additionally, the estimate
            # Rij might have a spurious J.

            # To estimate vij, it is essential that si = sj and all three estimates Rii, Rjj, and Rij have
            # a spurious J, or none do at all. Note: since g.T = g^(order-1), if Rii = Ri.T @ g^1 @ Ri, then
            # Rii.T = Ri.T @ g^(order-1) @ Ri.

            # The estimate vij should be of rank 1 with singular values [1, 0, 0].
            # We test 8 combinations of handedness and rotation by {g, g^(order-1)} for this condition to determine:
            # a. whether to transpose Rii
            # b. whether to J-conjugate Rii
            # c. whether to J-conjugate Rjj
            if order == 3:
                for ii_trans in [0, 1]:
                    for ii_conj in [0, 1]:
                        for jj_conj in [0, 1]:
                            _Rii = Rii.T if ii_trans else Rii
                            _Rii = J_conjugate(_Rii) if ii_conj else _Rii
                            _Rjj = J_conjugate(Rjj) if jj_conj else Rjj
                            opts[ii_trans, ii_conj, jj_conj] = (
                                Rij + _Rii @ Rij @ _Rjj + _Rii.T @ Rij @ _Rjj.T
                            ) / 3

            else:
                for ii_trans in [0, 1]:
                    for ii_conj in [0, 1]:
                        for jj_conj in [0, 1]:
                            _Rii = Rii.T if ii_trans else Rii
                            _Rii = J_conjugate(_Rii) if ii_conj else _Rii
                            _Rjj = J_conjugate(Rjj) if jj_conj else Rjj
                            opts[ii_trans, ii_conj, jj_conj] = (
                                Rij + _Rii @ Rij @ _Rjj
                            ) / 2

            opts = opts.reshape((8, 3, 3))
            svals = svd(opts, compute_uv=False)
            scores_rank1 = anorm(svals - [1, 0, 0], axes=[1])
            min_idx = np.argmin(scores_rank1)

            # Populate vijs with
            vijs[idx] = opts[min_idx]

        return vijs, viis

    #######################################
    # Secondary Methods for Global J Sync #
    #######################################

    def _J_sync_power_method(self, vijs):
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
        J_sync = np.sign(vec)

        return J_sync

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

        return new_vec

    ###########################################
    # Secondary Methods fo In-Plane Rotations #
    ###########################################
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def g_sync(rots, order, rots_gt):
        """
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
