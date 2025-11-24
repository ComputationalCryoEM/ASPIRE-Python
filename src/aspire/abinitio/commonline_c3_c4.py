import logging

import numpy as np
from numpy.linalg import norm, svd

from aspire.abinitio import CLOrient3D, JSync
from aspire.abinitio.sync_voting import _syncmatrix_ij_vote_3n
from aspire.operators import PolarFT
from aspire.utils import J_conjugate, Rotation, all_pairs, anorm, trange

from .commonline_utils import (
    _estimate_inplane_rotations,
    _estimate_third_rows,
    _generate_shift_phase_and_filter,
)

logger = logging.getLogger(__name__)


class CLSymmetryC3C4(CLOrient3D):
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
        n_theta=360,
        max_shift=0.15,
        shift_step=1,
        epsilon=1e-2,
        max_iters=1000,
        degree_res=1,
        seed=None,
        mask=True,
        **kwargs,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param symmetry: A string, 'C3' or 'C4', indicating the symmetry type.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction. Default = 360.
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
            **kwargs,
        )

        self._check_symmetry(symmetry)
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed

        self.J_sync = JSync(src.n, self.epsilon, self.max_iters, self.seed)

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
        vis = _estimate_third_rows(vijs, viis)

        logger.info("Estimating in-plane rotations and rotations matrices.")
        Ris = _estimate_inplane_rotations(
            vis,
            self.pf,
            self.max_shift,
            self.shift_step,
            self.order,
            self.degree_res,
        )

        self.rotations = Ris

        return self.rotations

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

        # Step 2: Detect self-common-lines in each image
        sclmatrix = self._self_clmatrix_c3_c4()

        # Step 3: Calculate self-relative-rotations
        Riis = self._estimate_all_Riis_c3_c4(sclmatrix)

        # Step 4: Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4()

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

        # Determine relative handedness of vijs.
        vijs = self.J_sync.global_J_sync(vijs)

        # Determine relative handedness of viis, given synchronized vijs.
        viis = self.J_sync.sync_viis(vijs, viis)

        return vijs, viis

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
        shifts, shift_phases, _ = _generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step, self.dtype
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

    def _estimate_all_Rijs_c3_c4(self):
        """
        Estimate Rijs using the voting method.
        """
        pairs = all_pairs(self.n_img)
        Rijs = np.zeros((len(pairs), 3, 3))
        for idx, (i, j) in enumerate(pairs):
            Rijs[idx] = _syncmatrix_ij_vote_3n(
                self.clmatrix,
                i,
                j,
                np.arange(self.n_img),
                self.n_theta,
                self.hist_bin_width,
                self.full_width,
            )

        return Rijs

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
