import logging

import numpy as np
from scipy.linalg import eigh

from aspire.abinitio import CLSymmetryC3C4
from aspire.utils import J_conjugate, Rotation, all_pairs

logger = logging.getLogger(__name__)


class CLSymmetryC2(CLSymmetryC3C4):
    """
    Define a class to estimate 3D orientations using common lines methods for molecules with C2 cyclic symmetry.

    The related publications are listed below:
    X. Cheng,
    Random Matrices in High-dimensional Data Analysis,
    PhD thesis, Princeton University, (2013).

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
        n_rad=None,
        n_theta=None,
        max_shift=0.15,
        shift_step=1,
        epsilon=1e-3,
        max_iters=1000,
        degree_res=1,
        min_dist_cls=25,
        seed=None,
        mask=True,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with C2 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param max_shift: Maximum range for shifts as a proportion of resolution. Default = 0.15.
        :param shift_step: Resolution of shift estimation in pixels. Default = 1 pixel.
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        :param degree_res: Degree resolution for estimating in-plane rotations.
        :param min_dist_cls: Minimum distance between mutual common-lines. Default = 25 degrees.
        :param seed: Optional seed for RNG.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """
        super().__init__(
            src,
            symmetry="C2",
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            epsilon=epsilon,
            max_iters=max_iters,
            degree_res=degree_res,
            seed=seed,
            mask=mask,
        )

        self.min_dist_cls = min_dist_cls
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed
        self.order = 2

    def _check_symmetry(self, symmetry):
        symmetry = symmetry.upper()
        if symmetry != "C2":
            raise NotImplementedError(
                f"Only C2 symmetry supported. {symmetry} was supplied."
            )

    def build_clmatrix(self):
        """
        Build common-lines matrix for molecules with C2 symmetry from Fourier stack of 2D images.
        This consists of finding for each pair of images the two common-lines induced by the 2-fold symmetry.
        """

        n_img = self.n_img

        if self.n_theta % 2 == 1:
            msg = "n_theta must be even"
            logger.error(msg)
            raise NotImplementedError(msg)

        # need to do a copy to prevent modifying self.pf for other functions.
        pf = self.pf.copy()

        # clmatrix contains the index in image i of the common line with image j
        # for the two sets of mutual common lines.
        clmatrix = -np.ones((2, n_img, n_img), dtype=int)

        # 1D shift between common-lines.
        shifts_1d = np.zeros((2, n_img, n_img))

        # Prepare the shift phases and generate filter for common-line detection.
        r_max = pf.shape[2]
        shifts, shift_phases, h = self._generate_shift_phase_and_filter(
            r_max, self.max_shift, self.shift_step
        )
        n_shifts = len(shifts)

        # Apply bandpass filter, normalize each ray of each image.
        # Note that only use half of each ray.
        pf = self._apply_filter_and_norm("ijk, k -> ijk", pf, r_max, h)

        # Pre-compute conjugated and shifted pf's.
        pf_shifted_flipped = np.conj(pf)[:, None] * shift_phases[:, None]
        pf_shifted_flipped = pf_shifted_flipped.reshape(
            (self.n_img, n_shifts * (self.n_theta // 2), r_max)
        )

        # Search for common lines between [i, j] pairs of images.
        for i in range(n_img - 1):
            p1 = pf[i]
            for j in range(i + 1, n_img):
                p2 = pf_shifted_flipped[j]
                corr = self._compute_correlations(p1, p2)
                corr = corr.reshape(self.n_theta, n_shifts, self.n_theta // 2)

                # Find first set of common-lines between the pair of images.
                first_cl1, first_shift, first_cl2 = np.unravel_index(
                    corr.argmax(), corr.shape
                )
                clmatrix[0, i, j] = first_cl1
                clmatrix[0, j, i] = first_cl2
                shifts_1d[0, i, j] = shifts[first_shift]

                # Mask corr around first set of common-lines to search for
                # second set of mutual common-lines.
                mask_dist = self.min_dist_cls * 2 * self.n_theta // 360
                corr_masked = corr * self._square_mask(
                    corr, first_cl1, first_cl2, mask_dist
                )

                # Find second set of mutual common-lines.
                second_cl1, second_shift, second_cl2 = np.unravel_index(
                    corr_masked.argmax(), corr.shape
                )
                clmatrix[1, i, j] = second_cl1
                clmatrix[1, j, i] = second_cl2
                shifts_1d[1, i, j] = shifts[second_shift]

        self.clmatrix = clmatrix
        self.shifts_1d = shifts_1d

    @staticmethod
    def _compute_correlations(a, b):
        """
        Compute the correlation between all pairs of lines in the polar Fourier images a and b.
        It is assumed that the polar Fourier images a and b include only half of the full set of
        rays, ie. in the range [0, 180) degrees. The correlation is expanded by computing the
        correlation with b and conj(b) to include the full set of rays associated with image b.

        :param a: 2D array size n_theta_a x radial_points.
        :param b: 2D array size n_theta_b x radial_points.

        :return: Correlation array of size n_theta_a x n_theta_b.
        """

        # Compute corrrelations in the positive and negative r directions.
        corr = np.zeros((2 * len(a), len(b)), dtype=a.dtype)
        corr[0 : len(a)] = np.real(a @ b.T)
        corr[len(a) :] = np.real(a @ np.conj(b).T)

        return corr

    @staticmethod
    def _square_mask(corr, theta_1, theta_2, dist):
        """
        For each shift we mask the correlation around the point (theta_1, theta_2)
        with a square mask of half-length `dist`.

        :param corr: Correlation array of shape (n_theta, n_shifts, n_theta // 2)
        :param theta_1: theta_1-coordinate for center of mask.
        :param theta_2: theta_2-coordinate for center of mask.
        :param dist: The distance from center to mask off.

        :return: Mask with square hole centered at (theta_1, theta_2) with shape = corr.shape.
        """
        n_theta, n_shifts, n_theta_half = corr.shape

        # Build mask.
        left = max(0, theta_1 - dist)
        right = min(n_theta, theta_1 + dist)
        bottom = max(0, theta_2 - dist)
        top = min(n_theta_half, theta_2 + dist)
        mask = np.ones((n_theta, n_theta_half), dtype=int)
        mask[left:right, bottom:top] = 0

        # Expand along shift axis.
        mask = np.repeat(mask, n_shifts, axis=0).reshape(corr.shape)

        return mask

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with C2 symmetry.
        """
        vijs, Rijs, Rijgs = self._estimate_relative_viewing_directions()

        logger.info("Estimating third rows of rotation matrices.")
        # The diagonal blocks of the 3n x 3n block matrix of relative rotations are
        # the identity since vi.T @ vi = I.
        viis = np.vstack((np.eye(3, dtype=self.dtype),) * self.n_img).reshape(
            self.n_img, 3, 3
        )
        vis = self._estimate_third_rows(vijs, viis)

        logger.info("Estimating in-plane rotations and rotations matrices.")
        Ris = self._estimate_inplane_rotations(vis, Rijs, Rijgs)

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
        # Step 1: Detect the two pairs of mutual common-lines between each pair of images
        self.build_clmatrix()

        # Step 2: Calculate relative rotations associated with both mutual common lines.
        Rijs, Rijgs = self._estimate_all_Rijs_c2(self.clmatrix)

        # Step 3: Inner J-synchronization
        Rijs, Rijgs = self._local_J_sync_c2(Rijs, Rijgs)

        # Step 4: Global J-synchronization.
        logger.info("Performing global handedness synchronization.")
        vijs, Rijs, Rijgs = self._global_J_sync(Rijs, Rijgs)

        return vijs, Rijs, Rijgs

    def _global_J_sync(self, Rijs, Rijgs):
        """
        Global handedness synchronization of relative rotations, Rijs and Rijgs,
        and relative viewing directions vijs.

        :param Rijs: Relative rotations between pairs of images, shape n_pairs x 3 x 3.
        :param Rijgs: Second set of relative rotations between pairs of images, shape n_pairs 3 x 3.
        :returns:
            - vijs - Globally synchronized relative viewing directions.
            - Rijs - Globally synchronized relative rotations.
            - Rijgs - Globally synchronized 2nd set of relative rotations.
        """
        # Here we use the fact that the mean over the set of relative rotations induced
        # by the cyclic symmetry gives us the outer product of the third rows of the rotations
        # from the i'th and j'th images. See corollary 6.2 (G. Pragier) for more details.
        vijs = (Rijs + Rijgs) / 2

        # Determine relative handedness of vijs.
        sign_ij_J = self._J_sync_power_method(vijs)

        # Synchronize relative rotations
        for i, sign in enumerate(sign_ij_J):
            if sign == -1:
                vijs[i] = J_conjugate(vijs[i])
                Rijs[i] = J_conjugate(Rijs[i])
                Rijgs[i] = J_conjugate(Rijgs[i])

        return vijs, Rijs, Rijgs

    def _estimate_inplane_rotations(self, vis, Rijs, Rijgs):
        """
        Estimate rotation matrices for each image by first constructing arbitrary rotations with
        the given third rows, vis, then applying in-plane rotations found with an angular
        synchronization procedure.

        :param vis: Estimated third rows of orientation matrices, shape n_img x 3.
        :param Rijs: First set of J-synchronized relative rotations, shape n_pairs x 3 x 3.
        :param Rijgs: Second set of J-synchronized relative rotations, shape n_pairs x 3 x 3.
        :return: Estimated rotations, Ris, shape n_img x 3 x 3.
        """
        H = np.zeros((self.n_img, self.n_img), dtype=complex)
        # Step 1: Construct all rotation matrices Ris_tilde whose third rows are equal to
        # the corresponding third rows vis.
        Ris_tilde = self._complete_third_row_to_rot(vis)

        pairs = all_pairs(self.n_img)
        for idx, (i, j) in enumerate(pairs):
            # Uij and Uijg below are xy-in-plane rotations.
            # We use svd to find the in-plane rotation which is closest to
            # Ris_tilde[i] @ Rijs[idx] @ Ris_tilde[j].T.
            # See section 8.3.1 (X. Cheng) for more details.
            Uij = Ris_tilde[i] @ Rijs[idx] @ Ris_tilde[j].T
            u, _, v = np.linalg.svd(Uij[0:2, 0:2])
            Uij = u @ v.T

            Uijg = Ris_tilde[i] @ Rijgs[idx] @ Ris_tilde[j].T
            u, _, v = np.linalg.svd(Uijg[0:2, 0:2])
            Uijg = u @ v.T

            U = (Uij @ Uij + Uijg @ Uijg) / 2
            u, _, v = np.linalg.svd(U)
            U = u @ v.T

            H[i, j] = U[0, 0] - 1j * U[1, 0]

        # Populate the lower triangle and diagonal of H.
        # Diagonals are 1 since e^{i*0}=1.
        H += np.conj(H).T + np.eye(self.n_img)

        # H is a rank-1 Hermitian matrix.
        eig_vals, eig_vecs = eigh(H, subset_by_index=[self.n_img - 5, self.n_img - 1])
        leading_eig_vec = eig_vecs[:, -1]
        logger.info(f"Top 5 eigenvalues of H are {str(eig_vals[::-1])}.")

        # Calculate R_thetas.
        R_thetas = Rotation.about_axis("z", np.angle(np.sqrt(leading_eig_vec)))

        # Form rotation matrices Ris.
        Ris = R_thetas @ Ris_tilde

        return Ris

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def _estimate_all_Rijs_c2(self, clmatrix):
        """
        Estimate the two sets of relative rotations, Rijs and Rijgs, between pairs
        of images using the voting method.

        :param clmatrix: 2 x n_img x n_img array holding two sets of mutual common-lines
            between pairs of images.
        :return: Relative rotations, Rijs and Rijgs.
        """
        k_list = np.arange(self.n_img)
        n_theta = self.n_theta
        pairs = all_pairs(self.n_img)
        Rijs = np.zeros((len(pairs), 3, 3), dtype=self.dtype)
        Rijgs = np.zeros((len(pairs), 3, 3), dtype=self.dtype)
        for idx, (i, j) in enumerate(pairs):
            Rijs[idx] = self._syncmatrix_ij_vote_3n(clmatrix[0], i, j, k_list, n_theta)
            Rijgs[idx] = self._syncmatrix_ij_vote_3n(clmatrix[1], i, j, k_list, n_theta)

        return Rijs, Rijgs

    def _local_J_sync_c2(self, Rijs, Rijgs):
        """
        For each pair of images J-synchronize the two corresponding relative rotations.

        :param Rijs: First set of relative rotations.
        :param Rijgs: Second set of relative rotations.

        Return: Pairwise J-synchronized relative rotations, Rijs and Rijgs.
        """
        e1 = np.array([1, 0, 0], dtype=self.dtype)
        vijs = (Rijs + Rijgs) / 2
        vijs_J = (J_conjugate(Rijs) + Rijgs) / 2

        s = np.linalg.svd(vijs, compute_uv=False)
        s_J = np.linalg.svd(vijs_J, compute_uv=False)

        # Indices to conjugate.
        ind_to_conj = np.argwhere(
            np.linalg.norm(s_J - e1, axis=1) < np.linalg.norm(s - e1, axis=1)
        )
        Rijgs[ind_to_conj] = J_conjugate(Rijgs[ind_to_conj])

        return Rijs, Rijgs
