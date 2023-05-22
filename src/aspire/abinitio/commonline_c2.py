import logging

import numpy as np

from aspire.abinitio import CLSymmetryC3C4
from aspire.utils import J_conjugate, Rotation, all_pairs
from aspire.utils.random import choice

logger = logging.getLogger(__name__)


class CLSymmetryC2(CLSymmetryC3C4):
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
    ):
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
        )

        self.min_dist_cls = min_dist_cls
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed
        self.order = 2

    def _check_symmetry(self, symmetry):
        # `symmetry` is not configurable for `CLSymmetryC2`
        pass

    def build_clmatrix(self):
        """
        Build common-lines matrix from Fourier stack of 2D images.
        """

        n_img = self.n_img
        n_check = self.n_check

        if self.n_theta % 2 == 1:
            msg = "n_theta must be even"
            logger.error(msg)
            raise NotImplementedError(msg)

        # need to do a copy to prevent modifying self.pf for other functions
        pf = self.pf.copy()

        # clmatrix[i,j] contains the index in image i of the common line with image j.
        # -1 means there is no common line such as clmatrix[i,i].
        clmatrix = -np.ones((2, n_img, n_img), dtype=self.dtype)

        # When cl_dist[i, j] is not -1, it stores the maximum value
        # of correlation between image i and j for all possible 1D shifts.
        # We will use cl_dist[i, j] = -1 (including j<=i) to
        # represent that there is no need to check common line
        # between i and j. Since it is symmetric,
        # only above the diagonal entries are necessary.
        cl_dist = -np.ones((2, n_img, n_img), dtype=self.dtype)

        # Allocate variables used for shift estimation

        # set maximum value of 1D shift (in pixels) to search
        # between common-lines.
        max_shift = self.max_shift
        # Set resolution of shift estimation in pixels.
        shift_step = self.shift_step
        # 1D shift between common-lines
        shifts_1d = np.zeros((2, n_img, n_img))

        # Prepare the shift phases and generate filter for common-line detection
        r_max = pf.shape[2]
        shifts, shift_phases, h = self._generate_shift_phase_and_filter(
            r_max, max_shift, shift_step
        )
        all_shift_phases = shift_phases.T

        # Apply bandpass filter, normalize each ray of each image
        # Note that only use half of each ray
        pf = self._apply_filter_and_norm("ijk, k -> ijk", pf, r_max, h)

        # Search for common lines between [i, j] pairs of images.
        # The random selection is implemented.
        for i in range(n_img - 1):
            p1 = pf[i]
            p1_real = np.real(p1)
            p1_imag = np.imag(p1)

            # build the subset of j images if n_check < n_img
            n_remaining = n_img - i - 1
            n_j = min(n_remaining, n_check)
            subset_j = np.sort(choice(n_remaining, n_j, replace=False) + i + 1)

            for j in subset_j:
                p2_flipped = np.conj(pf[j])

                for shift in range(len(shifts)):
                    shift_phases = all_shift_phases[shift]
                    p2_shifted_flipped = (shift_phases * p2_flipped).T

                    # Compute correlations in the positive and negative r directions
                    part1 = p1_real.dot(np.real(p2_shifted_flipped))
                    part2 = p1_imag.dot(np.imag(p2_shifted_flipped))
                    c1 = part1 - part2
                    c2 = part1 + part2
                    C = np.concatenate((c1, c2), axis=0)

                    s_ind = C.argmax()
                    first_cl1, first_cl2 = np.unravel_index(s_ind, C.shape)
                    s_val = C[first_cl1, first_cl2]
                    if s_val > cl_dist[0, i, j]:
                        clmatrix[0, i, j] = first_cl1
                        clmatrix[0, j, i] = first_cl2
                        cl_dist[0, i, j] = s_val
                        shifts_1d[0, i, j] = shifts[shift]

                    # Mask cl_dist to find second highest correlation.
                    mask_dist = self.min_dist_cls * 2 * self.n_theta // 360
                    C_masked = self.square_mask(C, first_cl1, first_cl2, mask_dist)
                    sidx = C_masked.argmax()
                    second_cl1, second_cl2 = np.unravel_index(sidx, C.shape)
                    s_val_2 = C[second_cl1, second_cl2]
                    if s_val_2 > cl_dist[1, i, j]:
                        clmatrix[1, i, j] = second_cl1
                        clmatrix[1, j, i] = second_cl2
                        cl_dist[1, i, j] = s_val_2
                        shifts_1d[1, i, j] = shifts[shift]

        self.clmatrix = clmatrix
        self.shifts_1d = shifts_1d

    @staticmethod
    def square_mask(arr, x, y, dist):
        """
        Mask input array around the point (x, y) with a square mask of half-length `dist`.
        """
        left = max(0, x - dist)
        right = min(len(arr), x + dist)
        bottom = max(0, y - dist)
        top = min(len(arr[0]), y + dist)
        mask = np.ones_like(arr)
        mask[left:right, bottom:top] = 0

        return arr * mask

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with C2 symmetry.
        """
        Rijs, Rijgs = self._estimate_relative_viewing_directions()

        logger.info("Performing global handedness synchronization.")
        vijs, Rijs, Rijgs = self._global_J_sync(Rijs, Rijgs)

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
        clmatrix = self.clmatrix

        # Step 2: Calculate relative rotations associated with both mutual common lines.
        Rijs, Rijgs = self._estimate_all_Rijs_c2(clmatrix)

        # Step 3: Inner J-synchronization
        Rijs, Rijgs = self._local_J_sync_c2(Rijs, Rijgs)

        return Rijs, Rijgs

    def _global_J_sync(self, Rijs, Rijgs):
        """
        Global handedness synchronization of relative rotations.
        """
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
        """
        H = np.zeros((self.n_img, self.n_img), dtype=complex)
        # Step 1: Construct all rotation matrices Ri_tildes whose third rows are equal to
        # the corresponding third rows vis.
        Ris_tilde = np.array([self._complete_third_row_to_rot(vi) for vi in vis])

        pairs = all_pairs(self.n_img)
        for idx, (i, j) in enumerate(pairs):
            # Uij and Uijg below are xy-in-plane rotations.
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
        eig_vals, eig_vecs = np.linalg.eigh(H)
        leading_eig_vec = eig_vecs[:, -1]
        logger.info(f"Top 5 eigenvalues of H are {str(eig_vals[-5:][::-1])}.")

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
        Estimate Rijs using the voting method.
        """
        n_img = self.n_img
        n_theta = self.n_theta
        pairs = all_pairs(n_img)
        Rijs = np.zeros((len(pairs), 3, 3))
        Rijgs = np.zeros((len(pairs), 3, 3))
        for idx, (i, j) in enumerate(pairs):
            Rijs[idx] = self._syncmatrix_ij_vote_3n(
                clmatrix[0], i, j, np.arange(n_img), n_theta
            )
            Rijgs[idx] = self._syncmatrix_ij_vote_3n(
                clmatrix[1], i, j, np.arange(n_img), n_theta
            )

        return Rijs, Rijgs

    def _local_J_sync_c2(self, Rijs, Rijgs):
        """
        Local J-synchronization of all relative rotations.
        """
        e1 = np.array([1, 0, 0], dtype=self.dtype)
        pairs = all_pairs(self.n_img)
        for idx, _ in enumerate(pairs):
            Rij = Rijs[idx]
            Rijg = Rijgs[idx]

            # Rij + Rij_g must be rank-1. If not, J-conjugate either of them.
            vij = (Rij + Rijg) / 2
            vij_J = (J_conjugate(Rij) + Rijg) / 2

            s = np.linalg.svd(vij, compute_uv=False)
            s_J = np.linalg.svd(vij_J, compute_uv=False)

            if np.linalg.norm(s_J - e1) < np.linalg.norm(s - e1):
                Rijgs[idx] = J_conjugate(Rijg)

        return Rijs, Rijgs
