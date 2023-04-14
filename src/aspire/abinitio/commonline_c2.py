import logging

import numpy as np

from aspire.abinitio import CLSymmetryC3C4
from aspire.utils import all_pairs
from aspire.utils.random import choice

logger = logging.getLogger(__name__)


class CLSymmetryC2(CLSymmetryC3C4):
    def __init__(
        self,
        src,
        symmetry=None,
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
            symmetry=symmetry,
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
        self._check_symmetry(symmetry)
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed

    def _check_symmetry(self, symmetry):
        if symmetry is None:
            raise NotImplementedError("Symmetry type not supplied.")
        else:
            symmetry = symmetry.upper()
            if symmetry not in ["C2"]:
                raise NotImplementedError(
                    f"Only C2 symmetry supported. {symmetry} was supplied."
                )
            self.order = int(symmetry[1])

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

        # Allocate local variables for return
        # clmatrix represents the common lines matrix.
        # Namely, clmatrix[i,j] contains the index in image i of
        # the common line with image j. Note the common line index
        # starts from 0 instead of 1 as Matlab version. -1 means
        # there is no common line such as clmatrix[i,i].
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
        # Set resolution of shift estimation in pixels. Note that
        # shift_step can be any positive real number.
        shift_step = self.shift_step
        # 1D shift between common-lines
        shifts_1d = np.zeros((2, n_img, n_img))

        # Prepare the shift phases to try and generate filter for common-line detection
        r_max = pf.shape[2]
        shifts, shift_phases, h = self._generate_shift_phase_and_filter(
            r_max, max_shift, shift_step
        )
        all_shift_phases = shift_phases.T

        # Apply bandpass filter, normalize each ray of each image
        # Note that only use half of each ray
        pf = self._apply_filter_and_norm("ijk, k -> ijk", pf, r_max, h)

        # Search for common lines between [i, j] pairs of images.
        # Creating pf and building common lines are different to the Matlab version.
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
                    # Compute correlations in the positive r direction
                    part1 = p1_real.dot(np.real(p2_shifted_flipped))
                    # Compute correlations in the negative r direction
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
                    dist = self.min_dist_cls * 2 * self.n_theta // 360
                    C_masked = self.square_mask(C, first_cl1, first_cl2, dist)
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
        left = max(0, x - dist)
        right = min(len(arr), x + dist)
        bottom = max(0, y - dist)
        top = min(len(arr[0]), y + dist)
        mask = np.ones_like(arr).astype(arr.dtype)
        mask[left:right, bottom:top] = 0

        return arr * mask

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

        # Step 2: Calculate relative rotations associated with both mutual common lines.
        Rijs, Rijgs = self._estimate_all_Rijs_c2(clmatrix)

        # Step 3: Inner J-synchronization
        vijs, viis = self._local_J_sync_c2(Rijs, Rijgs)

        return vijs, viis

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
                clmatrix[0], i, j, np.arange(n_img), n_theta
            )

        return Rijs, Rijgs

    def _local_J_sync_c2(Rijs, Rijgs):
        pass
