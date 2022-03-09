import logging

import numpy as np
from numpy import pi
from numpy.linalg import eig, norm
from tqdm import tqdm

from aspire.abinitio import CLOrient3D

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

    def __init__(self, src, n_symm=None, n_rad=None, n_theta=None):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_symm: The symmetry order of the molecule. 3 or 4.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """

        super().__init__(src, n_rad=n_rad, n_theta=n_theta)

        self.n_symm = n_symm
        self.n_ims = self.n_img

    def estimate_rotations(self):
        """
        Estimate rotation matrices for symmetric molecules.
        """
        pass

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _estimate_relative_viewing_directions_c3_c4(self, max_shift_1d):
        """
        Estimate the relative viewing directions vij = vi'vj, i<j, and vii = vi'vi, where
        vi is the third row of the i'th rotation matrix Ri.
        """

        n_symm = self.n_symm
        n_theta = self.n_theta

        # Step 1: Detect a single pair of common-lines between each pair of images

        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        if n_symm == 3:
            is_handle_equator_ims = False
        else:
            is_handle_equator_ims = True

        sclmatrix, corrs_stats, shifts_stats = self._self_clmatrix_c3_c4(
            is_handle_equator_ims
        )

        # Step 3: Calculate self-relative-rotations
        Riis = self._estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta)

        # Step 4: Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta)

        # Step 5: Inner J-synchronization
        vijs, viis = self._local_sync_J_c3_c4(n_symm, Rijs, Riis)

        return vijs, viis

    def _global_J_sync(self, vijs, viis):
        """
        Global J-synchronization of all third row outer products. Given 3x3 matrices vijs and viis, each
        of which might contain a spurious J, we return vijs and viis that all have either a spurious J
        or not.

        :param vijs: An nchoose2x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of matrices Ri and Rj. Each estimate might have a
        spurious J independently of other estimates.

        :param viis: An nx3x3 array where the ith slice holds an estimate for the outer product vi*vi^T
        between the third row of matrix Ri and itself. Each estimate might have a spurious J independently
        of other estimates.

        :return: vijs, viis all of which have a spurious J or not.
        """

        n_ims = viis.shape[0]
        n_vijs = vijs.shape[0]
        nchoose2 = int(n_ims * (n_ims - 1) / 2)
        assert viis.shape[1:] == (3, 3), "viis must be 3x3 matrices."
        assert vijs.shape[1:] == (3, 3), "vijs must be 3x3 matrices."
        assert n_vijs == nchoose2, "There must be n_ims-choose-2 vijs."

        # Determine relative handedness of vijs.
        sign_ij_J = self._J_sync_power_method(vijs)
        n_signs = len(sign_ij_J)
        assert (
            n_signs == n_vijs
        ), f"There must be a sign associated with each vij. There are {n_signs} signs and {n_vijs} vijs."

        # Synchronize vijs
        J = np.diag((-1, -1, 1))
        for i in range(n_signs):
            if sign_ij_J[i] == -1:
                vijs[i] = J @ vijs[i] @ J

        # Synchronize viis
        # We use the fact that if v_ii and v_ij are of the same handedness, then v_ii @ v_ij = v_ij.
        # If they are opposite handed then Jv_iiJ @ v_ij = v_ij. We compare each v_ii against all
        # previously synchronized v_ij to get a consensus on the handedness of v_ii.

        # All pairs (i,j) where i<j
        indices = np.arange(n_ims)
        pairs = [(i, j) for idx, i in enumerate(indices) for j in indices[idx + 1 :]]

        for i in range(n_ims):
            vii = viis[i]
            J_consensus = 0
            for j in range(n_ims):
                if j < i:
                    idx = pairs.index((j, i))
                    vji = vijs[idx]

                    err1 = norm(vji @ vii - vji)
                    err2 = norm(vji @ J @ vii @ J - vji)

                elif j > i:
                    idx = pairs.index((i, j))
                    vij = vijs[idx]

                    err1 = norm(vii @ vij - vij)
                    err2 = norm(J @ vii @ J @ vij - vij)

                else:
                    continue

                # Accumulate J consensus
                if err1 < err2:
                    J_consensus -= 1
                else:
                    J_consensus += 1

            if J_consensus > 0:
                viis[i] = J @ viis[i] @ J
        return vijs, viis

    def _estimate_third_rows(self, vijs, viis):
        """
        Find the third row of each rotation matrix given third row outer products.

        :param vijs: An n-choose-2x3x3 array where each 3x3 slice holds the third rows
        outer product of the corresponding pair of matrices.

        :param viis: An nx3x3 array where the i-th 3x3 slice holds the outer product of
        the third row of Ri with itself.

        :param n_symm: The underlying molecular symmetry.

        :return: vis, An n_imagesx3 matrix whose i-th row is the third row of the rotation matrix Ri.
        """

        n_ims = viis.shape[0]
        n_vijs = vijs.shape[0]
        nchoose2 = int(n_ims * (n_ims - 1) / 2)
        assert viis.shape[1:] == (3, 3), "viis must be 3x3 matrices."
        assert vijs.shape[1:] == (3, 3), "vijs must be 3x3 matrices."
        assert n_vijs == nchoose2, "There must be n_ims-choose-2 vijs."

        # Build 3nx3n matrix V whose (i,j)-th block of size 3x3 holds the outer product vij
        V = np.zeros((3 * n_ims, 3 * n_ims), dtype=vijs.dtype)

        # All pairs (i,j) where i<j
        indices = np.arange(n_ims)
        pairs = [(i, j) for idx, i in enumerate(indices) for j in indices[idx + 1 :]]

        # Populate upper triangle of V with vijs
        for idx, (i, j) in enumerate(pairs):
            V[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = vijs[idx]

        # Populate lower triangle of V with vjis, where vji = vij^T
        V = V + V.T

        # Populate diagonal of V with viis
        for i in range(n_ims):
            V[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = viis[i]

        # In a clean setting V is of rank 1 and its eigenvector is the concatenation
        # of the third rows of all rotation matrices.
        # In the noisy setting we use the eigenvector corresponding to the leading eigenvalue
        val, vec = eig(V)
        lead_idx = np.argsort(val)[-1]
        lead_vec = vec[:, lead_idx]

        vis = lead_vec.reshape((n_ims, 3))
        for i in range(n_ims):
            vis[i] = vis[i] / norm(vis[i])

        return vis

    def _estimate_inplane_rotations(
        self, pf, vis, inplane_rot_res, max_shift, shift_step
    ):
        # return rots
        pass

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def _self_clmatrix_c3_c4(self, is_handle_equator_ims):
        """
        Find the single pair of self-common-lines in each image assuming that the underlying
        symmetry is C3 or C4.
        """
        pf = self.pf.copy()
        n_ims = self.n_ims
        n_theta = self.n_theta
        max_shift_1d = np.ceil(2 * np.sqrt(2) * self.max_shift)
        shift_step = self.shift_step
        n_symm = self.n_symm
        assert n_symm in [3, 4], f"n_symm must be 3 or 4. Got n_symm:{n_symm}."

        # The angle between self-common-lines is in the range [60, 180] for C3 symmetry
        # and [90, 180] for C4 symmetry. Since antipodal lines are perfectly correlated
        # we search for common lines in a smaller window.
        if n_symm == 3:
            min_angle_diff = 60 * pi / 180
            max_angle_diff = 165 * pi / 180
        else:
            min_angle_diff = 90 * pi / 180
            max_angle_diff = 160 * pi / 180

        # The self-common-lines matrix holds two indices per image that represent
        # the two self common-lines in the image.
        sclmatrix = np.zeros((n_ims, 2))
        corrs_stats = np.zeros(n_ims)
        shifts_stats = np.zeros(n_ims)

        # We create a mask associated with angle differences that fall in the
        # range [min_angle_diff, max_angle_diff].
        X, Y = np.meshgrid(range(n_theta), range(n_theta // 2))
        diff = Y - X
        unsigned_angle_diff = np.arccos(np.cos(diff * 2 * pi / n_theta))
        good_diffs = np.logical_and(
            min_angle_diff < unsigned_angle_diff, unsigned_angle_diff < max_angle_diff
        )

        # Compute the correlation over all shifts.
        # Generate Shifts.
        r_max = pf.shape[0]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step
        )
        n_shifts = len(shifts)
        all_shift_phases = shift_phases.T

        # Transpose pf and reconstruct the full polar Fourier for use in correlation.
        # self.pf only consists of rays in the range [180, 360).
        pf = pf.transpose((2, 1, 0))
        pf_full = np.concatenate((np.conjugate(pf), pf), axis=1)

        for i in tqdm(range(n_ims)):
            pf_i = pf[i]
            pf_full_i = pf_full[i]

            # Generate shifted versions of images.
            pf_i_shifted = np.array(
                [pf_i * shift_phase for shift_phase in all_shift_phases]
            )
            pf_i_shifted = np.reshape(pf_i_shifted, (-1, r_max))

            # Normalize each ray.
            for ray in pf_full_i:
                ray /= norm(ray)
            for ray in pf_i_shifted:
                ray /= norm(ray)

            # Compute correlation.
            corrs = np.dot(pf_i_shifted, pf_full_i.T)
            corrs = np.reshape(corrs, (n_shifts, n_theta // 2, n_theta))

            # Mask with allowed combinations.
            corrs = np.array([corr * good_diffs for corr in corrs])

            # Find maximum correlation.
            shift, scl1, scl2 = np.unravel_index(np.argmax(np.real(corrs)), corrs.shape)
            sclmatrix[i] = [scl1, scl2]
            corrs_stats[i] = np.real(corrs[(shift, scl1, scl2)])
            shifts_stats[i] = shift

        return sclmatrix, corrs_stats, shifts_stats

    def _estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta):
        # return Riis
        pass

    def _estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta):
        # return Rijs
        pass

    def local_sync_J_c3_c4(n_symm, Rijs, Riis):
        # return vijs, viis
        pass

    #######################################
    # Secondary Methods for Global J Sync #
    #######################################

    def _J_sync_power_method(self, vijs):
        """
        Calculate the leading eigenvector of the J-synchronization matrix
        using the power method.

        As the J-synchronization matrix is of size (N choose 2)x(N choose 2), we
        use the power method to the compute the eigenvalues and eigenvectors,
        while constructing the matrix on-the-fly.

        :param vijs: nchoose2x3x3 array of estimates of relative orientation matrices.

        :return: Array of length N-choose-2 where the i-th entry indicates if vijs[i]
        should be J-conjugated or not to achieve global handedness consistency. This array
        consists only of +1 and -1.
        """

        n_vijs = vijs.shape[0]
        nchoose2 = (1 + np.sqrt(1 + 8 * n_vijs)) / 2
        assert nchoose2 == int(nchoose2), "There must be n_ims-choose-2 vijs."
        # assert n_eigs > 0, "n_eigs must be a positive integer."

        epsilon = 1e-2
        max_iters = 100

        # Initialize candidate eigenvectors
        vec = np.random.randn(n_vijs)
        vec = vec / norm(vec)
        dd = 1
        itr = 0

        # Power method iterations
        while itr < max_iters and dd > epsilon:
            itr += 1
            vec_new = self._signs_times_v(vijs, vec)
            # vec_new, eigenvalues = qr(vec_new)
            vec_new = vec_new / norm(vec_new)
            dd = norm(vec_new - vec)
            vec = vec_new

        logger.info(
            f"Power method used {itr} iterations. Maximum iterations set to {max_iters}."
        )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec)

        return J_sync

    def _signs_times_v(self, vijs, vec):
        """
        For each triplet of outer products vij, vjk, and vik, the associated elements of the "signs"
        matrix are populated with +1 or -1 and multiplied by the corresponding elements of
        the current candidate eigenvector supplied by the power method. The new candidate eigenvector
        is updated for each triplet.

        :param vijs: Nchoose2 x 3 x 3 array, where each 3x3 slice holds the outer product of vi and vj.

        :param vec: The current candidate eigenvector of length Nchoose2 from the power method.

        :return: New candidate eigenvector of length Nchoose2. The product of the signs matrix and vec.
        """
        # All pairs (i,j) and triplets (i,j,k) where i<j<k
        indices = np.arange(self.n_ims)
        pairs = [(i, j) for idx, i in enumerate(indices) for j in indices[idx + 1 :]]
        trips = [
            (i, j, k)
            for idx, i in enumerate(indices)
            for j in indices[idx + 1 :]
            for k in indices[j + 1 :]
        ]

        # There are four possible signs configurations for each triplet of nodes vij, vik, vjk.
        signs = np.zeros((4, 3))
        signs[0] = [1, 1, 1]
        signs[1] = [-1, 1, -1]
        signs[2] = [-1, -1, 1]
        signs[3] = [1, -1, -1]

        J = np.diag((-1, -1, 1))
        v = vijs
        new_vec = np.zeros_like(vec)

        for _, (i, j, k) in enumerate(trips):
            ij = pairs.index((i, j))
            jk = pairs.index((j, k))
            ik = pairs.index((i, k))

            # Conditions for relative handedness. The minimum of these conditions determines
            # the relative handedness of the triplet of vijs.
            c = np.zeros(4)
            c[0] = norm(v[ij] @ v[jk] - v[ik])
            c[1] = norm(J @ v[ij] @ J @ v[jk] - v[ik])
            c[2] = norm(v[ij] @ J @ v[jk] @ J - v[ik])
            c[3] = norm(v[ij] @ v[jk] - J @ v[ik] @ J)

            min_c = np.argmin(c)

            # Assign signs +-1 to edges between nodes vij, vik, vjk.
            s_ij_jk = signs[min_c][0]
            s_ik_jk = signs[min_c][1]
            s_ij_ik = signs[min_c][2]

            # Update multiplication of signs times vec
            new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
            new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
            new_vec[ik] += s_ij_jk * vec[ij] + s_ik_jk * vec[jk]

        return new_vec
