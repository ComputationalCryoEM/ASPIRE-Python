import logging
import os.path

import numpy as np
from numpy.linalg import norm

from aspire.utils import J_conjugate, all_pairs, random, trange

logger = logging.getLogger(__name__)


class JSync:
    """
    Class for handling J-synchronization methods.
    """

    # Initialize alternatives
    #
    # When we find the best J-configuration, we also compare it to the alternative 2nd best one.
    # this comparison is done for every pair in the triplet independently. to make sure that the
    # alternative is indeed different in relation to the pair, we document the differences between
    # the configurations in advance:
    # ALTS(:,best_conf,pair) = the two configurations in which J-sync differs from best_conf in relation to pair

    _ALTS = np.array(
        [
            [[1, 2, 1], [0, 2, 0], [0, 0, 1], [1, 0, 0]],
            [[2, 3, 3], [3, 3, 2], [3, 1, 3], [2, 1, 2]],
        ],
        dtype=int,
    )

    def __init__(
        self,
        n,
        epsilon=1e-2,
        max_iters=1000,
        seed=None,
        disable_gpu=False,
        J_weighting=False,
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
        self.J_weighting = J_weighting

        # Generate pair mappings
        self._pairs, self._pairs_to_linear = all_pairs(n, return_map=True)

        # Auto configure GPU
        self.__gpu_module = None
        if not disable_gpu:
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() >= 1:
                    gpu_id = cp.cuda.runtime.getDevice()
                    logger.info(
                        f"cupy and GPU {gpu_id} found by cuda runtime; enabling cupy."
                    )
                    self.__gpu_module = self.__init_cupy_module()
                else:
                    logger.info("GPU not found, defaulting to numpy.")

            except ModuleNotFoundError:
                logger.info("cupy not found, defaulting to numpy.")

    def global_J_sync(self, Rijs):
        """
        Apply global J-synchronization.

        Given all pairs of estimated rotation matrices `Rijs` with
        arbitrary handedness (J conjugation), attempt to detect and
        conjugate entries of `Rijs` such that all rotations have same
        handedness.

        :param Rijs: Array of all pairs of rotation matrices
        :return: Array of all pairs of J synchronized rotation matrices
        """

        # Determine relative handedness of Rijs.
        sign_ij_J = self.power_method(Rijs)

        # Synchronize Rijs
        logger.info("Applying global handedness synchronization.")
        Rijs_sync = Rijs.copy()
        mask = sign_ij_J == -1
        Rijs_sync[mask] = J_conjugate(Rijs_sync[mask])

        return Rijs_sync

    def power_method(self, Rijs):
        """
        Calculate the leading eigenvector of the J-synchronization matrix
        using the power method.

        As the J-synchronization matrix is of size (n-choose-2)x(n-choose-2), we
        use the power method to compute the eigenvalues and eigenvectors,
        while constructing the matrix on-the-fly.

        :param Rijs: (n-choose-2)x3x3 array of estimates of relative orientation matrices.

        :return: An array of length n-choose-2 consisting of 1 or -1, where the sign of the
        i'th entry indicates whether the i'th relative orientation matrix will be J-conjugated.
        """

        logger.info(
            "Initiating power method to estimate J-synchronization matrix eigenvector."
        )
        # Set power method tolerance and maximum iterations.
        epsilon = self.epsilon
        max_iters = self.max_iters

        # Initialize candidate eigenvectors
        n_Rijs = Rijs.shape[0]
        vec = random(n_Rijs, seed=self.seed)
        vec = vec / norm(vec)
        residual = 1
        itr = 0

        # Todo
        # I don't like that epsilon>1 (residual) returns signs of random vector
        #      maybe force to run once? or return vec as zeros in that case?
        #      Seems unintended, but easy to do.

        # Power method iterations
        while itr < max_iters and residual > epsilon:
            itr += 1
            # Todo, this code code actually needs double precision for accuracy... forcing.
            vec_new = self._signs_times_v(Rijs, vec).astype(np.float64, copy=False)
            vec_new = vec_new / norm(vec_new)
            residual = norm(vec_new - vec)
            vec = vec_new
            logger.info(
                f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
            )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec, dtype=Rijs.dtype)
        J_sync = np.sign(J_sync[0]) * J_sync  # Stabilize J_sync

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

    def _signs_times_v(self, Rijs, vec):
        """
        Multiplication of the J-synchronization matrix by a candidate eigenvector `vec`

        Wrapper for cpu/gpu dispatch.

        :param Rijs: An n-choose-2x3x3 array of estimates of relative rotations
        :param vec: The current candidate eigenvector of length n-choose-2 from the power method.
        :return: New candidate eigenvector.
        """
        # host/gpu dispatch
        if self.__gpu_module:
            new_vec = self._signs_times_v_cupy(Rijs, vec)
        else:
            new_vec = self._signs_times_v_host(Rijs, vec)

        return new_vec.astype(vec.dtype, copy=False)

    def _signs_times_v_host(self, Rijs, vec):
        """
        See `_signs_times_v`.

        CPU implementation.
        """

        new_vec = np.zeros_like(vec)

        _signs_confs = np.array(
            [[1, 1, 1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]], dtype=int
        )

        c = np.empty((4))
        desc = "Computing signs_times_v"
        if self.J_weighting:
            desc += " with J_weighting"
        for i in trange(self.n_img - 2, desc=desc):
            for j in range(
                i + 1, self.n_img - 1
            ):  # check bound (taken from MATLAB mex)
                ij = self._pairs_to_linear[i, j]
                Rij = Rijs[ij]
                for k in range(j + 1, self.n_img):
                    ik = self._pairs_to_linear[i, k]
                    jk = self._pairs_to_linear[j, k]
                    Rik = Rijs[ik]
                    Rjk = Rijs[jk]

                    # Compute conjugated rotats
                    Rij_J = J_conjugate(Rij)
                    Rik_J = J_conjugate(Rik)
                    Rjk_J = J_conjugate(Rjk)

                    # Compute R muls and norms
                    c[0] = np.sum(((Rij @ Rjk) - Rik) ** 2)
                    c[1] = np.sum(((Rij_J @ Rjk) - Rik) ** 2)
                    c[2] = np.sum(((Rij @ Rjk_J) - Rik) ** 2)
                    c[3] = np.sum(((Rij @ Rjk) - Rik_J) ** 2)

                    # Find best match
                    best_i = np.argmin(c)
                    best_val = c[best_i]

                    # MATLAB: scores_as_entries == 0
                    s_ij_jk = _signs_confs[best_i][0]
                    s_ik_jk = _signs_confs[best_i][1]
                    s_ij_ik = _signs_confs[best_i][2]

                    # Note there was a third J_weighting option (2) in MATLAB,
                    # but it was not exposed at top level.
                    if self.J_weighting:
                        # MATLAB: scores_as_entries == 1
                        # For each triangle side, find the best alternative
                        alt_ij_jk = c[self._ALTS[0][best_i][0]]
                        if c[self._ALTS[1][best_i][0]] < alt_ij_jk:
                            alt_ij_jk = c[self._ALTS[1][best_i][0]]

                        alt_ik_jk = c[self._ALTS[0][best_i][1]]
                        if c[self._ALTS[1][best_i][1]] < alt_ik_jk:
                            alt_ik_jk = c[self._ALTS[1][best_i][1]]

                        alt_ij_ik = c[self._ALTS[0][best_i][2]]
                        if c[self._ALTS[1][best_i][2]] < alt_ij_ik:
                            alt_ij_ik = c[self._ALTS[1][best_i][2]]

                        # Compute scores
                        s_ij_jk *= 1 - np.sqrt(best_val / alt_ij_jk)
                        s_ik_jk *= 1 - np.sqrt(best_val / alt_ik_jk)
                        s_ij_ik *= 1 - np.sqrt(best_val / alt_ij_ik)

                    # Update vector entries
                    new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
                    new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
                    new_vec[ik] += s_ij_ik * vec[ij] + s_ik_jk * vec[jk]

        return new_vec

    def _signs_times_v_cupy(self, Rijs, vec):
        """
        See `_signs_times_v`.

        CPU implementation.
        """
        import cupy as cp

        signs_times_v = self.__gpu_module.get_function("signs_times_v")

        Rijs_dev = cp.array(Rijs, dtype=np.float64)
        vec_dev = cp.array(vec, dtype=np.float64)
        new_vec_dev = cp.zeros((vec.shape[0]), dtype=np.float64)

        # call the kernel
        blkszx = 512
        nblkx = (self.n_img + blkszx - 1) // blkszx
        signs_times_v(
            (nblkx,),
            (blkszx,),
            (self.n_img, Rijs_dev, vec_dev, new_vec_dev, self.J_weighting),
        )

        # dtoh
        new_vec = new_vec_dev.get().astype(vec.dtype, copy=False)

        return new_vec

    @staticmethod
    def __init_cupy_module():
        """
        Private utility method to read in CUDA source and return as
        compiled CUPY module.
        """

        import cupy as cp

        # Read in contents of file
        src_dir = os.path.dirname(__file__)
        fp = os.path.join(src_dir, "J_sync.cu")
        with open(fp, "r") as fh:
            module_code = fh.read()

        # CUPY compile the CUDA code
        return cp.RawModule(
            code=module_code,
            backend="nvcc",
            options=("-I" + src_dir,),  # inject path for common_kernels
        )
