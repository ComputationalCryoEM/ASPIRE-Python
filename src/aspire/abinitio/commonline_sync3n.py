import logging
import os.path
import warnings

import numpy as np
from numpy.linalg import norm
from scipy.optimize import curve_fit

from aspire.abinitio import CLOrient3D, SyncVotingMixin
from aspire.utils import (
    J_conjugate,
    Rotation,
    all_pairs,
    nearest_rotations,
    tqdm,
    trange,
)
from aspire.utils.matlab_compat import stable_eigsh
from aspire.utils.random import rand

logger = logging.getLogger(__name__)


class CLSync3N(CLOrient3D, SyncVotingMixin):
    """
    Define a class to estimate 3D orientations using common lines Sync3N methods (2017).

    Ido Greenberg, Yoel Shkolnisky,
    Common lines modeling for reference free Ab-initio reconstruction in cryo-EM,
    Journal of Structural Biology,
    Volume 200, Issue 2,
    2017,
    Pages 106-117,
    ISSN 1047-8477,
    https://doi.org/10.1016/j.jsb.2017.09.007.
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
        src,
        n_rad=None,
        n_theta=360,
        max_shift=0.15,
        shift_step=1,
        hist_bin_width=1,
        full_width="adaptive",
        epsilon=1e-2,
        max_iters=1000,
        seed=None,
        mask=True,
        S_weighting=False,
        J_weighting=False,
        hist_intervals=100,
        disable_gpu=False,
    ):
        """
        Initialize object for estimating 3D orientations.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param max_shift: Maximum range for shifts as a proportion of box size. Default = 0.15.
        :param shift_step: Step size of shift estimation in pixels. Default = 1 pixel.
        :param hist_bin_width: Bin width in smoothing histogram (degrees).
        :param full_width: Selection width around smoothed histogram peak (degrees).
            `adaptive` will attempt to automatically find the smallest number of
            `hist_bin_width`s required to find at least one valid image index.
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        :param seed: Optional seed for RNG.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        :param S_weighting: Optionally apply probabilistic weighting
            to the `S` matrix.
        :param J_weighting: Optionally use `J` weights instead of
            signs when computing `signs_times_v`.
        :param hist_intervals: Number of histogram bins used to
            compute triangle scores when `S_weighting` enabled.
        :param disable_gpu: Disables GPU acceleration;
            forces CPU only code for this module.
            Defaults to automatically using GPU when available.
        """

        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            hist_bin_width=hist_bin_width,
            full_width=full_width,
            mask=mask,
        )

        # Generate pair mappings
        self._pairs, self._pairs_to_linear = all_pairs(self.n_img, return_map=True)

        self.epsilon = epsilon
        self.max_iters = max_iters
        self.seed = seed

        # Sync3N specific vars
        self.S_weighting = S_weighting
        self.J_weighting = J_weighting
        self._D_null = 1e-13
        self.hist_intervals = int(hist_intervals)
        # Warn if histogram may be too sparse for curve fitting
        if self.S_weighting and (src.n < hist_intervals):
            logger.warning(
                f"`hist_intervals` {hist_intervals} > src.n {src.n}."
                "  Consider reducing if curve fitting is infeasable."
            )

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

    ###########################################
    # High level algorithm steps              #
    ###########################################
    def estimate_rotations(self):
        """
        Estimate rotation matrices.

        :return: Array of rotation matrices, size n_imgx3x3.
        """

        logger.info(f"Estimating relative viewing directions for {self.n_img} images.")

        # Detect a single pair of common-lines between each pair of images
        self.build_clmatrix()

        # Initial estimate of viewing directions
        # Calculate relative rotations
        Rijs0 = self._estimate_all_Rijs(self.clmatrix)

        # Compute and apply global handedness
        Rijs = self._global_J_sync(Rijs0)

        # Build sync3n matrix
        S = self._construct_sync3n_matrix(Rijs)

        # Optionally compute S weights
        W = None
        if self.S_weighting is True:
            W = self._syncmatrix_weights(Rijs)

        # Yield rotations from S
        self.rotations = self._sync3n_S_to_rot(S, W)

    #######################
    # Main Sync3N Methods #
    #######################
    def _sync3n_S_to_rot(self, S, W=None, n_eigs=4):
        """
        Use eigen decomposition of S to estimate transforms,
        then project transforms to nearest rotations.

        :param S: Numpy array representing Synchronization matrix.
        :param W: Optional weights array, default `None` is equal weighting of `S`.
        :param n_eigs: Optional, number of eigenvalues to compute (min 3).
        """

        # Critical this occurs in double precision
        S = S.astype(np.float64, copy=False)

        if n_eigs < 3:
            raise ValueError(
                f"n_eigs must be greater than 3, default is 4. Invoked with {n_eigs}"
            )

        if W is not None:
            logger.info("Applying weights to synchronization matrix.")
            if not W.shape == (self.n_img, self.n_img):
                raise RuntimeError(
                    f"Shape of W should be {(self.n_img, self.n_img)}."
                    f" Received {W.shape}."
                )
            # Initialize D
            # Critical this occurs in double precision
            W = W.astype(np.float64, copy=False)
            D = np.mean(W, axis=1)

            Dhalf = D
            # Compute mask of trouble D values
            nulls = np.abs(D) < self._D_null
            # Avoid trouble values when exponentiating
            Dhalf[~nulls] = Dhalf[~nulls] ** (-0.5)
            # Flush trouble values to zero
            Dhalf[nulls] = 0
            # expand diagonal
            Dhalf = np.diag(Dhalf)

            # Report W Diagnostic
            W_normalized = Dhalf**2 @ W
            nzidx = np.sum(W_normalized, axis=1) != 0
            err = np.linalg.norm(np.sum(W_normalized[nzidx], axis=1) - self.n_img)
            if err > 1e-10:
                logger.warning(f"Large Weights Matrix Normalization Error: {err}")

            # Make W of size 3Nx3N
            W = np.kron(W, np.ones((3, 3), dtype=self.dtype))

            # Make Dhalf of size 3Nx3N
            Dhalf = np.diag(np.kron(np.diag(Dhalf), np.ones(3, dtype=np.float64)))

            # Apply weights to S
            S = Dhalf @ (W * S) @ Dhalf

        # Extract three eigenvectors corresponding to non-zero eigenvalues.
        d, v = stable_eigsh(S, n_eigs, which="LM")

        sort_idx = np.argsort(-d)
        logger.info(
            f"Top {n_eigs} eigenvalues from synchronization voting matrix: {d[sort_idx]}"
        )

        # Only need the top 3 eigen-vectors.
        v = v[:, sort_idx[:3]]

        # Cancel symmetrization when using weights W
        if W is not None:
            # Untill now we used a symmetrized variant of the weighted Sync matrix,
            # thus we didn't get the right eigenvectors. to fix that we just need
            # to multiply:
            v = Dhalf @ v

        # Yield estimated rotations from the eigen-vectors
        rotations = v.reshape(self.n_img, 3, 3).transpose(0, 2, 1)

        # Enforce we are returning actual rotations
        rotations = nearest_rotations(rotations, allow_reflection=True)

        return rotations.astype(self.dtype)

    def _construct_sync3n_matrix(self, Rij):
        """
        Construct sync3n matrix from estimated rotations Rij.

        :param Rij: Numpy array of estimated rotations (all pairs).
        :return: Synchronization matrix S, (3*N, 3*N).
        """

        # Initialize S with diag identity blocks
        n = self.n_img
        S = np.eye(3 * n, dtype=self.dtype).reshape(n, 3, n, 3)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # S( (3*i-2):(3*i) , (3*j-2):(3*j) ) = Rij(:,:,idx); % Rij
                S[i, :, j, :] = Rij[idx]
                # S( (3*j-2):(3*j) , (3*i-2):(3*i) ) = Rij(:,:,idx)'; % Rji = Rij'
                S[j, :, i, :] = Rij[idx].T
                idx += 1

        # Convert S shape to 3Nx3N
        S = S.reshape(3 * n, 3 * n)

        return S

    def _syncmatrix_weights(
        self,
        Rijs,
        permitted_inconsistency=1.5,
        p_domain_limit=0.7,
        max_iterations=12,
        min_p_permitted=0.04,
    ):
        """
        Given relative rotations matrix `Rij`,
        compute and return probability weights `W` for S.

        Default parameters here were taken from those in the MATLAB
        code, with the original author noting they were found
        empirically.

        :param permitted_inconsistency: Consistency condition is
            `mean(Pij)/permitted_inconsistency < P <
            mean(Pij)*permitted_inconsistency`.
        :param p_domain_limit: Domain of P is [Pmin,Pmax], with
            Pmin=p_domain_limit*Pmax
        :param max_iterations: Maximum iterations for P estimation.
        :param min_p_permitted: Small value at which to stop
            attempting to synchronize P.
        :return: Synchronization matrix weights `W`.
        """
        logger.info("Computing synchronization matrix weights.")

        def _body(prev_too_low, Pmin, Pmax, hist, p_domain_limit=p_domain_limit):
            """
            Helper function to run and test triangle_scores.
            """
            # Get inistial estimate for Pij
            P, sigma, Pij, hist = self._triangle_scores(Rijs, hist, Pmin, Pmax)

            # Check if P and Pij are consistent
            mean_Pij = np.mean(Pij)
            too_low = P < mean_Pij / permitted_inconsistency
            too_high = P > mean_Pij * permitted_inconsistency
            inconsistent = too_low | too_high

            # Check trend
            if prev_too_low is not None and too_low != prev_too_low:
                p_domain_limit = np.sqrt(p_domain_limit)

            # define limits for next P estimation
            if too_high:
                if P < min_p_permitted:
                    logger.error(
                        "Triangles Scores are poorly distributed, whatever small P we force."
                    )

                if Pmax is not None:
                    Pmax = Pmax * p_domain_limit
                else:
                    Pmax = P

                Pmin = Pmax * p_domain_limit
            else:  # too low
                if Pmin is not None:
                    Pmin = Pmin / p_domain_limit
                else:
                    Pmin = P

                Pmax = Pmin / p_domain_limit

            return inconsistent, Pij, (too_low, Pmin, Pmax, hist)

        # Repeat iteratively until estimations of P & Pij are consistent
        i = 0
        res = (None,) * 4
        inconsistent = True
        while inconsistent and i < max_iterations:
            inconsistent, Pij, res = _body(*res)
            i += 1

        # Pack W
        W = np.zeros((self.n_img, self.n_img), dtype=self.dtype)
        idx = 0
        for i in range(self.n_img):
            for j in range(i + 1, self.n_img):
                W[i, j] = Pij[idx]
                W[j, i] = Pij[idx]
                idx += 1

        return W

    def _triangle_scores_inner(self, Rijs):
        """
        Computes histogram of `triangle scores`.

        Wrapper for cpu/gpu dispatch.

        :param Rijs: nchoose2 by 3 by 3 array of rotations.
        :return: Histogram of triangle scores.
        """

        # host/gpu dispatch
        if self.__gpu_module:
            scores_hist = self._triangle_scores_inner_cupy(Rijs)
        else:
            scores_hist = self._triangle_scores_inner_host(Rijs)

        return scores_hist

    def _triangle_scores_inner_host(self, Rijs):
        """
        See _triangle_scores_inner.

        CPU implementation.
        """

        # The following is adopted from Matlab triangle_scores_mex.c

        # Initialize probability result arrays
        scores_hist = np.zeros(self.hist_intervals, dtype=np.uint32)

        c = np.empty((4), dtype=Rijs.dtype)
        s = np.empty((3), dtype=Rijs.dtype)
        for i in trange(self.n_img - 2, desc="Computing triangle scores"):
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
                    s[0] = 1 - np.sqrt(best_val / alt_ij_jk)  # s_ij_jk
                    s[1] = 1 - np.sqrt(best_val / alt_ik_jk)  # s_ik_jk
                    s[2] = 1 - np.sqrt(best_val / alt_ij_ik)  # s_ij_ik

                    # Update histogram
                    # Find integer bin [0,self.hist_intervals)
                    _l1, _l2, _l3 = np.maximum(
                        np.minimum(
                            (self.hist_intervals * s).astype(int),  # implicit floor
                            self.hist_intervals - 1,  # clamp upper bound
                        ),
                        0,  # clamp lower bound
                    )

                    scores_hist[_l1] += 1
                    scores_hist[_l2] += 1
                    scores_hist[_l3] += 1

        return scores_hist

    def _triangle_scores_inner_cupy(self, Rijs):
        """
        See _triangle_scores_inner.

        GPU implementation.
        """

        import cupy as cp

        triangle_scores = self.__gpu_module.get_function("triangle_scores_inner")

        Rijs_dev = cp.array(Rijs, dtype=np.float64)

        # This holds integer counts
        scores_hist_dev = cp.zeros((self.hist_intervals), dtype=np.uint32)

        # call the kernel
        blkszx = 512
        nblkx = (self.n_img + blkszx - 1) // blkszx
        triangle_scores(
            (nblkx,),
            (blkszx,),
            (
                self.n_img,
                Rijs_dev,
                self.hist_intervals,
                scores_hist_dev,
            ),
        )

        # d2h
        scores_hist = scores_hist_dev.get()

        return scores_hist

    def _pairs_probabilities(self, Rijs, P2, A, a, B, b, x0):
        """
        This function computes the probability of a pair `ij` having
        an observed value of triangles score under two priors.  Once
        given it has an indicative common line, and again once given
        it has an arbitrary common line.

        The probability of the common line to be indicative can then
        be derived by Bayes Theorem.

        Wrapper for cpu/gpu dispatch.

        :param Rijs: nchoose2 by 3 by 3 array of rotations.
        :param P2: distribution parameter
        :param A: distribution parameter
        :param a: distribution parameter
        :param B: distribution parameter
        :param b: distribution parameter
        :param x0: Initial guess
        :return: (log indicative probabilities, log arbitrary probabilities)
        """
        # These param values are passed to C, force doubles.
        params = np.array([P2, A, a, B, b, x0], dtype=np.float64)

        # host/gpu dispatch
        if self.__gpu_module:
            ln_f_ind, ln_f_arb = self._pairs_probabilities_cupy(Rijs, *params)
        else:
            ln_f_ind, ln_f_arb = self._pairs_probabilities_host(Rijs, *params)

        return ln_f_ind, ln_f_arb

    def _pairs_probabilities_host(self, Rijs, P2, A, a, B, b, x0):
        """
        See _pairs_probabilities.

        CPU implementation.
        """
        # The following is adopted from Matlab pairs_probabilities_mex.c `looper`

        # Initialize probability result arrays
        ln_f_ind = np.zeros(len(Rijs), dtype=Rijs.dtype)
        ln_f_arb = np.zeros(len(Rijs), dtype=Rijs.dtype)

        c = np.empty((4), dtype=Rijs.dtype)
        for i in trange(self.n_img - 2, desc="Computing pair probabilities"):
            for j in range(i + 1, self.n_img - 1):
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
                    s_ij_jk = 1 - np.sqrt(best_val / alt_ij_jk)
                    s_ik_jk = 1 - np.sqrt(best_val / alt_ik_jk)
                    s_ij_ik = 1 - np.sqrt(best_val / alt_ij_ik)

                    # Update probabilities
                    # # Probability of pair ij having score given indicicative common line
                    # P2, B, b, x0, A, a
                    f_ij_jk = np.log(
                        P2
                        * (
                            B
                            * np.power(1 - s_ij_jk, b)
                            * np.exp(-b / (1 - x0) * (1 - s_ij_jk))
                        )
                        + (1 - P2) * A * np.power((1 - s_ij_jk), a)
                    )
                    f_ik_jk = np.log(
                        P2
                        * (
                            B
                            * np.power(1 - s_ik_jk, b)
                            * np.exp(-b / (1 - x0) * (1 - s_ik_jk))
                        )
                        + (1 - P2) * A * np.power((1 - s_ik_jk), a)
                    )
                    f_ij_ik = np.log(
                        P2
                        * (
                            B
                            * np.power(1 - s_ij_ik, b)
                            * np.exp(-b / (1 - x0) * (1 - s_ij_ik))
                        )
                        + (1 - P2) * A * np.power((1 - s_ij_ik), a)
                    )
                    ln_f_ind[ij] += f_ij_jk + f_ij_ik
                    ln_f_ind[jk] += f_ij_jk + f_ik_jk
                    ln_f_ind[ik] += f_ik_jk + f_ij_ik

                    # # Probability of pair ij having score given arbitrary common line
                    f_ij_jk = np.log(A * np.power((1 - s_ij_jk), a))
                    f_ik_jk = np.log(A * np.power((1 - s_ik_jk), a))
                    f_ij_ik = np.log(A * np.power((1 - s_ij_ik), a))
                    ln_f_arb[ij] += f_ij_jk + f_ij_ik
                    ln_f_arb[jk] += f_ij_jk + f_ik_jk
                    ln_f_arb[ik] += f_ik_jk + f_ij_ik

        return ln_f_ind, ln_f_arb

    def _pairs_probabilities_cupy(self, Rijs, P2, A, a, B, b, x0):
        """
        See _pairs_probabilities.

        GPU implementation.
        """

        import cupy as cp

        pairs_probabilities = self.__gpu_module.get_function("pairs_probabilities")

        Rijs_dev = cp.array(Rijs, dtype=np.float64)
        ln_f_ind_dev = cp.zeros((self.n_img * (self.n_img - 1) // 2), dtype=np.float64)
        ln_f_arb_dev = cp.zeros((self.n_img * (self.n_img - 1) // 2), dtype=np.float64)

        # call the kernel
        blkszx = 512
        nblkx = (self.n_img + blkszx - 1) // blkszx
        pairs_probabilities(
            (nblkx,),
            (blkszx,),
            (self.n_img, Rijs_dev, P2, A, a, B, b, x0, ln_f_ind_dev, ln_f_arb_dev),
        )

        # accumulate over thread results
        ln_f_arb = ln_f_arb_dev.get().astype(self.dtype, copy=False)
        ln_f_ind = ln_f_ind_dev.get().astype(self.dtype, copy=False)

        return ln_f_ind, ln_f_arb

    def _triangle_scores(
        self,
        Rijs,
        scores_hist,
        Pmin,
        Pmax,
        a=2.2,
        peak2sigma=2.43e-2,
        P=0.5,
        b=2.5,
        x0=0.78,
    ):
        """
        Computes `triangle_scores`, attempts to fit curve to
        distribution, and uses estimated distribution to compute
        `pairs_probabilities`.

        Default parameters here were taken from those in the MATLAB
        code, with the original author noting they were found
        empirically.

        :param a: distribution parameter
        :param peak2sigma: empirical relation between the location of
            the peak of the histigram, and the mean error in the
            common lines estimations.
        :param P: distribution parameter
        :param b: distribution parameter
        :param x0: Initial guess
        :return: Tuple of pairs probabilty Pij and related terms
             (P, sigma, Pij, scores_hist)
        """

        Pmin = Pmin or 0
        Pmin = max(Pmin, 0)  # Clamp probability to [0,1]
        Pmax = Pmax or 1
        Pmax = min(Pmax, 1)  # Clamp probability to [0,1]

        if scores_hist is None:
            scores_hist = self._triangle_scores_inner(Rijs)

        # Histogram decomposition: P & sigma evaluation
        h = 1 / self.hist_intervals
        hist_x = np.arange(h / 2, 1, h)
        # normalization factor of one component of the histogram
        A = (
            (self.n_img * (self.n_img - 1) * (self.n_img - 2) / 2)
            / self.hist_intervals
            * (a + 1)
        )
        # normalization of 2nd component: B = P*N_delta/sum(f), where f is the component formula
        # B0 = (
        #     P
        #     * (self.n_img * (self.n_img - 1) * (self.n_img - 2) / 2)
        #     / np.sum(((1 - hist_x) ** b) * np.exp(-b / (1 - x0) * (1 - hist_x)))
        # )
        # P must be in lower and upper bounds or `curve_fit` will error
        # This was not the case for MATLAB...
        # P0 = np.clip(P, Pmin**3, Pmax**3)
        # Note, MATLAB suggests the following, but I feel it is a bug.
        # Will discuss with Yoel about the original code's intent.
        #     np.array([B0, P0, b, x0], dtype=np.float64)
        start_values = None
        lower_bounds = np.array([0, Pmin**3, 2, 0], dtype=np.float64)
        upper_bounds = np.array([np.inf, Pmax**3, np.inf, 1], dtype=np.float64)

        with np.printoptions(precision=2):
            logger.info(f"curve_fit lower_bounds:{lower_bounds}")
            logger.info(f"curve_fit start_values:{start_values}")
            logger.info(f"curve_fit upper_bounds:{upper_bounds}")

        # Fit distribution
        def fun(x, B, P, b, x0, A=A, a=a):
            """Function to fit. x is data vector."""
            return (1 - P) * A * (1 - x) ** a + P * B * (1 - x) ** b * np.exp(
                -b / (1 - x0) * (1 - x)
            )

        popt, pcov = curve_fit(
            fun,
            hist_x.astype(np.float64, copy=False),
            scores_hist.astype(np.float64, copy=False),
            p0=start_values,
            bounds=(lower_bounds, upper_bounds),
            method="trf",  # MATLAB used method "LAR" with algo "Trust-Region"
        )
        B, P, b, x0 = popt

        # Derive P and sigma
        P = P ** (1 / 3)
        sigma = (1 - x0) / peak2sigma

        logger.info(f"Estimated CL Errors P,STD:\t{100*P:.2f}%\t{sigma:.2f}")

        # Initialize probability computations
        # Local histograms analysis
        A = a + 1  # distribution 1st component normalization factor
        # distribution 2nd component normalization factor
        B = B / (
            (self.n_img * (self.n_img - 1) * (self.n_img - 2) / 2) / self.hist_intervals
        )

        # Calculate probabilities
        ln_f_ind, ln_f_arb = self._pairs_probabilities(Rijs, P**2, A, a, B, b, x0)

        with warnings.catch_warnings():
            # For large values of (ln_f_arb - ln_f_ind), numpy exponential will overflow. We still
            # get the intended result of Pij = 0, so we capture and ignore the overflow warning.
            warnings.filterwarnings("ignore", r".*overflow encountered in exp.*")

            Pij = 1 / (1 + (1 - P) / P * np.exp(ln_f_arb - ln_f_ind))

        # Fix singular output
        num_nan = np.sum(np.isnan(Pij))
        if num_nan > 0:
            logger.error(
                f"NaN probabilities occurred {num_nan} times out of {np.size(Pij)}. Setting NaNs to zero."
            )
            Pij = np.nan_to_num(Pij)

        logger.info(
            f"Common lines probabilities to be indicative Pij={100*np.mean(Pij):.2f}%"
        )

        return P, sigma, Pij, scores_hist

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _global_J_sync(self, Rijs):
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
        sign_ij_J = self._J_sync_power_method(Rijs)

        # Synchronize Rijs
        logger.info("Applying global handedness synchronization.")
        mask = sign_ij_J == -1
        Rijs[mask] = J_conjugate(Rijs[mask])

        return Rijs

    def _estimate_all_Rijs(self, clmatrix):
        """
        Estimate Rijs using the voting method.

        :param clmatrix: Common lines matrix
        :return: Estimated rotations
        """
        n_img = self.n_img
        n_theta = self.n_theta
        Rijs = np.zeros((len(self._pairs), 3, 3))

        for idx, (i, j) in enumerate(tqdm(self._pairs, desc="Estimate Rijs")):
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
        alphas, good_k = self._vote_ij(clmatrix, n_theta, i, j, k_list, sync=True)

        angles = np.zeros(3)

        if alphas is not None:
            angles[0] = clmatrix[i, j] * 2 * np.pi / n_theta + np.pi / 2
            angles[1] = np.mean(alphas)
            angles[2] = -np.pi / 2 - clmatrix[j, i] * 2 * np.pi / n_theta
            rot = Rotation.from_euler(angles).matrices

        else:
            # This is for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # We set to zero as in the Matlab code.
            rot = np.zeros((3, 3))

        return rot

    #######################################
    # Secondary Methods for Global J Sync #
    #######################################

    def _J_sync_power_method(self, Rijs):
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
        vec = rand(n_Rijs, seed=self.seed)
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
        J_sync = np.sign(vec)
        J_sync = np.sign(J_sync[0]) * J_sync  # Stabilize J_sync

        return J_sync

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
        fp = os.path.join(os.path.dirname(__file__), "commonline_sync3n.cu")
        with open(fp, "r") as fh:
            module_code = fh.read()

        # CUPY compile the CUDA code
        return cp.RawModule(code=module_code)
