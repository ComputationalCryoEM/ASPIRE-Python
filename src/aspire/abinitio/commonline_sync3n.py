import logging

import numpy as np
from numpy.linalg import norm
from scipy.optimize import curve_fit

from aspire.abinitio import CLOrient3D, SyncVotingMixin
from aspire.utils import J_conjugate, all_pairs, nearest_rotations, trange
from aspire.utils.matlab_compat import stable_eigsh
from aspire.utils.random import randn

logger = logging.getLogger(__name__)

# Initialize alternatives
#
# When we find the best J-configuration, we also compare it to the alternative 2nd best one.
# this comparison is done for every pair in the triplete independently. to make sure that the
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


class CLSync3N(CLOrient3D, SyncVotingMixin):
    """
    Define a class to estimate 3D orientations using common lines Sync3N methods (2017).
    """

    def __init__(
        self,
        src,
        n_rad=None,
        n_theta=None,
        max_shift=0.15,
        shift_step=1,
        epsilon=1e-2,
        max_iters=1000,
        degree_res=1,
        seed=None,
        mask=True,
        S_weighting=False,
        J_weighting=False,
    ):
        """
        Initialize object for estimating 3D orientations.

        :param src: The source object of 2D denoised or class-averaged images with metadata
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

        # Generate pair mappings
        self._pairs, self._pairs_to_linear = all_pairs(self.n_img, return_map=True)

        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed

        # Sync3N specific vars
        self.S_weighting = S_weighting
        self.J_weighting = J_weighting
        self._D_null = 1e-13

        # Auto configure GPU
        self._use_gpu = False
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() >= 1:
                gpu_id = cp.cuda.runtime.getDevice()
                logger.info(
                    f"cupy and GPU {gpu_id} found by cuda runtime; enabling cupy."
                )
                self._use_gpu = True
            else:
                logger.info("GPU not found, defaulting to numpy.")
        except ModuleNotFoundError:
            logger.info("cupy not found, defaulting numpy.")

    ###########################################
    # High level algorithm steps              #
    ###########################################
    def estimate_rotations(self):
        """
        Estimate rotation matrices.

        :return: Array of rotation matrices, size n_imgx3x3.
        """

        # Initial estimate of viewing directions
        Rijs0 = self._estimate_relative_viewing_directions()

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

    ###########################################
    # The hackberries taste like hackberries  #
    ###########################################
    def _sync3n_S_to_rot(self, S, W=None, n_eigs=4):
        """
        Use eigen decomposition of S to estimate transforms,
        then project transforms to nearest rotations.
        """

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
            D = np.mean(W, axis=1)  # D, check axis

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
            W = np.kron(W, np.ones((3, 3)))

            # Make Dhalf of size 3Nx3N
            Dhalf = np.diag(np.kron(np.diag(Dhalf), np.ones((1, 3)))[0])

            # Apply weights to S
            S = Dhalf @ (W * S) @ Dhalf

        # Extract three eigenvectors corresponding to non-zero eigenvalues.
        d, v = stable_eigsh(S, n_eigs)
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
        rotations = nearest_rotations(rotations)

        return rotations

    def _construct_sync3n_matrix(self, Rij):
        """
        Construct sync3n matrix from estimated rotations Rij.
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
        compute and return probability weights for S.
        """
        logger.info("Computing synchronization matrix weights.")

        def body(prev_too_low, Pmin, Pmax, hist, p_domain_limit=p_domain_limit):
            # Get inistial estimate for Pij
            P, sigma, Pij, hist, cum_scores = self._triangle_scores(
                Rijs, hist, Pmin, Pmax
            )

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
            inconsistent, Pij, res = body(*res)
            i += 1

        # Pack W
        W = np.zeros((self.n_img, self.n_img))
        idx = 0
        for i in range(self.n_img):
            for j in range(i + 1, self.n_img):
                W[i, j] = Pij[idx]
                W[j, i] = Pij[idx]
                idx += 1

        return W

    def _triangle_scores_mex(self, Rijs, hist_intervals):
        # The following is adopted from Matlab triangle_scores_mex.c

        # Initialize probability result arrays
        cum_scores = np.zeros(len(Rijs), dtype=self.dtype)
        scores_hist = np.zeros(hist_intervals, dtype=self.dtype)
        h = 1 / hist_intervals

        c = np.empty((4), dtype=self.dtype)
        for i in trange(self.n_img, desc="Computing triangle scores"):
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
                    c[0] = np.sum(((Rij @ Rjk.T) - Rik) ** 2)
                    c[1] = np.sum(((Rij_J @ Rjk.T) - Rik) ** 2)
                    c[2] = np.sum(((Rij @ Rjk_J.T) - Rik) ** 2)
                    c[3] = np.sum(((Rij @ Rjk.T) - Rik_J) ** 2)

                    # Find best match
                    best_i = np.argmin(c)
                    best_val = c[best_i]

                    # For each triangle side, find the best alternative
                    alt_ij_jk = c[_ALTS[0][best_i][0]]
                    if c[_ALTS[1][best_i][0]] < alt_ij_jk:
                        alt_ij_jk = c[_ALTS[1][best_i][0]]

                    alt_ik_jk = c[_ALTS[0][best_i][1]]
                    if c[_ALTS[1][best_i][1]] < alt_ik_jk:
                        alt_ik_jk = c[_ALTS[1][best_i][1]]

                    alt_ij_ik = c[_ALTS[0][best_i][2]]
                    if c[_ALTS[1][best_i][2]] < alt_ij_ik:
                        alt_ij_ik = c[_ALTS[1][best_i][2]]

                    # Compute scores
                    s_ij_jk = 1 - np.sqrt(best_val / alt_ij_jk)
                    s_ik_jk = 1 - np.sqrt(best_val / alt_ik_jk)
                    s_ij_ik = 1 - np.sqrt(best_val / alt_ij_ik)

                    # Update cumulated scores
                    cum_scores[ij] += s_ij_jk + s_ij_ik
                    cum_scores[jk] += s_ij_jk + s_ik_jk
                    cum_scores[ik] += s_ik_jk + s_ij_ik

                    # Update histogram
                    threshold = 0
                    for _l1 in range(hist_intervals):
                        threshold += h
                        if s_ij_jk < threshold:
                            break

                    for _l2 in range(hist_intervals):
                        threshold += h
                        if s_ik_jk < threshold:
                            break

                    for _l3 in range(hist_intervals):
                        threshold += h
                        if s_ij_ik < threshold:
                            break

                    scores_hist[_l1] += 1
                    scores_hist[_l2] += 1
                    scores_hist[_l3] += 1

        return cum_scores, scores_hist

    def _pairs_probabilities(self, Rijs, P2, A, a, B, b, x0):
        # The following is adopted from Matlab pairas_probabilities_mex.c `looper`

        # Initialize probability result arrays
        ln_f_ind = np.zeros(len(Rijs), dtype=self.dtype)
        ln_f_arb = np.zeros(len(Rijs), dtype=self.dtype)

        c = np.empty((4), dtype=self.dtype)
        for i in trange(self.n_img, desc="Computing pair probabilities"):
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
                    c[0] = np.sum(((Rij @ Rjk.T) - Rik) ** 2)
                    c[1] = np.sum(((Rij_J @ Rjk.T) - Rik) ** 2)
                    c[2] = np.sum(((Rij @ Rjk_J.T) - Rik) ** 2)
                    c[3] = np.sum(((Rij @ Rjk.T) - Rik_J) ** 2)

                    # Find best match
                    best_i = np.argmin(c)
                    best_val = c[best_i]

                    # For each triangle side, find the best alternative
                    alt_ij_jk = c[_ALTS[0][best_i][0]]
                    if c[_ALTS[1][best_i][0]] < alt_ij_jk:
                        alt_ij_jk = c[_ALTS[1][best_i][0]]
                    alt_ik_jk = c[_ALTS[0][best_i][1]]
                    if c[_ALTS[1][best_i][1]] < alt_ik_jk:
                        alt_ik_jk = c[_ALTS[1][best_i][1]]
                    alt_ij_ik = c[_ALTS[0][best_i][2]]
                    if c[_ALTS[1][best_i][2]] < alt_ij_ik:
                        alt_ij_ik = c[_ALTS[1][best_i][2]]

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

    def _triangle_scores(
        self,
        Rijs,
        scores_hist,
        Pmin,
        Pmax,
        hist_intervals=100,
        a=2.2,
        peak2sigma=2.43e-2,
        P=0.5,
        b=2.5,
        x0=0.78,
    ):
        """
        Todo

        :param a: magic number
        :param peak2sigma: empirical relation between the location of
            the peak of the histigram, and the mean error in the
            common lines estimations.
            AKA, magic number
        :param P:
        :param b:
        :param x0:
        """

        Pmin = Pmin or 0
        Pmin = max(Pmin, 0)  # Clamp probability to [0,1]
        Pmax = Pmax or 1
        Pmax = min(Pmax, 1)  # Clamp probability to [0,1]

        cum_scores = None  # XXX Why do we even need cum_scores?
        if scores_hist is None:
            cum_scores, scores_hist = self._triangle_scores_mex(Rijs, hist_intervals)

            # Normalize cumulated scores
            cum_scores /= len(Rijs)

        # Histogram decomposition: P & sigma evaluation
        h = 1 / hist_intervals
        hist_x = np.arange(h / 2, 1, h)
        # normalization factor of one component of the histogram
        A = (
            (self.n_img * (self.n_img - 1) * (self.n_img - 2) / 2)
            / hist_intervals
            * (a + 1)
        )
        # normalization of 2nd component: B = P*N_delta/sum(f), where f is the component formula
        B0 = P ** (self.n_img * (self.n_img - 1) * (self.n_img - 2) / 2) / np.sum(
            ((1 - hist_x) ** b) * np.exp(-b / (1 - x0) * (1 - hist_x))
        )
        start_values = np.array([B0, P, b, x0], dtype=np.float64)
        lower_bounds = np.array([0, Pmin**3, 2, 0], dtype=np.float64)
        upper_bounds = np.array([np.inf, Pmax**3, np.inf, 1], dtype=np.float64)

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
        )
        B, P, b, x0 = popt

        # Derive P and sigma
        P = P ** (1 / 3)
        peak = x0  # can rm later
        sigma = (1 - peak) / peak2sigma

        # Initialize probability computations
        # Local histograms analysis
        A = a + 1  # distribution 1st component normalization factor
        # distribution 2nd component normalization factor
        B = B / (
            (self.n_img * (self.n_img - 1) * (self.n_img - 2) / 2) / hist_intervals
        )

        # Calculate probabilities
        ln_f_ind, ln_f_arb = self._pairs_probabilities(Rijs, P**2, A, a, B, b, x0)
        Pij = 1 / (1 + (1 - P) / P * np.exp(ln_f_arb - ln_f_ind))

        # Fix singular output
        num_nan = np.sum(np.isnan(Pij))
        if num_nan > 0:
            logger.error(
                f"NaN probabilities occurred {num_nan} times out of {np.size(Pij)}. Setting NaNs to zero."
            )
            Pij = np.nan_to_num(Pij)

        return P, sigma, Pij, scores_hist, cum_scores

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _estimate_relative_viewing_directions(self):
        """
        Estimate the relative viewing directions vij = vi*vj^T, i<j, and vii = vi*vi^T, where
        vi is the third row of the i'th rotation matrix Ri.
        """
        logger.info(f"Estimating relative viewing directions for {self.n_img} images.")
        # Detect a single pair of common-lines between each pair of images
        self.build_clmatrix()

        # Calculate relative rotations
        Rijs = self._estimate_all_Rijs(self.clmatrix)

        return Rijs

    def _global_J_sync(self, Rijs):
        """ """

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
        """
        n_img = self.n_img
        n_theta = self.n_theta
        Rijs = np.zeros((len(self._pairs), 3, 3))

        for idx, (i, j) in enumerate(self._pairs):
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
        good_k = self._vote_ij(clmatrix, n_theta, i, j, k_list)

        rots = self._rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta)

        if rots is not None:
            rot_mean = np.mean(rots, 0)

        else:
            # This is for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # We set to zero as in the Matlab code.
            rot_mean = np.zeros((3, 3))

        return rot_mean

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
        vec = randn(n_Rijs, seed=self.seed)
        vec = vec / norm(vec)
        residual = 1
        itr = 0

        # XXX, I don't like that epsilon>1 (residual) returns signs of random vector
        #      maybe force to run once? or return vec as zeros in that case?
        #      Seems unintended, but easy to do.

        # Power method iterations
        while itr < max_iters and residual > epsilon:
            itr += 1
            vec_new = self._signs_times_v(Rijs, vec)
            vec_new = vec_new / norm(vec_new)
            residual = norm(vec_new - vec)
            vec = vec_new
            logger.info(
                f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
            )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec)

        return J_sync

    def _signs_times_v(self, Rijs, vec):

        # host/gpu dispatch
        if self._use_gpu:
            assert self.J_weighting is False, "not implemented yet"
            new_vec = _signs_times_v_cupy(
                self.n_img, Rijs, vec, self.J_weighting, _ALTS
            )
        else:
            new_vec = _signs_times_v_host(
                self.n_img, Rijs, vec, self.J_weighting, _ALTS, self._pairs_to_linear
            )

        return new_vec


def _signs_times_v_host(n, Rijs, vec, J_weighting, _ALTS, _pairs_to_linear):
    """
    Ported from _signs_times_v_mex.c

    n: n_img
    Rijs: nchoose2x3x3 array
    vec: input array
    new_vec: output array
    J_weighting: bool
    _ALTS= 2x4x3 const lut array
    _signs_confs = 4x3 const lut array
    """

    new_vec = np.zeros_like(vec)

    _signs_confs = np.array(
        [[1, 1, 1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]], dtype=int
    )
    c = np.empty((4))
    desc = "Computing signs_times_v"
    if J_weighting:
        desc += " with J_weighting"
    for i in trange(n, desc=desc):
        for j in range(i + 1, n - 1):  # check bound (taken from MATLAB mex)
            ij = _pairs_to_linear[i, j]
            Rij = Rijs[ij]
            for k in range(j + 1, n):
                ik = _pairs_to_linear[i, k]
                jk = _pairs_to_linear[j, k]
                Rik = Rijs[ik]
                Rjk = Rijs[jk]

                # Compute conjugated rotats
                Rij_J = J_conjugate(Rij)
                Rik_J = J_conjugate(Rik)
                Rjk_J = J_conjugate(Rjk)

                # Compute R muls and norms
                c[0] = np.sum(((Rjk @ Rij) - Rik) ** 2)
                c[1] = np.sum(((Rjk @ Rij_J) - Rik) ** 2)
                c[2] = np.sum(((Rjk_J @ Rij) - Rik) ** 2)
                c[3] = np.sum(((Rjk @ Rij) - Rik_J) ** 2)

                # Find best match
                best_i = np.argmin(c)
                best_val = c[best_i]

                # MATLAB: scores_as_entries == 0
                s_ij_jk = _signs_confs[best_i][0]
                s_ik_jk = _signs_confs[best_i][1]
                s_ij_ik = _signs_confs[best_i][2]

                # Note there was a third J_weighting option (2) in MATLAB,
                # but it was not exposed at top level.
                if J_weighting:
                    # MATLAB: scores_as_entries == 1
                    # For each triangle side, find the best alternative
                    alt_ij_jk = c[_ALTS[0][best_i][0]]
                    if c[_ALTS[1][best_i][0]] < alt_ij_jk:
                        alt_ij_jk = c[_ALTS[1][best_i][0]]

                    alt_ik_jk = c[_ALTS[0][best_i][1]]
                    if c[_ALTS[1][best_i][1]] < alt_ik_jk:
                        alt_ik_jk = c[_ALTS[1][best_i][1]]

                    alt_ij_ik = c[_ALTS[0][best_i][2]]
                    if c[_ALTS[1][best_i][2]] < alt_ij_ik:
                        alt_ij_ik = c[_ALTS[1][best_i][2]]

                    # Compute scores
                    s_ij_jk *= 1 - np.sqrt(best_val / alt_ij_jk)
                    s_ik_jk *= 1 - np.sqrt(best_val / alt_ik_jk)
                    s_ij_ik *= 1 - np.sqrt(best_val / alt_ij_ik)

                # Update vector entries
                new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
                new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
                new_vec[ik] += s_ij_ik * vec[ij] + s_ik_jk * vec[jk]

    return new_vec


def _signs_times_v_cupy(n, Rijs, vec, J_weighting, _ALTS):
    """
    Ported from _signs_times_v_mex.c

    n: n_img
    Rijs: nchoose2x3x3 array
    vec: input array
    new_vec: output array
    #todo J_weighting: bool
    #todo _ALTS= 2x4x3 const lut array
    #todo _signs_confs = 4x3 const lut array
    """
    import cupy as cp

    code = r"""

/* from i,j indoces to the common index in the N-choose-2 sized array */
#define PAIR_IDX(N,I,J) ((2*N-I-1)*I/2 + J-I-1)


inline void mult_3x3(double *out, double *R1, double *R2) {
  /* 3X3 matrices multiplication: out = R1*R2 */
  int i,j;
  for (i=0; i<3; i++) {
    for (j=0;j<3;j++) {
      out[3*j+i] = R1[3*0+i]*R2[3*j+0] + R1[3*1+i]*R2[3*j+1] + R1[3*2+i]*R2[3*j+2];
    }
  }
}

inline void JRJ(double *R, double *A) {
/* multiple 3X3 matrix by J from both sizes: A = JRJ */
        A[0]=R[0];
        A[1]=R[1];
        A[2]=-R[2];
        A[3]=R[3];
        A[4]=R[4];
        A[5]=-R[5];
        A[6]=-R[6];
        A[7]=-R[7];
        A[8]=R[8];
}

inline double diff_norm_3x3(const double *R1, const double *R2) {
/* difference 2 matrices and return squared norm: ||R1-R2||^2 */
        int i;
        double norm = 0;
        for (i=0; i<9; i++) {norm += (R1[i]-R2[i])*(R1[i]-R2[i]);}
        return norm;
}


extern "C" __global__
void signs_times_v(int n, double* Rijs, const double* vec, double* new_vec)
{
    /* thread index (1d), represents "i" index */
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* no-op when out of bounds */
    if(i >= n) return;

    double c[4];
    int j;
    int k;
    for(k=0;k<4;k++){c[k]=0;}
    unsigned long ij, jk, ik;
    int best_i;
    double best_val;
    int s_ij_jk, s_ik_jk, s_ij_ik;

    double *Rij, *Rjk, *Rik;
    double JRijJ[9], JRjkJ[9], JRikJ[9];
    double tmp[9];

    /* le sigh */
    int signs_confs[4][3];
    for(int a=0; a<4; a++) { for(k=0; k<3; k++) { signs_confs[a][k]=1; } }
    signs_confs[2-1][1-1]=-1; signs_confs[2-1][3-1]=-1;
    signs_confs[3-1][1-1]=-1; signs_confs[3-1][2-1]=-1;
    signs_confs[4-1][2-1]=-1; signs_confs[4-1][3-1]=-1;


    for(j=i+1; j< (n - 1); j++){
        ij = PAIR_IDX(n, i, j);
        for(k=j+1; k< n; k++){
            ik = PAIR_IDX(n, i, k);
            jk = PAIR_IDX(n, j, k);

            /* compute configurations matches scores */
            Rij = Rijs + 9*ij;
            Rjk = Rijs + 9*jk;
            Rik = Rijs + 9*ik;

            JRJ(Rij, JRijJ);
            JRJ(Rjk, JRjkJ);
            JRJ(Rik, JRikJ);

            mult_3x3(tmp,Rij,Rjk);
            c[0] = diff_norm_3x3(tmp,Rik);

            mult_3x3(tmp,JRijJ,Rjk);
            c[1] = diff_norm_3x3(tmp,Rik);

            mult_3x3(tmp,Rij,JRjkJ);
            c[2] = diff_norm_3x3(tmp,Rik);

            mult_3x3(tmp,Rij,Rjk);
            c[3] = diff_norm_3x3(tmp,JRikJ);

            /* find best match */
            best_i=0; best_val=c[0];
            if (c[1]<best_val) {best_i=1; best_val=c[1];}
            if (c[2]<best_val) {best_i=2; best_val=c[2];}
            if (c[3]<best_val) {best_i=3; best_val=c[3];}

            /* set triangles entries to be signs */
            s_ij_jk = signs_confs[best_i][0];
            s_ik_jk = signs_confs[best_i][1];
            s_ij_ik = signs_confs[best_i][2];

            /* update multiplication */
            new_vec[ij*n + i] += s_ij_jk*vec[jk] + s_ij_ik*vec[ik];
            new_vec[jk*n + i] += s_ij_jk*vec[ij] + s_ik_jk*vec[ik];
            new_vec[ik*n + i] += s_ij_ik*vec[ij] + s_ik_jk*vec[jk];

        } /* k */
    } /* j */

    return;
};
"""

    module = cp.RawModule(code=code)
    signs_times_v = module.get_function("signs_times_v")

    Rijs_dev = cp.array(Rijs)
    vec_dev = cp.array(vec)
    # 2d over i then accum to avoid race on i
    new_vec_dev = cp.zeros((vec.shape[0], n))

    # call the kernel
    blkszx = 512
    nblkx = (n + blkszx - 1) // blkszx

    signs_times_v((nblkx,), (blkszx,), (n, Rijs_dev, vec_dev, new_vec_dev))

    # accumulate, can reuse the vec_dev array now.
    cp.sum(new_vec_dev, axis=1, out=vec_dev)

    # dtoh
    new_vec = vec_dev.get()

    return new_vec
