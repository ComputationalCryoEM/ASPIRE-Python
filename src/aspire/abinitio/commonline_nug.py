import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import factorial

from aspire.abinitio import Orient3D
from aspire.abinitio.sync_voting import _syncrotations
from aspire.numeric import xp
from aspire.operators import PolarFT
from aspire.utils import Rotation, cart2sph, tqdm, trange
from aspire.volume import (
    CnSymmetryGroup,
    DnSymmetryGroup,
    IdentitySymmetryGroup,
    SymmetryGroup,
)

from .commonline_utils import _generate_shift_phase_and_filter, saff_kuijlaars

logger = logging.getLogger(__name__)


class CommonlineNUG(Orient3D):
    """
    Estimate orientations of cyclically or dihedrally symmetric molecules using the non-unique games framework.
    """

    def __init__(
        self,
        src,
        symmetry=None,
        n_rad=None,
        n_theta=360,
        max_shift=0.15,
        shift_step=1,
        mask=True,
        Lmax=12,
        T=36,
        max_iter=501,
        rho=0.05,
        ratio=1,
        factor=1.0,
        mult=1.5,
        S2_grid=441,
        Nstep_yI=10,
        c1_pair_batch_size=1024,
        pr_iters=None,
        verbose=True,
        **kwargs,
    ):
        """
        Initialize the NUG orientation estimator.

        The estimator supports asymmetric molecules (C1) and molecules with
        cyclic (Cn) or dihedral (Dn) symmetry. Most defaults match those used
        for the experiments in the associated publication. Proximal refinement
        is available only for nontrivial cyclic and dihedral symmetry groups.

        For complete experimental workflows, see the NUG examples in
        ``gallery/experiments``.

        :param src: Source containing the input projection images.
        :param symmetry: Cyclic or dihedral symmetry specification, such as 'C1', 'C3',
            or 'D4'. If None, use the symmetry specified by `src`. Default is None.
        :param n_rad: Number of radial samples in the polar Fourier transform. If None,
            n_rad will default to the ceiling of half the resolution of the source.
        :param n_theta: Number of angular samples in the polar Fourier transform. This
            value must be even. Default is 360.
        :param max_shift: Determines maximum range for shifts for common-line detection
            as a proportion of the resolution. Default is 0.15.
        :param shift_step: Resolution of shift estimation common-line detection in pixels.
            Default is 1 pixel.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        :param Lmax: Maximum Wigner representation degree used in the relaxation.
            Default is 12.
        :param T: Quadrature resolution used to compute the Fourier coefficients.
            Default is 36.
        :param max_iter: Number of ADMM iterations. Default is 501.
        :param rho: Initial ADMM penalty parameter. Default 0.05.
        :param ratio: Residual ratio used when updating the ADMM penalty. Default is 1.
        :param factor: Scaling factor used when updating the ADMM penalty. Default is 1.0.
        :param mult: Step-size multiplier for the ADMM primal update. Default is 1.5.
        :param S2_grid: Number of sphere samples used to discretize SO(3). Default is 441.
        :param Nstep_yI: Number of inequality-multiplier updates per ADMM iteration.
            Default is 10.
        :param c1_pair_batch_size: Number of distinct image-pair columns processed per
            C1 Fejer operation. Default is 1024.
        :param pr_iters: Number of proximal-refinement iterations for nontrivial
            cyclic and dihedral symmetry groups. If None, do not perform proximal
            refinement. Supplying a value for an asymmetric C1 problem raises a
            `ValueError`. Default is None.
        :param verbose: Whether to log ADMM and proximal refinement progress.
            Default is True.
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

        self.Lmax = Lmax
        self.T = T
        self.max_iter = max_iter
        self.rho = rho
        self.ratio = ratio
        self.factor = factor
        self.mult = mult
        self.S2_grid = S2_grid
        self.Nstep_yI = Nstep_yI
        self.verbose = verbose

        if c1_pair_batch_size <= 0:
            raise ValueError("c1_pair_batch_size must be positive")
        self.c1_pair_batch_size = c1_pair_batch_size

        # Handle symmetry.
        # Get from source if not provided. Warn on mismatch if provided.
        if symmetry is None:
            logger.info(
                f"Symmetry not provided. Using Source symmetry: {str(self.src.symmetry_group)}"
            )
            self.sym_grp = self.src.symmetry_group
        else:
            if symmetry != str(self.src.symmetry_group):
                logger.info(
                    f"Provided symmetry, {symmetry}, does not match source, {str(self.src.symmetry_group)}"
                )
            logger.info(f"Using provided symmetry: {symmetry}")
            self.sym_grp = SymmetryGroup.parse(symmetry)

        if not isinstance(
            self.sym_grp, (IdentitySymmetryGroup, CnSymmetryGroup, DnSymmetryGroup)
        ):
            raise ValueError(
                f"This algorithm supports cyclic, dihedral, and asymmetric molecules. Found {str(self.sym_grp)}."
            )

        # Get Euler angles for the symmetry group.
        # Supress expected gimbal-lock warning (due to identity matrix).
        sym_rotations = Rotation(
            self.sym_grp.matrices,
            gimble_lock_warnings=False,
        )
        self.sym_euler = sym_rotations.angles
        self.n_sym = len(self.sym_euler)

        # Proximal refinement is implemented only for cyclic and dihedral symmetry groups.
        if self.n_sym == 1 and pr_iters is not None:
            raise ValueError(
                "Proximal refinement is not supported for asymmetric (C1) NUG. "
                "Set pr_iters=None."
            )

        # Set up proximal refinement terms.
        if pr_iters is not None:
            self.pr_weights = 1 / (1 + np.arange(self.Lmax))
            self.pr_penalty = [1] * pr_iters
            self.pr_rank = list(range(pr_iters - 1, -1, -1))  # [pr_iters - 1,..., 0]
        self.pr_iters = pr_iters

        self._build_full_pft()

    def _build_full_pft(self):
        """
        Construct the full polar Fourier transforms and candidate shift phases.
        """
        pf = self.pf
        self.pf_full = PolarFT.half_to_full(pf)

        # Prepare the shift phases for common-line detection
        r_max = self.pf_full.shape[2]
        self.shifts, self.shift_phases, _ = _generate_shift_phase_and_filter(
            r_max, self.max_shift, self.shift_step, self.dtype
        )

    def estimate_rotations(self):
        """
        Estimate rotations by computing NUG coefficients, solving the SDP relaxation, and recovering rotations.

        :return: Estimated rotation matrices.
        """
        self.compute_coeff()
        self.perform_admm()
        self.recover_rotations()

        # Return unused CuPy allocations before downstream reconstruction.
        if hasattr(xp, "get_default_memory_pool"):
            xp.get_default_memory_pool().free_all_blocks()

        return self.rotations

    #####################################
    # Objective Coefficient Computation #
    #####################################

    def compute_coeff(self):
        """
        Compute the truncated Fourier coefficient matrices of the pairwise common-line losses.
        """
        # Build degree-wise coefficient matrices C[k] for the linear SDP objective.
        N = self.n_img
        n_theta = self.n_theta
        Lmax = self.Lmax
        T = self.T

        # Quadrature grid used to sample the pairwise loss over SO(3) Euler angles.
        alpha_grid = np.arange(2 * T, dtype=np.float64) * np.pi / T
        beta_grid = (2 * np.arange(2 * T, dtype=np.float64) + 1) * np.pi / 4 / T
        gamma_grid = np.arange(2 * T, dtype=np.float64) * np.pi / T

        # Map the sampled angles to polar Fourier ray indices once for all image pairs.
        alpha_idx = (
            np.round((alpha_grid - np.pi / 2) * n_theta / 2 / np.pi) % n_theta
        ).astype(int)
        gamma_idx = (
            np.round((-gamma_grid - np.pi / 2) * n_theta / 2 / np.pi) % n_theta
        ).astype(int)

        # Beta quadrature weights for the truncated Wigner-D Fourier expansion.
        bT = np.zeros(2 * T, dtype=np.float64)
        for n in range(2 * T):
            ss = 0
            for m in range(T):
                ss = ss + np.sin(beta_grid[n] * (2 * m + 1)) / (2 * m + 1)
            bT[n] = 2 / T * np.sin(beta_grid[n]) * ss

        # Precompute degree-wise beta/Wigner weight matrices used in each coefficient transform.
        BTK = []
        for k in range(1, Lmax + 1):
            btk = np.sum(bT[:, None, None] * self._Wd(k, beta_grid), axis=0)
            BTK.append(btk.T)

        # Allocate one coefficient matrix per Wigner degree;
        # each is block-indexed by image pair.
        C = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            C.append(np.zeros((N * dk, N * dk), dtype=np.complex128))

        # Compute off-diagonal image-pair losses and insert their degree-wise coefficients.
        with tqdm(
            total=N * (N - 1) // 2,
            desc="Computing off-diagonal coefficients",
        ) as pbar:
            for i in range(N):
                for j in range(i + 1, N):
                    Fij = self._pairwise_loss(i, j, alpha_idx, gamma_idx)
                    for k_idx in range(Lmax):
                        k = k_idx + 1
                        dk = 2 * k + 1
                        C[k_idx][j * dk : (j + 1) * dk, i * dk : (i + 1) * dk] = (
                            self._pairwise_fourier_coeff(
                                k, Fij, alpha_grid, gamma_grid, BTK[k_idx]
                            )
                        )
                    pbar.update(1)

        # Fill the conjugate transpose blocks so each coefficient matrix is Hermitian.
        for k_idx in range(Lmax):
            C[k_idx] = C[k_idx] + C[k_idx].conj().T

        # Diagonal blocks encode self-pair losses and are handled separately.
        for i in trange(N, desc="Computing diagonal coefficients"):
            Fii = self._pairwise_loss(i, i, alpha_idx, gamma_idx)
            for k_idx in range(Lmax):
                k = k_idx + 1
                dk = 2 * k + 1
                C[k_idx][i * dk : (i + 1) * dk, i * dk : (i + 1) * dk] = (
                    self._pairwise_fourier_coeff(
                        k, Fii, alpha_grid, gamma_grid, BTK[k_idx]
                    )
                )

        # Convert complex Wigner coefficients to the real representation basis used by ADMM.
        for k_idx in range(Lmax):
            k = k_idx + 1
            T, Tinv = self._complex2real(k)
            C[k_idx] = np.real(
                np.kron(np.eye(N, dtype=np.float64), Tinv)
                @ C[k_idx]
                @ np.kron(np.eye(N, dtype=np.float64), T)
            )

        self.C = C

    def _pairwise_loss(self, i, j, alpha_idx, gamma_idx):
        """
        Evaluate the common-line loss for a pair of images.

        Select polar Fourier rays using the supplied angular indices. For each
        pair of rays, calculate the L1 mismatch over radial samples and retain
        the smallest mismatch over the candidate one-dimensional shifts.

        :param i: Index of the first image.
        :param j: Index of the second image. May equal i for a self-pair.
        :param alpha_idx: One-dimensional array selecting rays from image i.
        :param gamma_idx: One-dimensional array selecting rays from image j.

        :return: Loss array of shape (len(alpha_idx), len(gamma_idx)).
            Rows follow alpha_idx and columns follow gamma_idx.
        """
        rays_i = self.pf_full[i, alpha_idx]
        rays_j = self.pf_full[j, gamma_idx]
        loss_grid = np.empty((len(alpha_idx), len(gamma_idx)), dtype=np.float64)

        for gamma_col, ray_j in enumerate(rays_j):
            shifted_j = self.shift_phases * ray_j
            differences = rays_i[:, None, :] - shifted_j[None, :, :]
            loss_grid[:, gamma_col] = np.linalg.norm(differences, ord=1, axis=-1).min(
                axis=-1
            )

        return loss_grid

    @staticmethod
    def _pairwise_fourier_coeff(k, F, alpha_grid, gamma_grid, beta_weights):
        """
        Approximate a degree-k Fourier coefficient matrix of a pairwise loss.

        Sum the sampled loss against complex exponentials on the alpha and gamma
        grids, then apply the degree-specific beta quadrature weights.

        :param k: Representation degree.
        :param F: Sampled pairwise loss array of shape (2*T, 2*T), with rows
            following alpha_grid and columns following gamma_grid.
        :param alpha_grid: Sampled alpha angles in radians, of shape (2*T,).
        :param gamma_grid: Sampled gamma angles in radians, of shape (2*T,).
        :param beta_weights: Degree-specific beta/Wigner weight matrix,
            of shape (2*k+1, 2*k+1).

        :return: Complex coefficient matrix of shape (2*k+1, 2*k+1).
        """
        dk = 2 * k + 1

        # Quadrature resolution: each angular grid contains 2*T samples.
        T = len(alpha_grid) // 2

        exp_alpha_grid = np.zeros((2 * T, dk), dtype=np.complex128)
        for m in range(-k, k + 1):
            exp_alpha_grid[:, m + k] = np.exp(1j * m * alpha_grid)

        exp_gamma_grid = np.zeros((2 * T, dk), dtype=np.complex128)
        for m in range(-k, k + 1):
            exp_gamma_grid[:, m + k] = np.exp(1j * m * gamma_grid)

        S = (exp_alpha_grid.T @ F @ exp_gamma_grid).T
        return beta_weights * S / 4 / T**2

    #############
    # ADMM Step #
    #############

    def perform_admm(self):
        """
        Solve the NUG relaxation and optionally apply proximal refinement.

        The trivial symmetry group C1 uses the asymmetric formulation.
        Nontrivial cyclic and dihedral symmetry groups use the
        symmetry-constrained formulation.
        """
        is_asymmetric = self.n_sym == 1

        if is_asymmetric:
            X_est = self.admm_plain_J(self.C, self.verbose)
        else:
            X_est = self.admm_sym_J(self.C, self.verbose)

        if self.pr_iters is not None:
            X_est = self.proximal_refine(
                X_est,
                self.pr_weights,
                self.pr_penalty,
                self.pr_rank,
            )

        self.X_est = X_est

    def admm_plain_J(self, C, verbose):
        """
        Solve the asymmetric NUG semidefinite relaxation using ADMM.

        To limit peak GPU memory use, Fejer inequality operations process
        off-diagonal image pairs in chunks, and large temporary arrays are
        released before memory intensive projections.

        :param C: Fourier coefficient matrices of the NUG objective.
        :param verbose: Whether to log ADMM progress.

        :return: Relaxed representation matrices.
        """
        pre = self.ADMM_preprocessing(C)

        # Prepare the diagonal and quaternion equality constraints.
        quaternions = _QuaternionConstraints(pre.blocks)
        equality = _C1EqualityConstraints(
            blocks=pre.blocks,
            quaternions=quaternions,
        )

        solver = _C1ADMM(
            pre=pre,
            equality=equality,
            max_iter=self.max_iter,
            rho=self.rho,
            ratio=self.ratio,
            factor=self.factor,
            mult=self.mult,
            Nstep_yI=self.Nstep_yI,
            verbose=verbose,
        )

        # Release references to initial arrays before the solver replaces them.
        del pre

        return solver.solve()

    def admm_sym_J(self, C, verbose):
        """
        Solve the symmetry-constrained NUG semidefinite relaxation using ADMM.

        :param C: Fourier coefficient matrices of the NUG objective.
        :param verbose: Whether to log ADMM progress.

        :return: Relaxed representation matrices.
        """
        pre = self.ADMM_preprocessing(C)

        rank_Ak, _ = self._compute_rank()
        logger.info(f"Rank of Ak: {rank_Ak}")

        # Prepare the symmetry/diagonal and quaternion equality constraints.
        quaternions = _QuaternionConstraints(pre.blocks)
        equality = _SymmetricEqualityConstraints(
            blocks=pre.blocks,
            quaternions=quaternions,
            rank_Ak=rank_Ak,
        )

        solver = _SymmetricADMM(
            pre=pre,
            equality=equality,
            max_iter=self.max_iter,
            rho=self.rho,
            ratio=self.ratio,
            factor=self.factor,
            mult=self.mult,
            Nstep_yI=self.Nstep_yI,
            verbose=verbose,
        )

        # Release references to initial arrays before the solver replaces them.
        del pre

        return solver.solve()

    ############################
    # ADMM Problem Preparation #
    ############################

    def ADMM_preprocessing(self, C):
        """
        Construct the transformed coefficients, constraints, indices, and
        initial variables used by ADMM.

        :param C: Full objective coefficient matrices. Entries in this list
            are replaced by normalized arrays on the active backend.

        :return: An _ADMMData instance containing the constraints, layout,
            and ADMM variables.
        """
        # normalize C matrix
        Cnorm = 0
        Xnorm = 0
        for k_idx in range(self.Lmax):
            k = k_idx + 1  # degree
            dk = 2 * k + 1

            Cnorm += np.linalg.norm(C[k_idx]) ** 2
            Xnorm += dk * self.n_img**2

        Cnorm = np.sqrt(Cnorm)
        Xnorm = np.sqrt(Xnorm)

        for k_idx in range(self.Lmax):
            C[k_idx] = xp.asarray(Xnorm / Cnorm * C[k_idx])

        # Prepare the representation layout and transform the normalized coefficients.
        blocks = _RepresentationBlocks(self.n_img, self.Lmax)
        C0, C1 = blocks._transform_coeff(C)
        normC = xp.sqrt(xp.linalg.norm(C0) ** 2 + xp.linalg.norm(C1) ** 2)

        # Construct the Fejer inequality operators from representation weights.
        W0, W1 = self._compute_fejer_weights()
        fejer = _FejerConstraints(
            W0=W0,
            W1=W1,
            blocks=blocks,
            pair_batch_size=self.c1_pair_batch_size,
            is_asymmetric=self.n_sym == 1,
        )

        # Release representation weights after constructing the operators.
        del W0, W1

        # Initialize the primal variables in upper-triangular image-pair form.
        D0, D1 = blocks.D0, blocks.D1
        n_packed = blocks.n_packed
        X0 = xp.zeros((D0, n_packed), dtype=np.float64)
        X1 = xp.zeros((D1, n_packed), dtype=np.float64)
        Xq = xp.zeros((16, self.n_img * (self.n_img - 1) // 2), dtype=np.float64)

        # The initial slack matrices are identity blocks for self-pairs and zero
        # for distinct image pairs. Build them directly in packed form.
        d0, d1 = blocks.d0, blocks.d1
        idx_diag = blocks.idx_diag
        S0 = xp.zeros_like(X0)
        S1 = xp.zeros_like(X1)
        for k_idx in range(self.Lmax):
            k = k_idx + 1
            S0[d0[k_idx] : d0[k_idx + 1] : k + 1, idx_diag] = 1
            S1[d1[k_idx] : d1[k_idx + 1] : k + 2, idx_diag] = 1

        Sq = xp.zeros(Xq.shape, dtype=np.float64)

        return _ADMMData(
            C0=C0,
            C1=C1,
            normC=normC,
            fejer=fejer,
            blocks=blocks,
            X0=X0,
            X1=X1,
            Xq=Xq,
            S0=S0,
            S1=S1,
            Sq=Sq,
        )

    def _compute_fejer_weights(self):
        """
        Evaluate the real Wigner representation blocks used by the discretized Fejer inequality constraints.

        :return: Two lists of degree-wise representation weight arrays.
        """
        SO3_grid = self._discretize_SO3()
        Ngrid = SO3_grid.shape[0]

        TT = []
        TTI = []
        for k_idx in range(self.Lmax):
            k = k_idx + 1
            T, Tinv = self._complex2real(k)
            TT.append(T)
            TTI.append(Tinv)

        W0 = []
        W1 = []
        for k_idx in range(self.Lmax):
            k = k_idx + 1
            W0k = np.zeros((Ngrid, k, k), dtype=np.float64)
            W1k = np.zeros((Ngrid, k + 1, k + 1), dtype=np.float64)

            TkT = TT[k_idx].T
            TinvkT = TTI[k_idx].T

            w = np.real(TkT @ self._WD(k, SO3_grid).conj() @ TinvkT)
            W0k, W1k = self._fejer_weight_blocks(w, k)

            W0.append(W0k)
            W1.append(W1k)
        return W0, W1

    @staticmethod
    def _fejer_weight_blocks(Ak, k):
        """
        Separate degree-k representation matrices into Fejer weight blocks.

        Reorder both matrix axes to group the two representation components,
        then select their diagonal blocks. Cross-component blocks are omitted.

        :param Ak: Array of representation matrices with shape
            (Ngrid, 2*k+1, 2*k+1).
        :param k: Representation degree.

        :return: Two weight arrays with shapes (Ngrid, k, k) and
            (Ngrid, k+1, k+1).
        """
        dk = 2 * k + 1
        Pk = np.eye(dk, dtype=np.float64)
        for m in range(k):
            for ell in range(k - m):
                Pk[(m + 2 * ell, m + 2 * ell + 1), :] = Pk[
                    (m + 2 * ell + 1, m + 2 * ell), :
                ]

        AkP = Pk @ Ak @ Pk.T
        return AkP[..., :k, :k], AkP[..., k:, k:]

    def _discretize_SO3(self):
        """
        Construct an approximately uniform Euler-angle grid over SO(3).

        :return: Array of ZYZ Euler angles.
        """
        S2 = saff_kuijlaars(self.S2_grid)
        S2_size = S2.shape[0]

        # discretize S1
        S1_size = round(np.sqrt(np.pi * S2_size))
        alpha = np.linspace(0, 2 * np.pi, S1_size)

        # discretize S2
        gamma, beta, _ = cart2sph(S2[:, 0], S2[:, 1], S2[:, 2])
        beta = np.pi / 2 - beta
        gamma = gamma + np.pi

        # SO(3) in Euler ZYZ
        SO3 = np.zeros((S2_size * S1_size, 3), dtype=np.float64)
        count = 0
        for i in range(S1_size):
            for j in range(S2_size):
                SO3[count] = [alpha[i], beta[j], gamma[j]]
                count += 1

        return SO3

    def _compute_rank(self):
        """
        Compute the ranks and matrices of the symmetry-averaging projectors at each degree.

        :return: Ranks and symmetry-averaging matrices for each degree.
        """
        rk = np.zeros(self.Lmax, dtype=np.float64)
        A = []
        for k_idx in range(self.Lmax):
            k = k_idx + 1  # degree
            Ak = np.sum(self._WD(k, self.sym_euler), axis=0)
            Ak = np.round(Ak / self.n_sym, 6)
            A.append(Ak)
            rk[k_idx] = np.linalg.matrix_rank(Ak)
        return rk, A

    ############################
    # Proximal Refinement Step #
    ############################

    def proximal_refine(self, X_admm, weight, Penalty, r):
        """
        Refine the relaxed solution by iteratively encouraging lower-rank representation matrices.

        :param X_admm: Initial relaxed representation matrices.
        :param weight: Degree-dependent refinement weights.
        :param Penalty: Penalty value for each refinement step.
        :param r: Rank offset for each refinement step.

        :return: Refined representation matrices.
        """
        C = self.C

        # The symmetry projector ranks give the expected low-rank structure
        # for each degree of a valid orbit solution.
        rank_Ak = np.zeros(self.Lmax, dtype=np.float64)

        # Keep an unmodified copy of the original objective coefficients; each
        # refinement step adds a temporary proximal term to these.
        C_base = [None] * self.Lmax
        for k_idx in range(self.Lmax):
            k = k_idx + 1
            C_base[k_idx] = xp.asnumpy(C[k_idx]).copy()
            rank_Ak[k_idx] = np.linalg.matrix_rank(
                self._symmetry_projector(k), tol=1e-6, hermitian=True
            )

        # CC stores the modified objective coefficients for the next proximal ADMM solve.
        CC = [None] * self.Lmax
        current = [np.copy(Xk) for Xk in X_admm]

        # Proximal refinement step.
        # Repeatedly bias the SDP objective toward the low-rank projection and resolve.
        for step in range(self.pr_iters):
            X_proj = self._low_rank_projection(current, rank_Ak, r[step])

            for k_idx in range(self.Lmax):
                CC[k_idx] = (
                    C_base[k_idx]
                    - Penalty[step]
                    * weight[k_idx]
                    * (X_proj[k_idx] + X_proj[k_idx].T)
                    / 2
                )

            X_next = self.admm_sym_J(CC, verbose=False)

            if self.verbose:
                logger.info(
                    f"Proximal refine step {step + 1}/{self.pr_iters}: "
                    f"relative update {self._relative_change(X_next, current):.3e}"
                )

            # Use the refined SDP solution as the starting point for the next projection step
            current = [np.copy(Xk) for Xk in X_next]

        return current

    def _symmetry_projector(self, k):
        """
        Average degree-k Wigner matrices over the estimator's symmetry group.

        :param k: Representation degree.

        :return: Symmetry-projector matrix of shape (2*k+1, 2*k+1),
            rounded to ten decimal places.
        """
        order = self.sym_euler.shape[0]
        A = self._WD(k, self.sym_euler).sum(axis=0)
        return np.round(A / order, 10)

    @staticmethod
    def _relative_change(A, B, eps=1e-12):
        """
        Measure relative change between two lists of representation matrices.

        :param A: First list of degree-wise representation matrices.
        :param B: Reference list with matching lengths and matrix shapes.
        :param eps: Positive lower bound for the denominator.

        :return: Relative change across all supplied matrices.
        """
        num = 0.0
        den = 0.0
        for k in range(len(A)):
            num += np.linalg.norm(A[k] - B[k]) ** 2
            den += np.linalg.norm(B[k]) ** 2
        return np.sqrt(num) / max(np.sqrt(den), eps)

    def _low_rank_projection(self, X, rank_Ak, r_step):
        """
        Approximate diagonal image blocks by matrices of reduced rank.

        For each degree, retain at most twice the symmetry-projector rank
        plus r_step singular values, capped at the block dimension.
        Off-diagonal image blocks are unchanged.

        :param X: List of full representation matrices for degrees 1 through
            Lmax. The degree-k matrix has shape (N*(2*k+1), N*(2*k+1)),
            where N is the number of images.
        :param rank_Ak: Symmetry projector ranks, one per degree.
        :param r_step: Integer rank offset for the current refinement step.

        :return: List of copied representation matrices with their diagonal
            image blocks replaced by the low-rank approximations.
        """
        Xproj = []
        for k_idx in range(self.Lmax):
            k = k_idx + 1
            dk = 2 * k + 1
            rk = min(int(rank_Ak[k_idx] * 2) + r_step, dk)
            tmp = np.copy(X[k_idx])

            for i in range(self.n_img):
                u, s, v = np.linalg.svd(
                    tmp[i * dk : (i + 1) * dk, i * dk : (i + 1) * dk]
                )
                tmp[i * dk : (i + 1) * dk, i * dk : (i + 1) * dk] = (
                    u[:, :rk] @ np.diag(s[:rk]) @ v[:rk]
                )

            Xproj.append(tmp)

        return Xproj

    ##########################
    # Rotation Recovery Step #
    ##########################

    def recover_rotations(self):
        """
        Recover rotations from the NUG representation matrices, using synchronization-based
        rounding for asymmetric molecules and symmetry-specific Euler estimation for cyclic and
        dihedral molecules.
        """
        X_est = self.X_est
        if isinstance(self.sym_grp, IdentitySymmetryGroup):
            R_est = self.rounding_C1(X_est[0])
        elif isinstance(self.sym_grp, CnSymmetryGroup):
            R_est = self.euler_est_Cm(X_est[0], X_est[self.n_sym - 1])
        elif isinstance(self.sym_grp, DnSymmetryGroup):
            R_est = self.euler_est_Dm(X_est)

        # self.Euler_est = Euler_est
        self.rotations = R_est.astype(self.dtype)

    def rounding_C1(self, X1):
        """
        Recover Euler angles for asymmetric molecules.

        :param X1: Relaxed degree-one representation matrix.

        :return: Estimated rotation matrices and Euler angles.
        """
        S = self.syncmatrix_from_X1(X1)
        rots = _syncrotations(S)
        return rots

    def euler_est_Cm(self, X1, XS):
        """
        Recover Euler angles for cyclic symmetry.

        :param X1: Relaxed degree-one representation matrix.
        :param XS: Relaxed representation matrix at the symmetry order.

        :return: Estimated rotation matrices.
        """
        S = self.n_sym

        # Convert the relaxed NUG matrices from the real representation used by
        # the SDP solver to the complex Wigner-D basis used in the recovery formulas.
        X1 = self._real_to_complex_representation(X1, degree=1)
        XS = self._real_to_complex_representation(XS, degree=S)

        # Estimate alpha/beta Euler angles.
        alpha_est = self._find_alpha(X1)
        beta_est = self._find_beta(X1)

        # Estimate the remaining Euler angle from the degree-S solution.
        # The cyclic pair estimator extracts measurements of
        # exp(i * S * (gamma_i - gamma_j)), which are then synchronized.
        gamma_est = self._estimate_gamma(
            XS,
            alpha_est,
            beta_est,
            degree=S,
            pair_estimator=self._cyclic_gamma_pair,
        )

        R_est = self._assemble_rotation_estimates(
            alpha_est,
            beta_est,
            gamma_est,
        )

        return R_est

    def euler_est_Dm(self, X_est):
        """
        Recover Euler angles for dihedral symmetry.

        :param X_est: Relaxed representation matrices.

        :return: Estimated rotation matrices and Euler angles.
        """
        S = self.sym_grp.order

        # Convert the degree-two and degree-S solutions from the real
        # representation used by the SDP solver to the complex Wigner-D basis.
        X2 = self._real_to_complex_representation(X_est[1], degree=2)
        XS = self._real_to_complex_representation(X_est[S - 1], degree=S)

        # Estimate alpha/beta Euler angles.
        alpha_est, beta_est = self._find_alpha_beta(X2)

        # Estimate gamma from the degree-S solution. The dihedral pair
        # estimator separates the gamma-difference and gamma-sum terms,
        # retaining the difference phases needed for synchronization.
        gamma_est = self._estimate_gamma(
            XS,
            alpha_est,
            beta_est,
            degree=S,
            pair_estimator=self._dihedral_gamma_pair,
            wrap=True,
        )

        R_est = self._assemble_rotation_estimates(
            alpha_est,
            beta_est,
            gamma_est,
        )

        return R_est

    #############################
    # Rotation Recovery Helpers #
    #############################

    def syncmatrix_from_X1(self, X1):
        """
        Construct the synchronization matrix directly from the degree-one
        NUG solution.

        :param X1: Degree-one NUG solution of shape (3 * n_img, 3 * n_img).

        :return: Synchronization matrix of shape (2 * n_img, 2 * n_img).
        """
        # View X1 as an n_img-by-n_img array of 3-by-3 blocks.
        X1_blocks = X1.reshape(self.n_img, 3, self.n_img, 3)

        # The NUG block decomposition gives each degree-one block X1_ij the
        # structure
        #
        #     [a  0  b]
        #     [0  c  0]
        #     [d  0  e].
        #
        # The one-dimensional component occupies index 1, while the
        # two-dimensional component used to construct the synchronization
        # matrix occupies indices 0 and 2. Extract that 2-by-2 component from
        # every image-pair block.
        corner_blocks = X1_blocks[:, [0, 2], :, :]
        corner_blocks = corner_blocks[:, :, :, [0, 2]]

        # The extracted blocks use the degree-one real spherical-harmonic
        # coordinate ordering (y, -x), while the synchronization matrix uses
        # Cartesian ordering (x, y). If
        #
        #     P = [[0, 1],
        #          [-1, 0]],
        #
        # maps Cartesian coordinates to the real-basis coordinates, then each
        # block must be converted according to
        #
        #     S_ij = P.T @ corner_ij @ P.
        #
        # The assignments below apply this change of basis to every image-pair
        # block simultaneously.
        syncmatrix = np.empty_like(corner_blocks)
        syncmatrix[:, 0, :, 0] = corner_blocks[:, 1, :, 1]
        syncmatrix[:, 0, :, 1] = -corner_blocks[:, 1, :, 0]
        syncmatrix[:, 1, :, 0] = -corner_blocks[:, 0, :, 1]
        syncmatrix[:, 1, :, 1] = corner_blocks[:, 0, :, 0]

        # Collapse the image and Cartesian-coordinate axes to obtain the
        # conventional 2*n_img-by-2*n_img synchronization matrix.
        syncmatrix = syncmatrix.reshape(2 * self.n_img, 2 * self.n_img)

        return syncmatrix

    @staticmethod
    def _find_phase(A, B):
        """
        Find the unit-modulus scalar c that minimizes ||cA - B||_F.

        :param A: First complex-valued matrix.
        :param B: Second complex-valued matrix.

        :return: Unit-modulus complex phase.
        """
        Ar = np.real(A)
        Ai = np.imag(A)
        Br = np.real(B)
        Bi = np.imag(B)

        denominator = np.vdot(Ar, Ar) + np.vdot(Ai, Ai)

        c = (np.vdot(Ar, Br) + np.vdot(Ai, Bi)) / denominator + 1j * (
            np.vdot(Ar, Bi) - np.vdot(Ai, Br)
        ) / denominator

        return c / abs(c)

    def _find_beta(self, X1):
        """
        Estimate beta angles for cyclic symmetry from the degree-one solution.

        Entries of each image pair block provide approximate outer products of
        sin(beta) and cos(beta). Recover those factors from their leading
        eigenvectors, then combine them to estimate one beta angle per image.

        :param X1: Degree-one representation matrix in the complex Wigner basis,
            with shape (3 * n_img, 3 * n_img).

        :return: Estimated beta angles in radians, with shape (n_img,).
        """
        N = self.n_img
        B1 = np.zeros((N, N), dtype=np.float64)
        B2 = np.zeros((N, N), dtype=np.float64)

        # The cyclic symmetry degree-one block gives
        #
        #   2 * abs(X1_ij[0, 0]) = sin(beta_i) * sin(beta_j)
        #   real(X1_ij[1, 1])    = cos(beta_i) * cos(beta_j).
        #
        # Therefore, B1 and B2 are approximately rank-one outer-product
        # matrices containing the sine and cosine factors, respectively.
        for i in range(N):
            for j in range(N):
                Xij = X1[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)]
                B1[i, j] = abs(Xij[0, 0]) * 2
                B2[i, j] = np.real(Xij[1, 1])

        # Recover vectors approximating sin(beta_i) and cos(beta_i) from
        # the leading eigenpairs of their rank-one Gram matrices.
        e1, v1 = np.linalg.eigh(B1)
        idx = np.argmax(e1)
        b1 = -v1[:, idx] * np.sqrt(e1[idx])

        e2, v2 = np.linalg.eigh(B2)
        idx = np.argmax(e2)
        b2 = v2[:, idx] * np.sqrt(e2[idx])

        # Combine the synchronized sine and cosine factors to recover beta.
        # The eigenvector sign ambiguity corresponds to the global
        # beta <-> pi - beta ambiguity described in the paper.
        beta = np.arctan(b1 / b2) % np.pi
        return beta

    def _find_alpha(self, X1):
        """
        Estimate alpha angles for cyclic symmetry from the degree-one solution.

        Extract pairwise alpha difference and sum phases from the image pair
        blocks. Synchronize the difference phases and use the sum phases to
        resolve the remaining common phase.

        :param X1: Degree-one representation matrix in the complex Wigner basis,
            with shape (3 * n_img, 3 * n_img).

        :return: Estimated alpha angles in radians, with shape (n_img,).
        """
        N = self.n_img
        ZZbar = np.zeros((N, N), dtype=np.complex128)
        ZZ = np.zeros((N, N), dtype=np.complex128)

        # The same degree-one block also gives the pairwise phase measurements
        #
        #   X1_ij[0, 0] / abs(X1_ij[0, 0])
        #       = exp(i * (alpha_i - alpha_j))
        #
        #  -X1_ij[0, 2] / abs(X1_ij[0, 2])
        #       = exp(i * (alpha_i + alpha_j)).
        #
        # Store these as the difference-phase and sum-phase matrices.
        for i in range(N):
            for j in range(N):
                Xij = X1[
                    3 * i : 3 * (i + 1),
                    3 * j : 3 * (j + 1),
                ]

                z = Xij[0, 0]
                ZZbar[i, j] = z / abs(z)

                z = Xij[0, 2]
                ZZ[i, j] = -z / abs(z)

        # Angular synchronization of the difference phases recovers
        # exp(i * alpha_i) up to a common phase.
        evals, evecs = np.linalg.eigh(ZZbar)
        idx = np.argmax(abs(evals))
        Z = evecs[:, idx] * np.sqrt(abs(evals[idx]))

        # Use the sum phases to resolve the remaining common phase.
        c = self._find_phase(Z[:, None] @ Z[:, None].T, ZZ)
        Z = np.sqrt(c) * Z
        return np.angle(Z).astype(np.float64)

    def _find_alpha_beta(self, X2):
        """
        Estimate alpha and beta angles for dihedral symmetry.

        Use the degree-two image pair blocks to estimate beta from an approximate
        rank-one matrix. Then extract pairwise alpha difference and sum phases,
        synchronize them, and resolve the remaining common phase. The beta
        estimates retain the ambiguity inherent in the dihedral data.

        :param X2: Degree-two representation matrix in the complex Wigner basis,
            with shape (5 * n_img, 5 * n_img).

        :return: Tuple of estimated alpha and beta angle arrays in radians;
            each has shape (n_img,).
        """
        N = self.n_img
        B1 = np.zeros((N, N), dtype=np.float64)

        # Combining two entries of the dihedral degree-two block isolates
        #
        #   B1[i, j] = cos(beta_i)^2 * cos(beta_j)^2.
        #
        # Thus, B1 is approximately a rank-one outer-product matrix.
        for i in range(N):
            for j in range(N):
                Xij = X2[5 * i : 5 * (i + 1), 5 * j : 5 * (j + 1)]
                B1[i, j] = np.real((Xij[2, 2] - 2 * abs(Xij[0, 0]) + 0.5) / 3 * 2)

        # Recover the vector cos(beta_i)^2 from the leading eigenpair.
        # Its entries should be nonnegative, so orient the eigenvector
        # using the sign of its first entry.
        e1, v1 = np.linalg.eigh(B1)
        idx = np.argmax(e1)
        b1 = v1[:, idx] * np.sqrt(e1[idx]) * np.sign(v1[0, idx])

        # Taking the square root recovers abs(cos(beta_i)). Consequently,
        # each image retains an independent beta_i <-> pi - beta_i
        # ambiguity, as described in the paper.
        beta_est = np.arccos(np.clip(np.sqrt(b1), -1, 1)) % np.pi

        Aminus = np.zeros((N, N), dtype=np.complex128)
        Aplus = np.zeros((N, N), dtype=np.complex128)

        # Divide two degree-two block entries by their known sin(2*beta)
        # factors to recover the pairwise phases:
        #
        #   Aminus[i, j] = exp(i * (alpha_i - alpha_j))
        #   Aplus[i, j]  = exp(i * (alpha_i + alpha_j)).
        for i in range(N):
            for j in range(N):
                Xij = X2[5 * i : 5 * (i + 1), 5 * j : 5 * (j + 1)]

                # Avoid division by a numerically vanishing
                # sin(2*beta) factor.
                if abs(beta_est[i]) < 1e-6:
                    beta_est[i] = 1e-6
                if abs(beta_est[j]) < 1e-6:
                    beta_est[j] = 1e-6
                Aminus[i, j] = (
                    Xij[1, 1]
                    / np.sin(2 * beta_est[i])
                    / np.sin(2 * beta_est[j])
                    * 8
                    / 3
                )
                Aplus[i, j] = (
                    -Xij[1, 3]
                    / np.sin(2 * beta_est[i])
                    / np.sin(2 * beta_est[j])
                    * 8
                    / 3
                )

        # Angular synchronization of the difference phases recovers
        # exp(i * alpha_i) up to a common phase.
        evals, evecs = np.linalg.eigh(Aminus)
        idx = np.argmax(abs(evals))
        Z = evecs[:, idx] * np.sqrt(abs(evals[idx]))

        # Use the sum phases to resolve the remaining common phase.
        # The square root leaves a global pi ambiguity.
        c = self._find_phase(Z[:, None] @ Z[:, None].T, Aplus)
        Z = np.sqrt(c) * Z
        alpha_est = (np.angle(Z)) % (2 * np.pi)

        return alpha_est, beta_est

    @staticmethod
    def _assemble_rotation_estimates(alpha, beta, gamma):
        """
        Assemble internal NUG Euler estimates and convert rotation matrices.

        :param alpha: Estimated alpha angles.
        :param beta: Estimated beta angles.
        :param gamma: Estimated gamma angles.

        :return: rotation matrices.
        """
        euler_angles = np.column_stack((alpha, beta, gamma)).astype(
            np.float64,
            copy=False,
        )

        rotations = Rotation.from_euler(euler_angles).matrices.swapaxes(-1, -2)

        return rotations

    def _estimate_gamma(
        self,
        Xm,
        alpha,
        beta,
        degree,
        pair_estimator,
        wrap=False,
    ):
        """
        Estimate gamma angles by synchronizing pairwise phase measurements.

        Xm must already be expressed in the complex Wigner basis.

        :param Xm: Complex-basis representation matrix at `degree`.
        :param alpha: Estimated alpha angles.
        :param beta: Estimated beta angles.
        :param degree: Representation degree used for gamma recovery.
        :param pair_estimator: Callable that computes one pairwise phase
            measurement.
        :param wrap: Whether to wrap the results to [0, 2*pi).

        :return: Estimated gamma angles.
        """
        N = self.n_img
        dk = 2 * degree + 1
        modes = np.arange(-degree, degree + 1)

        # Store pairwise estimates of exp(i * degree * (gamma_i - gamma_j)).
        C = np.zeros((N, N), dtype=np.complex128)

        # Construct the handedness-conjugation matrix.
        Jk = self._handedness_matrix(degree)

        # Evaluate the Wigner small-d matrices at the recovered beta angles.
        ws = self._Wd(degree, beta)

        for i in range(N):
            wi = ws[i]
            Di = np.exp(-1j * modes * alpha[i])

            for j in range(i + 1, N):
                wj = ws[j]
                Dj = np.exp(-1j * modes * alpha[j])

                # Each degree-m image-pair block factors as
                #
                #   X_ij = D(alpha_i) @ M_ij @ D(alpha_j)^*,
                #
                # where M_ij contains the beta- and gamma-dependent terms.
                Xij = Xm[
                    dk * i : dk * (i + 1),
                    dk * j : dk * (j + 1),
                ]

                # Cancel the known alpha-dependent factors to isolate M_ij:
                #
                #   D(alpha_i)^* @ X_ij @ D(alpha_j).
                DXijD = np.diag(Di.conj()) @ Xij @ np.diag(Dj)

                # Since beta is already estimated, the remaining unknown coefficients
                # are pairwise phases involving gamma_i and gamma_j.
                C[i, j] = pair_estimator(
                    wi,
                    wj,
                    DXijD,
                    Jk,
                    degree,
                )

        # Complete the Hermitian synchronization matrix.
        C += C.T.conj() + np.eye(N, dtype=np.float64)

        # Angular synchronization recovers exp(i * degree * gamma_i) up to one common phase.
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        leading_idx = np.argmax(eigenvalues)
        phase_vector = eigenvectors[:, leading_idx] * np.sqrt(eigenvalues[leading_idx])

        # Divide by the representation degree to recover gamma. Each estimate
        # remains defined modulo 2*pi/degree, as expected from the symmetry.
        gamma = np.angle(phase_vector) / degree

        if wrap:
            gamma %= 2 * np.pi

        return gamma

    @staticmethod
    def _cyclic_gamma_pair(wi, wj, DXijD, Jk, degree):
        """
        Estimate the pairwise gamma phase for cyclic symmetry.

        Construct the handedness-invariant Wigner terms induced by images
        i and j, subtract the contribution of the central Wigner component
        from the corresponding representation block, and fit the remaining
        real and imaginary components. The resulting complex value is inserted
        into the phase-synchronization matrix used to recover the gamma angles.

        :param wi: Degree-k Wigner small-d matrix evaluated at the
            estimated beta angle of image i. Shape (2 * k + 1, 2 * k + 1).
        :param wj: Degree-k Wigner small-d matrix evaluated at the
            estimated beta angle of image j. Shape (2 * k + 1, 2 * k + 1).
        :param DXijD: Complex representation block for images i and j after
            removing their estimated alpha phases. Shape (2 * k + 1, 2 * k + 1).
        :param Jk: Degree-k handedness-conjugation matrix. Shape
            (2 * k + 1, 2 * k + 1).
        :param degree: Wigner representation degree (k).

        :return: Complex pairwise phase measurement used for gamma
            synchronization.
        """
        outer_first = np.outer(wi[:, 0], wj[:, 0])
        C1 = (outer_first + Jk @ outer_first @ Jk) / 2

        outer_last = np.outer(wi[:, -1], wj[:, -1])
        C2 = (outer_last + Jk @ outer_last @ Jk) / 2

        outer_middle = np.outer(wi[:, degree], wj[:, degree])
        C3 = DXijD - (outer_middle + Jk @ outer_middle @ Jk) / 2

        return np.vdot(C1 + C2, np.real(C3)) / np.vdot(C1 + C2, C1 + C2) + 1j * np.vdot(
            C1 - C2, np.imag(C3)
        ) / np.vdot(C1 - C2, C1 - C2)

    @staticmethod
    def _solve_dihedral_gamma_pair(W1, W2, W3, W4, Br, Bi):
        """
        Fit one dihedral pairwise gamma-phase measurement.

        Solve separate two-variable least-squares systems for the real and
        imaginary components of the pairwise phase. The systems express the
        real and imaginary parts of the alpha-corrected representation block
        in terms of handedness-invariant combinations of the four extreme
        Wigner outer products.

        :param W1: Handedness-invariant outer product formed from the first
            Wigner columns of both images.
        :param W2: Handedness-invariant outer product formed from the last
            Wigner column of the first image and the first Wigner column of
            the second image.
        :param W3: Handedness-invariant outer product formed from the first
            Wigner column of the first image and the last Wigner column of
            the second image.
        :param W4: Handedness-invariant outer product formed from the last
            Wigner columns of both images.
        :param Br: Scaled real part of the alpha-corrected representation
            block.
        :param Bi: Scaled imaginary part of the alpha-corrected representation
            block.

        :return: Complex pairwise phase measurement whose real and imaginary
          components are obtained from the two least-squares fits.
        """
        A = np.array(
            [
                [np.vdot(W1 + W4, W1 + W4), np.vdot(W1 + W4, W2 + W3)],
                [np.vdot(W2 + W3, W1 + W4), np.vdot(W2 + W3, W2 + W3)],
            ]
        )
        B = np.array([np.vdot(W1 + W4, Br), np.vdot(W2 + W3, Br)])
        a, _ = np.linalg.lstsq(A, B, rcond=None)[0]

        A = np.array(
            [
                [np.vdot(W1 - W4, W1 - W4), np.vdot(W1 - W4, W3 - W2)],
                [np.vdot(W1 - W4, W3 - W2), np.vdot(W3 - W2, W3 - W2)],
            ]
        )
        B = np.array([np.vdot(W1 - W4, Bi), np.vdot(W3 - W2, Bi)])
        b, _ = np.linalg.lstsq(A, B, rcond=None)[0]

        return a + 1j * b

    def _dihedral_gamma_pair(self, wi, wj, DXijD, Jk, degree):
        """
        Estimate the pairwise gamma phase for dihedral symmetry.

        Form the four handedness-invariant outer products associated with the
        outer Wigner columns for images i and j. Fit these terms to the real
        and imaginary parts of the alpha-corrected representation block to
        obtain the complex measurement used for gamma synchronization.

        :param wi: Degree-k Wigner small-d matrix evaluated at the estimated
            beta angle of image i. Shape (2 * k + 1, 2 * k + 1).
        :param wj: Degree-k Wigner small-d matrix evaluated at the estimated
            beta angle of image j. Shape (2 * k + 1, 2 * k + 1).
        :param DXijD: Complex representation block for images i and j after
            removing their estimated alpha phases. Shape (2 * k + 1, 2 * k + 1).
        :param Jk: Degree-k handedness-conjugation matrix. Shape (2 * k + 1, 2 * k + 1).
        :param degree: Wigner representation degree. This parameter is included
            to provide the common pair-estimator interface used by `_estimate_gamma`.

        :return: Complex pairwise phase measurement used for gamma synchronization.
        """
        outer_11 = np.outer(wi[:, 0], wj[:, 0])
        outer_21 = np.outer(wi[:, -1], wj[:, 0])
        outer_12 = np.outer(wi[:, 0], wj[:, -1])
        outer_22 = np.outer(wi[:, -1], wj[:, -1])

        W1 = outer_11 + Jk @ outer_11 @ Jk
        W2 = outer_21 + Jk @ outer_21 @ Jk
        W3 = outer_12 + Jk @ outer_12 @ Jk
        W4 = outer_22 + Jk @ outer_22 @ Jk

        return self._solve_dihedral_gamma_pair(
            W1,
            W2,
            W3,
            W4,
            np.real(4 * DXijD),
            np.imag(4 * DXijD),
        )

    def _real_to_complex_representation(self, X, degree):
        """
        Convert a degree-wise representation matrix from the real basis used
        by ADMM to the complex Wigner basis.

        :param X: Representation matrix of shape
            (n_img * (2 * degree + 1), n_img * (2 * degree + 1)).
        :param degree: Wigner representation degree.

        :return: Representation matrix in the complex Wigner basis.
        """
        T, Tinv = self._complex2real(degree)
        identity = np.eye(self.n_img, dtype=np.float64)

        return np.kron(identity, T) @ X @ np.kron(identity, Tinv)

    @staticmethod
    def _handedness_matrix(degree):
        """
        Construct the handedness-conjugation matrix at a Wigner degree.

        :param degree: Wigner representation degree.

        :return: Diagonal matrix of shape (2 * degree + 1, 2 * degree + 1).
        """
        signs = np.ones(2 * degree + 1)
        signs[degree + 1 :: 2] = -1
        signs[degree - 1 :: -2] = -1

        return np.diag(signs)

    ########################################
    # Wigner Matrices and Basis Conversion #
    ########################################

    def _WD(self, k, euler):
        """
        Evaluate degree-k Wigner D matrices at the supplied ZYZ Euler angles.

        :param k: Wigner representation degree.
        :param euler: ZYZ Euler angles with shape (n_rotations, 3).

        :return: Complex Wigner D matrices with shape (n_rotations, 2 * k + 1, 2 * k + 1).
        """
        # compute Wigner D matrix
        alpha = euler[:, 0]
        beta = euler[:, 1]
        gamma = euler[:, 2]
        d = self._Wd(k, beta)

        m = np.arange(-k, k + 1)
        left = np.exp(-1j * alpha[:, None] * m[None, :])
        right = np.exp(-1j * gamma[:, None] * m[None, :])
        D = left[:, :, None] * d * right[:, None, :]

        return D

    @staticmethod
    def _Wd(k, beta):
        """
        Evaluate degree-k Wigner small-d matrices at the supplied polar angles.

        Matrix rows and columns correspond to orders from -k through k. These
        real, beta-dependent matrices are used to construct the corresponding
        Wigner D matrices.

        :param k: Wigner representation degree.
        :param beta: One-dimensional array of polar angles in radians, with
          shape (n_rotations,).

        :return: Wigner small-d matrices with shape
            (n_rotations, 2 * k + 1, 2 * k + 1).
        """
        # compute Wigner small d matrix
        d = np.zeros((len(beta), 2 * k + 1, 2 * k + 1), dtype=beta.dtype)
        for m in range(-k, k + 1):
            for n in range(-k, k + 1):
                smin = max(0, m - n)
                smax = min(k + m, k - n)
                for s in range(smin, smax + 1):
                    mul = (
                        np.sqrt(factorial(k + m))
                        / factorial(k + m - s)
                        * np.sqrt(factorial(k + n))
                        / factorial(s)
                        * np.sqrt(factorial(k - m))
                        / factorial(n - m + s)
                        * np.sqrt(factorial(k - n))
                        / factorial(k - n - s)
                    )
                    d[:, n + k, m + k] += (
                        mul
                        * (-1) ** (n - m + s)
                        * (np.cos(beta / 2)) ** (2 * k + m - n - 2 * s)
                        * (np.sin(beta / 2)) ** (n - m + 2 * s)
                    )
        return d

    @staticmethod
    def _complex2real(ell):
        """
        Construct the transformation matrices between complex and real degree ell representations.

        :param ell: Wigner representation degree.

        :return: Forward and inverse change-of-basis matrices.
        """
        diml = 2 * ell + 1
        Tinv = np.zeros((diml, diml), dtype=np.complex128)
        for i in range(diml):
            if i < ell:
                Tinv[i, i] = 1j / np.sqrt(2)
                Tinv[i, diml - 1 - i] = -1j * (-1) ** (i - ell) / np.sqrt(2)
            if i == ell:
                Tinv[i, i] = 1
            if i > ell:
                Tinv[i, i] = (-1) ** (i - ell) / np.sqrt(2)
                Tinv[i, diml - 1 - i] = 1 / np.sqrt(2)

        T = np.zeros((diml, diml), dtype=np.complex128)
        for i in range(diml):
            if i < ell:
                T[i, i] = -1j / np.sqrt(2)
                T[i, diml - 1 - i] = 1 / np.sqrt(2)
            if i == ell:
                T[i, i] = 1
            if i > ell:
                T[i, i] = (-1) ** (i - ell) / np.sqrt(2)
                T[i, diml - 1 - i] = 1j * (-1) ** (i - ell) / np.sqrt(2)
        return T, Tinv

    ######################
    # Validation Helpers #
    ######################

    def form_ground_truth_X(self, euler_angles):
        """
        Construct handedness-averaged ground-truth NUG representation matrices.

        This helper is intended for validating the relaxed ADMM solution when
        ground-truth rotations are available.

        :param euler_angles: Ground-truth ZYZ Euler angles of shape (n_img, 3).
        :return: List containing one ground-truth representation matrix for
            each degree from 1 through Lmax.
        """
        X_gt = []

        # Internally, euler_angles correspond to R.T, so we adjust them
        # here so X_gt corresponds to the ground truth rotations.
        euler_angles = -euler_angles[:, ::-1]

        for k in range(1, self.Lmax + 1):
            dk = 2 * k + 1

            # Evaluate the degree-k Wigner representations.
            wigner = self._WD(k, euler_angles)

            # Average the degree-k representation over the molecular symmetry
            # group. For the asymmetric case, self.sym_euler contains only the
            # identity and Ak is therefore the identity matrix.
            Ak = np.mean(self._WD(k, self.sym_euler), axis=0)

            # Construct the globally handedness-conjugated solution.
            Jk = np.ones(dk)
            Jk[k + 1 :: 2] = -1
            Jk[k - 1 :: -2] = -1
            Jk = np.diag(Jk)

            wigner_J = Jk @ wigner @ Jk

            # Convert each representation to the real basis used by ADMM.
            _, Tinv = self._complex2real(k)
            wigner_real = Tinv @ wigner
            wigner_J_real = Tinv @ wigner_J

            # Stack the image representations and construct their Gram matrices.
            wigner_real = wigner_real.reshape(self.n_img * dk, dk)
            wigner_J_real = wigner_J_real.reshape(self.n_img * dk, dk)

            Xk = wigner_real @ Ak @ wigner_real.conj().T
            XJk = wigner_J_real @ Ak @ wigner_J_real.conj().T

            # Average the two globally indistinguishable handedness choices.
            X_gt.append(np.real((Xk + XJk) / 2))

        return X_gt


################
# ADMM Solvers #
################


class _C1ADMM:
    """
    Solve the asymmetric NUG relaxation using ADMM.

    Owns the objective, iteration variables, and solver settings.
    Fixed layout and constraint operations are supplied by blocks,
    equality, and fejer.

    The solver releases unused Fejer operator arrays during construction
    and result assembly. Its prepared inputs are intended for one solve.
    """

    def __init__(
        self,
        pre,
        equality,
        max_iter,
        rho,
        ratio,
        factor,
        mult,
        Nstep_yI,
        verbose,
    ):
        """
        Initialize a C1 ADMM solve from prepared problem data.

        :param pre: Prepared objective, representation layout, Fejer
            constraints, and initial representation and slack arrays.
        :param equality: Fixed C1 equality constraints.
        :param max_iter: Number of ADMM iterations.
        :param rho: Initial ADMM penalty parameter.
        :param ratio: Residual comparison ratio used to adjust rho.
        :param factor: Multiplicative factor used to adjust rho.
        :param mult: Step multiplier for representation-variable updates.
        :param Nstep_yI: Number of inequality-multiplier updates per iteration.
        :param verbose: Whether to log iteration diagnostics.
        """
        self.blocks = pre.blocks
        self.equality = equality
        self.fejer = pre.fejer
        self.fejer.AI_mat_diag = None  # C1 does not use the self-pair operator.

        self.C0 = pre.C0
        self.C1 = pre.C1
        self.normC = pre.normC

        self.X0 = pre.X0
        self.X1 = pre.X1
        self.Xq = pre.Xq
        self.S0 = pre.S0
        self.S1 = pre.S1
        self.Sq = pre.Sq

        self.max_iter = max_iter
        self.rho = rho
        self.ratio = ratio
        self.factor = factor
        self.mult = mult
        self.Nstep_yI = Nstep_yI
        self.verbose = verbose

        self.Ngrid = self.fejer.AI_mat_offdiag.shape[0]

        # Initialize the constraint multipliers.
        self.yI = xp.zeros((self.Ngrid, self.blocks.n_pairs), dtype=np.float64)
        self.yE0 = xp.zeros(equality.bE0.shape, dtype=np.float64)
        self.yE1 = xp.zeros(equality.bE1.shape, dtype=np.float64)
        self.yEq = xp.zeros(equality.bEq.shape, dtype=np.float64)

        self.iteration = 0
        self.res_X = None

    def solve(self):
        """
        Run ADMM and reconstruct the degree-wise representation matrices.

        Updates the current solver state and releases selected work arrays
        before result assembly. This instance is intended for one solve.

        :return: List of relaxed degree-wise representation matrices
            converted to NumPy arrays.
        """
        update_order = np.arange(3)
        for t in trange(self.max_iter, desc="C1 NUG ADMM"):
            self.iteration = t
            np.random.shuffle(update_order)

            for update_idx in update_order:
                if update_idx == 0:
                    self.update_S()
                elif update_idx == 1:
                    self.update_yE()
                else:
                    for _ in range(self.Nstep_yI):
                        self.update_yI()

            self.update_X()
            if t % 100 == 0:
                self.print_updates()
            self.update_rho()

        # Release ADMM work arrays before reconstructing full degree matrices.
        self.yI = None
        self.C0 = self.C1 = None
        self.S0 = self.S1 = self.Sq = None
        self.Xq = None
        self.fejer.AI_mat_offdiag = None

        # Reconstruct full degree-wise matrices for rotation recovery.
        X_admm = self.blocks._transform_coeff_back(self.X0, self.X1)
        return [xp.asnumpy(Xk) for Xk in X_admm]

    def update_S(self):
        """
        Update and project the asymmetric PSD slack variables.

        Replaces S0, S1, and Sq. Releases full-size temporary arrays before
        the representation PSD projections.
        """
        Z0, Z1, Zq = self.equality.fun_AET(self.yE0, self.yE1, self.yEq)
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)

        S0 = self.C0.copy()
        S0 -= Z0
        S0 -= AIT_yI0
        xp.divide(self.X0, self.rho, out=Z0)  # Z0 is no longer needed.
        S0 -= Z0

        S1 = self.C1.copy()
        S1 -= Z1
        S1 -= AIT_yI1
        xp.divide(self.X1, self.rho, out=Z1)  # Z1 is no longer needed.
        S1 -= Z1

        # S0 = C0 - Z0 - AIT_yI0 - X0 / rho
        # S1 = C1 - Z1 - AIT_yI1 - X1 / rho
        Sq = -Zq - self.Xq / self.rho

        # These full size adjoint and equality arrays are no longer needed during
        # the PSD eigendecompositions. Release them for the eigensolver workspace.
        del Z0, Z1, AIT_yI0, AIT_yI1

        S0, S1 = self.blocks._project_representation_blocks(S0, S1)

        # This is the batched equivalent of projecting each 4x4 quaternion block
        # and storing tmp.T.reshape(16) in each column.
        Sq = self.blocks._psd_projection(Sq.T.reshape(self.blocks.n_pairs, 4, 4))
        Sq = Sq.T.reshape(16, self.blocks.n_pairs)
        self.S0, self.S1, self.Sq = S0, S1, Sq

    def update_yE(self):
        """
        Update the asymmetric equality multipliers.

        Replaces yE0, yE1, and yEq using the current representation arrays,
        slacks, inequality multipliers, and penalty parameter.
        """
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)
        z0, z1, zq = self.equality.fun_AE(
            -self.X0 / self.rho + self.C0 - self.S0 - AIT_yI0,
            -self.X1 / self.rho + self.C1 - self.S1 - AIT_yI1,
            -self.Xq / self.rho - self.Sq,
        )
        self.yE0, self.yE1, self.yEq = self.equality.solve_normal(
            self.equality.bE0 / self.rho + z0,
            self.equality.bE1 / self.rho + z1,
            self.equality.bEq / self.rho + zq,
        )

    def update_yI(self):
        """
        Update the nonnegative Fejer multipliers in image-pair chunks.

        Modifies yI in place to limit temporary GPU allocations.
        """
        AET_yE0, AET_yE1, _ = self.equality.fun_AET(self.yE0, self.yE1, self.yEq)

        for columns in self.fejer.pair_chunks():
            packed_columns = self.blocks.idx_offdiag[columns]
            AIT_yI = self.fejer.fun_AIT_chunk(self.yI[:, columns])

            tmp0 = (
                -self.X0[:, packed_columns] / self.rho
                + self.C0[:, packed_columns]
                - self.S0[:, packed_columns]
                - AIT_yI[: self.blocks.D0]
                - AET_yE0[:, packed_columns]
            )
            tmp1 = (
                -self.X1[:, packed_columns] / self.rho
                + self.C1[:, packed_columns]
                - self.S1[:, packed_columns]
                - AIT_yI[self.blocks.D0 :]
                - AET_yE1[:, packed_columns]
            )

            # This slice shares storage with self.yI; updates below modify it directly.
            yI_chunk = self.yI[:, columns]
            yI_chunk += self.fejer.bI / self.rho / self.fejer.Lambda
            yI_chunk += self.fejer.fun_AI(tmp0, tmp1) / self.fejer.Lambda
            xp.maximum(yI_chunk, 0, out=yI_chunk)

    def update_X(self):
        """
        Update the representation and quaternion arrays.

        Replaces X0, X1, and Xq, and stores their combined update-equation
        residual in res_X.
        """
        Z0, Z1, Zq = self.equality.fun_AET(self.yE0, self.yE1, self.yEq)
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)

        tmp = self.S0 + AIT_yI0 + Z0 - self.C0
        self.X0 = self.X0 + self.mult * self.rho * tmp
        resX0 = xp.linalg.norm(tmp)

        tmp = self.S1 + AIT_yI1 + Z1 - self.C1
        self.X1 = self.X1 + self.mult * self.rho * tmp
        resX1 = xp.linalg.norm(tmp)

        tmp = self.Sq + Zq
        self.Xq = self.Xq + self.mult * self.rho * tmp
        resXq = xp.linalg.norm(tmp)

        self.res_X = xp.sqrt(resX0**2 + resX1**2 + resXq**2)

    def update_rho(self):
        """
        Adjust the ADMM penalty parameter using primal and dual residuals.

        Modifies rho using the current constraint residuals and the res_X
        value calculated by update_X.
        """
        z0, z1, zq = self.equality.fun_AE(self.X0, self.X1, self.Xq)
        res_eq = (
            xp.linalg.norm(self.equality.bE0 - z0)
            / (1 + xp.linalg.norm(self.equality.bE0))
            + xp.linalg.norm(self.equality.bE1 - z1)
            / (1 + xp.linalg.norm(self.equality.bE1))
            + xp.linalg.norm(self.equality.bEq - zq)
            / (1 + xp.linalg.norm(self.equality.bEq))
        )
        res_inq = self.fejer.residual_norm(self.X0, self.X1) / (
            1 + abs(self.fejer.bI) * np.sqrt(self.Ngrid * self.blocks.n_pairs)
        )

        p_resnorm = res_eq + res_inq
        d_resnorm = self.res_X / (1 + self.normC)
        if d_resnorm > self.ratio * p_resnorm:
            self.rho = self.rho * self.factor
        if d_resnorm < self.ratio * p_resnorm:
            self.rho = self.rho / self.factor

    # ----------------------------- Optional iteration logging ------------------------------

    def print_updates(self):
        """
        Log objectives, residuals, and norms for the current iterate.

        Does nothing when verbose is false. Does not modify numerical state.
        """
        if not self.verbose:
            return

        obj_p = (
            xp.vdot(self.C0[:, self.blocks.idx_diag], self.X0[:, self.blocks.idx_diag])
            + xp.vdot(
                self.C1[:, self.blocks.idx_diag], self.X1[:, self.blocks.idx_diag]
            )
            + 2
            * xp.vdot(
                self.C0[:, self.blocks.idx_offdiag], self.X0[:, self.blocks.idx_offdiag]
            )
            + 2
            * xp.vdot(
                self.C1[:, self.blocks.idx_offdiag], self.X1[:, self.blocks.idx_offdiag]
            )
        )
        obj_d = (
            xp.vdot(self.yE0, self.equality.bE0)
            + xp.vdot(self.yE1, self.equality.bE1)
            + 2 * xp.vdot(self.yEq, self.equality.bEq)
            + 2 * self.fejer.bI * xp.sum(self.yI)
        )

        z0, z1, zq = self.equality.fun_AE(self.X0, self.X1, self.Xq)
        res_eq = (
            xp.linalg.norm(self.equality.bE0 - z0)
            / (1 + xp.linalg.norm(self.equality.bE0))
            + xp.linalg.norm(self.equality.bE1 - z1)
            / (1 + xp.linalg.norm(self.equality.bE1))
            + xp.linalg.norm(self.equality.bEq - zq)
            / (1 + xp.linalg.norm(self.equality.bEq))
        )

        res_inq = self.fejer.residual_norm(self.X0, self.X1) / (
            1 + abs(self.fejer.bI) * np.sqrt(self.Ngrid * self.blocks.n_pairs)
        )

        res_psdX = 0
        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1
            block0_rows = slice(self.blocks.d0[k_idx], self.blocks.d0[k_idx + 1])
            block1_rows = slice(self.blocks.d1[k_idx], self.blocks.d1[k_idx + 1])

            tmp = self.blocks.mat_block(self.X0[block0_rows], k)
            res_psdX += xp.linalg.norm(self.blocks._psd_projection(-tmp))

            tmp = self.blocks.mat_block(self.X1[block1_rows], k + 1)
            res_psdX += xp.linalg.norm(self.blocks._psd_projection(-tmp))

        res_psdX /= 1 + xp.linalg.norm(self.X0) + xp.linalg.norm(self.X1)

        Xq_blocks = self.Xq.T.reshape(self.blocks.n_pairs, 4, 4)
        res_psdQ = xp.linalg.norm(
            self.blocks._psd_projection(-Xq_blocks), axis=(-2, -1)
        ).sum()
        res_psdQ /= 1 + xp.linalg.norm(self.Xq)

        normS = xp.sqrt(
            xp.linalg.norm(self.S0) ** 2
            + xp.linalg.norm(self.S1) ** 2
            + xp.linalg.norm(self.Sq) ** 2
        )
        normX = xp.sqrt(
            xp.linalg.norm(self.X0) ** 2
            + xp.linalg.norm(self.X1) ** 2
            + xp.linalg.norm(self.Xq) ** 2
        )
        p_res = res_eq + res_inq + res_psdX + res_psdQ
        d_res = self.res_X / (1 + self.normC)

        logger.info(
            f"Iter {self.iteration}: p_res={p_res:.5f}, d_res={d_res:.5f}, "
            f"obj_primal={obj_p:.2f}, obj_dual={obj_d:.2f}, "
            f"duality gap={obj_p - obj_d:.2f}\n"
            f"        eq_res={res_eq:.5f}, inq_res={res_inq:.5f}, "
            f"psd_res={res_psdX + res_psdQ:.5f}, "
            f"|S|={normS:.2f}, |X|={normX:.2f}"
        )


class _SymmetricADMM:
    """
    Solve the symmetry-constrained NUG relaxation using ADMM.

    Owns the objective, iteration variables, solver settings, and
    degree-wise permutations used to project auxiliary blocks.
    Fixed layout and constraint operations are supplied by blocks,
    equality, and fejer.
    """

    def __init__(
        self,
        pre,
        equality,
        max_iter,
        rho,
        ratio,
        factor,
        mult,
        Nstep_yI,
        verbose,
    ):
        """
        Initialize a symmetric ADMM solve from prepared problem data.

        :param pre: Prepared objective, representation layout, Fejer
            constraints, and initial representation and slack arrays.
        :param equality: Fixed symmetric equality constraints.
        :param max_iter: Number of ADMM iterations.
        :param rho: Initial ADMM penalty parameter.
        :param ratio: Residual comparison ratio used to adjust rho.
        :param factor: Multiplicative factor used to adjust rho.
        :param mult: Step multiplier for representation-variable updates.
        :param Nstep_yI: Number of inequality-multiplier updates per iteration.
        :param verbose: Whether to log iteration diagnostics.
        """
        self.blocks = pre.blocks
        self.equality = equality
        self.fejer = pre.fejer

        self.C0 = pre.C0
        self.C1 = pre.C1
        self.normC = pre.normC

        self.X0 = pre.X0
        self.X1 = pre.X1
        self.Xq = pre.Xq
        self.S0 = pre.S0
        self.S1 = pre.S1
        self.Sq = pre.Sq

        self.max_iter = max_iter
        self.rho = rho
        self.ratio = ratio
        self.factor = factor
        self.mult = mult
        self.Nstep_yI = Nstep_yI
        self.verbose = verbose

        self.Ngrid = self.fejer.AI_mat_offdiag.shape[0]

        # Permutations used when projecting the auxiliary diagonal blocks.
        self.P = []
        for k in range(1, self.blocks.Lmax + 1):
            dk = 2 * k + 1
            Pk = np.eye(dk, dtype=np.float64)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
            self.P.append(xp.asarray(Pk))

        # Initialize the auxiliary representation and slack arrays.
        self.Xd0 = xp.zeros((self.blocks.D0, self.blocks.n_img), dtype=xp.float64)
        self.Xd1 = xp.zeros((self.blocks.D1, self.blocks.n_img), dtype=xp.float64)
        self.Sd0 = xp.zeros(self.Xd0.shape, dtype=xp.float64)
        self.Sd1 = xp.zeros(self.Xd1.shape, dtype=xp.float64)

        # Initialize the constraint multipliers.
        self.yI = xp.zeros((self.Ngrid, self.blocks.n_packed), dtype=xp.float64)
        self.yE = xp.zeros(equality.bE.shape, dtype=xp.float64)
        self.yEq = xp.zeros(equality.bEq.shape, dtype=xp.float64)

        self.iteration = 0
        self.res_X = None

    def solve(self):
        """
        Run ADMM and reconstruct the degree-wise representation matrices.

        Updates the current solver state. This instance is intended
        for one solve.

        :return: List of relaxed degree-wise representation matrices
            converted to NumPy arrays.
        """
        # Randomly order the slack and constraint-multiplier updates.
        update_order = np.arange(3)
        for t in trange(self.max_iter, desc="Symmetric NUG ADMM"):
            self.iteration = t
            np.random.shuffle(update_order)

            for update_idx in update_order:
                if update_idx == 0:
                    self.update_S()
                if update_idx == 1:
                    self.update_yE()
                if update_idx == 2:
                    for _ in range(self.Nstep_yI):
                        self.update_yI()

            self.update_X()
            if t % 100 == 0:
                self.print_updates()
            self.update_rho()

        # Reconstruct full degree-wise matrices for rotation recovery.
        X_admm = self.blocks._transform_coeff_back(self.X0, self.X1)
        return [xp.asnumpy(Xk) for Xk in X_admm]

    def update_S(self):
        """
        Update PSD slack variables for the current ADMM iterate.

        Form the unconstrained slack updates and project the representation,
        auxiliary diagonal, and quaternion blocks onto the PSD cone.

        Replaces S0, S1, Sd0, Sd1, and Sq.
        """
        # Equality multipliers mapped back to each primal variable block.
        Z0, Z1, Zd0, Zd1, Zq = self.equality.fun_AET(self.yE, self.yEq)

        # Inequality multipliers mapped back to the X0/X1 variable blocks.
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)

        # Unconstrained slack updates before PSD projection.
        S0 = self.C0 - Z0 - AIT_yI0 - self.X0 / self.rho
        S1 = self.C1 - Z1 - AIT_yI1 - self.X1 / self.rho
        Sd0 = -Zd0 - self.Xd0 / self.rho
        Sd1 = -Zd1 - self.Xd1 / self.rho
        Sq = -Zq - self.Xq / self.rho

        # Project packed X0/X1 blocks degree by degree.
        S0, S1 = self.blocks._project_representation_blocks(S0, S1)

        # Project the diagonal coupling blocks.
        Sd0 = Sd0.T
        Sd1 = Sd1.T
        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1  # degree
            block0_rows = slice(self.blocks.d0[k_idx], self.blocks.d0[k_idx + 1])
            block1_rows = slice(self.blocks.d1[k_idx], self.blocks.d1[k_idx + 1])
            permutation = self.P[k_idx]

            tmp = self.blocks.transform_back_block(
                Sd0[:, block0_rows],
                Sd1[:, block1_rows],
                k,
                permutation,
            )
            tmp = self.blocks._psd_projection(tmp)

            Sd0[:, block0_rows], Sd1[:, block1_rows] = self.blocks.transform_block(
                tmp, k, permutation
            )

        Sd0 = Sd0.T
        Sd1 = Sd1.T

        # Project each 4x4 quaternion slack block for off-diagonal image pairs.
        Sq = self.blocks._psd_projection(Sq.T.reshape(self.blocks.n_pairs, 4, 4))
        Sq = Sq.T.reshape(-1, self.blocks.n_pairs)

        self.S0, self.S1, self.Sd0, self.Sd1, self.Sq = S0, S1, Sd0, Sd1, Sq

    def update_yE(self):
        """
        Update equality multipliers for the current ADMM iterate.

        yE enforces the per-degree symmetry/diagonal constraints.
        yEq enforces the off-diagonal quaternion constraints.

        Replaces yE and yEq.
        """
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)
        z, zq = self.equality.fun_AE(
            -self.X0 / self.rho + self.C0 - self.S0 - AIT_yI0,
            -self.X1 / self.rho + self.C1 - self.S1 - AIT_yI1,
            -self.Xd0 / self.rho - self.Sd0,
            -self.Xd1 / self.rho - self.Sd1,
            -self.Xq / self.rho - self.Sq,
        )
        self.yE, self.yEq = self.equality.solve_normal(
            self.equality.bE / self.rho + z,
            self.equality.bEq / self.rho + zq,
        )

    def update_yI(self):
        """
        Update the nonnegative Fejer inequality multipliers.

        Each multiplier corresponds to one SO(3) grid point and one
        upper-triangular image pair, including self-pairs.

        Replaces yI using the current solver state and penalty parameter.
        """
        Z0, Z1, _, _, _ = self.equality.fun_AET(self.yE, self.yEq)
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)
        tmp = self.fejer.fun_AI(
            -self.X0 / self.rho + self.C0 - self.S0 - Z0 - AIT_yI0,
            -self.X1 / self.rho + self.C1 - self.S1 - Z1 - AIT_yI1,
        )
        self.yI = (
            self.yI
            + self.fejer.bI / self.rho / self.fejer.Lambda
            + tmp / self.fejer.Lambda
        )
        self.yI = xp.maximum(self.yI, 0)

    def update_X(self):
        """
        Update the representation, auxiliary diagonal, and quaternion arrays.

        Advance each array by mult * rho times its update-equation residual.
        Store the combined residual norm in res_X.

        Replaces X0, X1, Xd0, Xd1, and Xq.
        """
        Z0, Z1, Zd0, Zd1, Zq = self.equality.fun_AET(self.yE, self.yEq)
        AIT_yI0, AIT_yI1 = self.fejer.fun_AIT(self.yI)
        tmp = self.S0 + Z0 + AIT_yI0 - self.C0
        self.X0 = self.X0 + self.mult * self.rho * tmp
        resX0 = xp.linalg.norm(tmp)
        tmp = self.S1 + Z1 + AIT_yI1 - self.C1
        self.X1 = self.X1 + self.mult * self.rho * tmp
        resX1 = xp.linalg.norm(tmp)
        tmp = self.Sd0 + Zd0
        self.Xd0 = self.Xd0 + self.mult * self.rho * tmp
        resXd0 = xp.linalg.norm(tmp)
        tmp = self.Sd1 + Zd1
        self.Xd1 = self.Xd1 + self.mult * self.rho * tmp
        resXd1 = xp.linalg.norm(tmp)
        tmp = self.Sq + Zq
        self.Xq = self.Xq + self.mult * self.rho * tmp
        resXq = xp.linalg.norm(tmp)
        self.res_X = xp.sqrt(resX0**2 + resX1**2 + resXd0**2 + resXd1**2 + resXq**2)

    def update_rho(self):
        """
        Adjust the ADMM penalty parameter using primal and dual residuals.

        Calculate the primal residual measure from the equality and Fejer
        constraints, and the dual residual measure from res_X.

        Modifies rho according to the configured ratio and factor.
        """
        z, zq = self.equality.fun_AE(self.X0, self.X1, self.Xd0, self.Xd1, self.Xq)
        res_eq = xp.linalg.norm(z - self.equality.bE) / (
            1 + xp.linalg.norm(self.equality.bE)
        ) + xp.linalg.norm(zq - self.equality.bEq) / (
            1 + xp.linalg.norm(self.equality.bEq)
        )
        res_inq = self.fejer.residual_norm(self.X0, self.X1) / (
            1 + abs(self.fejer.bI) * np.sqrt(self.Ngrid * self.blocks.n_img**2)
        )
        p_resnorm = res_eq + res_inq
        d_resnorm = self.res_X / (1 + self.normC)
        if d_resnorm > self.ratio * p_resnorm:
            self.rho = self.rho * self.factor
        if d_resnorm < self.ratio * p_resnorm:
            self.rho = self.rho / self.factor

    # ----------------------------- Optional iteration logging ------------------------------

    def print_updates(self):
        """
        Log objectives, residuals, and norms for the current symmetric iterate.

        Includes PSD residuals for the representation, auxiliary diagonal,
        and quaternion blocks. Does nothing when verbose is false and does
        not modify numerical state.
        """
        if not self.verbose:
            return
        obj_p = (
            xp.vdot(self.C0[:, self.blocks.idx_diag], self.X0[:, self.blocks.idx_diag])
            + xp.vdot(
                self.C1[:, self.blocks.idx_diag], self.X1[:, self.blocks.idx_diag]
            )
            + 2
            * xp.vdot(
                self.C0[:, self.blocks.idx_offdiag], self.X0[:, self.blocks.idx_offdiag]
            )
            + 2
            * xp.vdot(
                self.C1[:, self.blocks.idx_offdiag], self.X1[:, self.blocks.idx_offdiag]
            )
        )

        obj_d = (
            xp.vdot(self.yE, self.equality.bE)
            + 2 * xp.vdot(self.yEq, self.equality.bEq)
            + self.fejer.bI * xp.sum(self.yI[:, self.blocks.idx_diag])
            + 2 * self.fejer.bI * xp.sum(self.yI[:, self.blocks.idx_offdiag])
        )

        z, zq = self.equality.fun_AE(self.X0, self.X1, self.Xd0, self.Xd1, self.Xq)
        res_eq = xp.linalg.norm(z - self.equality.bE) / (
            1 + xp.linalg.norm(self.equality.bE)
        ) + xp.linalg.norm(zq - self.equality.bEq) / (
            1 + xp.linalg.norm(self.equality.bEq)
        )
        res_inq = self.fejer.residual_norm(self.X0, self.X1) / (
            1
            + abs(self.fejer.bI)
            * np.sqrt(self.Ngrid * self.blocks.n_img * (self.blocks.n_img + 1) / 2)
        )

        res_psdX = 0
        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1  # degree
            block0_rows = slice(self.blocks.d0[k_idx], self.blocks.d0[k_idx + 1])
            block1_rows = slice(self.blocks.d1[k_idx], self.blocks.d1[k_idx + 1])

            tmp = self.blocks.mat_block(self.X0[block0_rows], k)
            res_psdX += xp.linalg.norm(self.blocks._psd_projection(-tmp))

            tmp = self.blocks.mat_block(self.X1[block1_rows], k + 1)
            res_psdX += xp.linalg.norm(self.blocks._psd_projection(-tmp))

        res_psdX = res_psdX / (1 + xp.linalg.norm(self.X0) + xp.linalg.norm(self.X1))

        res_psdD = 0
        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1  # degree
            block0_rows = slice(self.blocks.d0[k_idx], self.blocks.d0[k_idx + 1])
            block1_rows = slice(self.blocks.d1[k_idx], self.blocks.d1[k_idx + 1])

            tmp = self.blocks.transform_back_block(
                self.Xd0[block0_rows].T,
                self.Xd1[block1_rows].T,
                k,
                self.P[k_idx],
            )
            res_psdD += xp.linalg.norm(
                self.blocks._psd_projection(-tmp), axis=(-2, -1)
            ).sum()
        res_psdD = res_psdD / (1 + xp.linalg.norm(self.Xd0) + xp.linalg.norm(self.Xd1))

        res_psdQ = 0
        for pair_idx in range(self.blocks.n_pairs):
            tmp = self.Xq[:, pair_idx].reshape(4, 4).T
            res_psdQ += xp.linalg.norm(self.blocks._psd_projection(-tmp))
        res_psdQ = res_psdQ / (1 + xp.linalg.norm(self.Xq))

        normS = xp.sqrt(
            xp.linalg.norm(self.S0) ** 2
            + xp.linalg.norm(self.S1) ** 2
            + xp.linalg.norm(self.Sd0) ** 2
            + xp.linalg.norm(self.Sd1) ** 2
            + xp.linalg.norm(self.Sq) ** 2
        )
        normX = xp.sqrt(
            xp.linalg.norm(self.X0) ** 2
            + xp.linalg.norm(self.X1) ** 2
            + xp.linalg.norm(self.Xd0) ** 2
            + xp.linalg.norm(self.Xd1) ** 2
            + xp.linalg.norm(self.Xq) ** 2
        )
        p_res = res_eq + res_inq + res_psdX + res_psdD + res_psdQ
        d_res = self.res_X / (1 + self.normC)
        logger.info(
            f"Iter {self.iteration}: p_res={p_res:.5f}, d_res={d_res:.5f}, "
            f"obj_primal={obj_p:.2f}, obj_dual={obj_d:.2f}, "
            f"duality gap={obj_p - obj_d:.2f}\n"
            f"        eq_res={res_eq:.5f}, inq_res={res_inq:.5f}, "
            f"psd_res={res_psdX + res_psdD + res_psdQ:.5f}, "
            f"|S|={normS:.2f}, |X|={normX:.2f}"
        )


#############################
# ADMM Equality Constraints #
#############################


class _C1EqualityConstraints:
    """
    Fixed equality constraints for asymmetric (C1) ADMM.

    Owns the identity right-hand sides for self-pair representation blocks
    and uses the shared quaternion constraints. Current ADMM variables
    are supplied explicitly to each operation.

    In the docstrings below, N is the number of images. X0 and X1 contain
    the two representation block families, with D0 and D1 rows, respectively.
    Their columns correspond to upper-triangular image pairs, including
    self-pairs.
    """

    def __init__(self, blocks, quaternions):
        """
        Prepare the C1 equality operators and right-hand sides.

        :param blocks: Fixed representation layout.
        :param quaternions: Shared quaternion equality constraints.
        """
        self.blocks = blocks
        self.quaternions = quaternions

        # Local references used to construct the formulation-specific right-hand sides.
        n_img = blocks.n_img
        d0, d1 = blocks.d0, blocks.d1

        # Build the right-hand sides requiring each self-pair block
        # of X0 and X1 to be an identity matrix.
        bE0 = xp.zeros(blocks.D0, dtype=np.float64)
        bE1 = xp.zeros(blocks.D1, dtype=np.float64)

        for k_idx in range(len(d0) - 1):
            k = k_idx + 1

            # A flattened m-by-m identity has ones every m+1 entries.
            bE0[d0[k_idx] : d0[k_idx + 1] : k + 1] = 1
            bE1[d1[k_idx] : d1[k_idx + 1] : k + 2] = 1

        self.bE0 = xp.repeat(bE0[:, None], n_img, axis=1)
        self.bE1 = xp.repeat(bE1[:, None], n_img, axis=1)

    @property
    def bEq(self):
        """
        Right-hand side of the shared quaternion equality constraints.
        """
        return self.quaternions.bEq

    def fun_AE(self, X0, X1, Xq):
        """
        Apply the C1 equality operator.

        Select the self-pair representation blocks and apply the quaternion
        operator to Xq and the lowest-degree distinct-pair entries of X0/X1.

        :param X0: First representation array of shape (D0, N(N+1)/2).
        :param X1: Second representation array of shape (D1, N(N+1)/2).
        :param Xq: Quaternion array of shape (16, N(N-1)/2).

        :return: Tuple (z0, z1, zq), with shapes (D0, N), (D1, N),
            and (17, N(N-1)/2). The equality constraints require
            z0 == bE0, z1 == bE1, and zq == bEq.
        """
        z0 = X0[:, self.blocks.idx_diag]
        z1 = X1[:, self.blocks.idx_diag]
        zq = self.quaternions.fun_AE(X0, X1, Xq)
        return z0, z1, zq

    def fun_AET(self, yE0, yE1, yEq):
        """
        Apply the adjoint of the C1 equality operator.

        Map equality multipliers back to contributions in the representation
        and quaternion arrays.

        :param yE0: Multipliers for the X0 self-pair equalities,
            of shape (D0, N).
        :param yE1: Multipliers for the X1 self-pair equalities,
            of shape (D1, N).
        :param yEq: Multipliers for the quaternion equalities,
            of shape (17, N(N-1)/2).

        :return: Tuple (Z0, Z1, Zq), with shapes (D0, N(N+1)/2),
            (D1, N(N+1)/2), and (16, N(N-1)/2), respectively.
        """
        Z0 = xp.zeros((self.blocks.D0, self.blocks.n_packed), dtype=np.float64)
        Z1 = xp.zeros((self.blocks.D1, self.blocks.n_packed), dtype=np.float64)
        Z0[:, self.blocks.idx_diag] = yE0
        Z1[:, self.blocks.idx_diag] = yE1

        # Place the quaternion adjoint contributions in distinct-pair columns.
        Zq0, Zq1, Zq = self.quaternions.fun_AET(yEq)
        Z0[:1, self.blocks.idx_offdiag] = Zq0
        Z1[:4, self.blocks.idx_offdiag] = Zq1

        return Z0, Z1, Zq

    def solve_normal(self, rhs0, rhs1, rhs_q):
        """
        Apply the pseudoinverse of the C1 equality operator times its adjoint.

        The self-pair selection operators times their adjoints are identity
        operators, so their right-hand sides are returned unchanged.
        Apply the pseudoinverse of AEq @ AEq.T to rhs_q.

        :param rhs0: Right-hand side for the X0 self-pair system,
            of shape (D0, N).
        :param rhs1: Right-hand side for the X1 self-pair system,
            of shape (D1, N).
        :param rhs_q: Right-hand side for the quaternion system,
            of shape (17, N(N-1)/2).

        :return: Tuple (yE0, yE1, yEq), with the same shapes as the inputs.
            The first two outputs are the supplied rhs0 and rhs1 arrays.
        """
        return rhs0, rhs1, self.quaternions.solve_normal(rhs_q)


class _SymmetricEqualityConstraints:
    """
    Fixed equality constraints for symmetric ADMM.

    Owns the per-degree symmetry/diagonal operators and right-hand sides,
    and uses the shared quaternion constraints. Current ADMM variables
    are supplied explicitly to each operation.

    In the docstrings below, N is the number of images and Lmax is the
    maximum representation degree. X0 and X1 contain the two representation
    block families, with D0 and D1 rows, respectively. Their columns correspond
    to upper-triangular image pairs, including self-pairs.
    """

    def __init__(self, blocks, quaternions, rank_Ak):
        """
        Construct the symmetric equality operators and right-hand sides.

        :param blocks: Fixed representation layout.
        :param quaternions: Shared quaternion equality constraints.
        :param rank_Ak: Symmetry-projector rank at each degree.
        """
        self.blocks = blocks
        self.quaternions = quaternions

        # Local references used to construct the formulation-specific right-hand sides.
        n_img = blocks.n_img
        d0, d1 = blocks.d0, blocks.d1

        # Build the left-hand-side operators for the symmetry equality constraints.
        self.AE = []
        self.AEAETinv = []
        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1
            size0 = k**2
            size1 = (k + 1) ** 2
            n_components = size0 + size1

            AEk = xp.zeros(
                (1 + n_components, 2 * n_components),
                dtype=np.float64,
            )

            # Trace constraint on the two representation components.
            AEk[0, : size0 : k + 1] = 1
            AEk[0, size0 : n_components : k + 2] = 1

            # Couple representation entries to auxiliary diagonal entries.
            for component_idx in range(n_components):
                row_idx = component_idx + 1
                AEk[row_idx, component_idx] = 1
                AEk[row_idx, component_idx + n_components] = 1

            self.AE.append(AEk)
            self.AEAETinv.append(xp.linalg.pinv(AEk @ AEk.T))

        # Build the right-hand sides: symmetry-projector ranks and identity-matrix entries.
        bE = xp.zeros(
            blocks.Lmax + blocks.D0 + blocks.D1,
            dtype=np.float64,
        )

        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1

            start = k_idx + d0[k_idx] + d1[k_idx]
            split = k_idx + 1 + d0[k_idx + 1] + d1[k_idx]
            stop = k_idx + 1 + d0[k_idx + 1] + d1[k_idx + 1]

            bE[start] = rank_Ak[k_idx]
            bE[start + 1 : split : k + 1] = 1
            bE[split : stop : k + 2] = 1

        self.bE = xp.repeat(bE[:, None], n_img, axis=1)

    @property
    def bEq(self):
        """
        Right-hand side of the shared quaternion equality constraints.
        """
        return self.quaternions.bEq

    def _constraint_rows(self, k_idx):
        """
        Return the equality-row slice for one degree.

        :param k_idx: Zero-based degree index.

        :return: Slice selecting the trace and block sum equality rows for this degree.
        """
        start = k_idx + self.blocks.d0[k_idx] + self.blocks.d1[k_idx]
        stop = k_idx + 1 + self.blocks.d0[k_idx + 1] + self.blocks.d1[k_idx + 1]
        return slice(start, stop)

    def fun_AE(self, X0, X1, Xd0, Xd1, Xq):
        """
        Apply the symmetric equality operator.

        The per-image constraints couple self-pair blocks of X0/X1 to the
        auxiliary blocks Xd0/Xd1 and constrain the combined representation trace.

        The quaternion constraints couple Xq to the lowest degree entries of
        X0/X1 for distinct image pairs.

        :param X0: First representation array of shape (D0, N(N+1)/2).
        :param X1: Second representation array of shape (D1, N(N+1)/2).
        :param Xd0: First auxiliary diagonal-block array of shape (D0, N).
        :param Xd1: Second auxiliary diagonal-block array of shape (D1, N).
        :param Xq: Quaternion array of shape (16, N(N-1)/2).

        :return: Tuple (z, zq), where z has shape (Lmax+D0+D1, N) and zq
            has shape (17, N(N-1)/2). The equality constraints require
            z == bE and zq == bEq.
        """
        z = xp.zeros(self.bE.shape, dtype=np.float64)

        for k_idx in range(self.blocks.Lmax):
            block0_rows = slice(self.blocks.d0[k_idx], self.blocks.d0[k_idx + 1])
            block1_rows = slice(self.blocks.d1[k_idx], self.blocks.d1[k_idx + 1])

            z[self._constraint_rows(k_idx)] = self.AE[k_idx] @ xp.concatenate(
                (
                    X0[block0_rows, self.blocks.idx_diag],
                    X1[block1_rows, self.blocks.idx_diag],
                    Xd0[block0_rows],
                    Xd1[block1_rows],
                ),
                axis=0,
            )

        zq = self.quaternions.fun_AE(X0, X1, Xq)
        return z, zq

    def fun_AET(self, yE, yEq):
        """
        Apply the adjoint of the symmetric equality operator.

        Map equality multipliers back to contributions in the representation,
        auxiliary diagonal-block, and quaternion arrays.

        :param yE: Multipliers for the per-image trace and block-sum equalities,
            of shape (Lmax+D0+D1, N).
        :param yEq: Multipliers for the quaternion equalities,
            of shape (17, N(N-1)/2).

        :return: Tuple (Z0, Z1, Zd0, Zd1, Zq), with shapes
            (D0, N(N+1)/2), (D1, N(N+1)/2), (D0, N), (D1, N),
            and (16, N(N-1)/2), respectively.
        """
        Z0 = xp.zeros((self.blocks.D0, self.blocks.n_packed), dtype=np.float64)
        Z1 = xp.zeros((self.blocks.D1, self.blocks.n_packed), dtype=np.float64)
        Zd0 = xp.zeros((self.blocks.D0, self.blocks.n_img), dtype=np.float64)
        Zd1 = xp.zeros((self.blocks.D1, self.blocks.n_img), dtype=np.float64)

        for k_idx in range(self.blocks.Lmax):
            k = k_idx + 1
            size0 = k**2
            size1 = (k + 1) ** 2

            block0_rows = slice(self.blocks.d0[k_idx], self.blocks.d0[k_idx + 1])
            block1_rows = slice(self.blocks.d1[k_idx], self.blocks.d1[k_idx + 1])

            Ztmp = self.AE[k_idx].T @ yE[self._constraint_rows(k_idx)]

            Z0[block0_rows, self.blocks.idx_diag] = Ztmp[:size0]
            Z1[block1_rows, self.blocks.idx_diag] = Ztmp[size0 : size0 + size1]
            Zd0[block0_rows] = Ztmp[size0 + size1 : 2 * size0 + size1]
            Zd1[block1_rows] = Ztmp[2 * size0 + size1 : 2 * size0 + 2 * size1]

        # Place the quaternion adjoint contributions in distinct-pair columns.
        Zq0, Zq1, Zq = self.quaternions.fun_AET(yEq)
        Z0[:1, self.blocks.idx_offdiag] = Zq0
        Z1[:4, self.blocks.idx_offdiag] = Zq1

        return Z0, Z1, Zd0, Zd1, Zq

    def solve_normal(self, rhs, rhs_q):
        """
        Apply the pseudoinverses of the equality operators times their adjoints.

        Apply each per-degree pseudoinverse of AEk @ AEk.T to the corresponding
        rows of rhs, and the pseudoinverse of AEq @ AEq.T to rhs_q.

        :param rhs: Right-hand side for the per-image equality system,
            of shape (Lmax+D0+D1, N).
        :param rhs_q: Right-hand side for the quaternion equality system,
            of shape (17, N(N-1)/2).

        :return: Tuple (yE, yEq) containing the solutions obtained with the
            pseudoinverses, with the same shapes as rhs and rhs_q.
        """
        yE = xp.zeros(self.bE.shape, dtype=np.float64)

        for k_idx in range(self.blocks.Lmax):
            rows = self._constraint_rows(k_idx)
            yE[rows] = self.AEAETinv[k_idx] @ rhs[rows]

        yEq = self.quaternions.solve_normal(rhs_q)
        return yE, yEq


class _QuaternionConstraints:
    """
    Quaternion equality constraints shared by the C1 and symmetric solvers.

    For each pair of different images, these equations relate an auxiliary
    4-by-4 quaternion matrix to five entries from the degree-one representation
    blocks in X0 and X1. They also require the quaternion matrix to have trace one.

    This class builds the constraint matrix and right-hand side, applies the
    operator and its adjoint, and solves the associated multiplier equations.

    In the method docstrings, N is the number of images, and D0/D1 are the
    row counts of X0/X1.
    """

    def __init__(self, blocks):
        """
        Construct the quaternion operator, pseudoinverse, and right-hand side.

        :param blocks: Fixed representation layout.
        """
        self.blocks = blocks
        self.AEq = xp.asarray(self._construct_AEq())
        self.AEqAEqtinv = xp.linalg.pinv(self.AEq @ self.AEq.T)

        bEq = xp.zeros(17, dtype=np.float64)
        bEq[:16:5] = 0.25  # Flattened identity matrix divided by four.
        bEq[-1] = 1  # Unit trace.
        self.bEq = xp.repeat(bEq[:, None], blocks.n_pairs, axis=1)

    def fun_AE(self, X0, X1, Xq):
        """
        Apply the quaternion equality operator.

        :param X0: First representation array of shape (D0, N(N+1)/2).
        :param X1: Second representation array of shape (D1, N(N+1)/2).
        :param Xq: Quaternion array of shape (16, N(N-1)/2).

        :return: Equality values of shape (17, N(N-1)/2).
        """
        return self.AEq @ xp.concatenate(
            (
                Xq,
                X0[:1, self.blocks.idx_offdiag],
                X1[:4, self.blocks.idx_offdiag],
            ),
            axis=0,
        )

    def fun_AET(self, yEq):
        """
        Apply the adjoint of the quaternion equality operator.

        :param yEq: Quaternion equality multipliers of shape (17, N(N-1)/2).

        :return: Tuple containing contributions to the lowest-degree X0
            entries, lowest-degree X1 entries, and Xq. Their shapes are
            (1, N(N-1)/2), (4, N(N-1)/2), and (16, N(N-1)/2).
            The representation contributions correspond to distinct pairs.
        """
        Z = self.AEq.T @ yEq
        return Z[16:17], Z[17:], Z[:16]

    def solve_normal(self, rhs):
        """
        Apply the pseudoinverse of AEq @ AEq.T.

        :param rhs: Right-hand side of shape (17, N(N-1)/2).

        :return: Solution obtained with the pseudoinverse, of the same shape.
        """
        return self.AEqAEqtinv @ rhs

    @staticmethod
    def _construct_AEq():
        """
        Construct the linear equality operator encoding the quaternion constraints.

        The operator acts on 21 variables consisting of the 16 vectorized
        entries of an auxiliary 4-by-4 quaternion matrix followed by five
        entries from the lowest-degree representation blocks. The first
        16 rows couple the quaternion and representation variables. The
        final row constrains the trace of the quaternion matrix.

        :return: Equality-constraint matrix with shape (17, 21).
        """
        AEq = np.zeros((17, 21), np.float64)

        # First 16 rows: identity constraints on first 16 variables
        AEq[:16, :16] = np.eye(16, dtype=np.float64)

        # Columns 16:21 map the low-degree X0/X1 entries into the quaternion
        # convex-hull constraint Xq = I/4 - linear(X^(1)).
        extra = 0.25 * np.array(
            [
                [-1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0],
                [-1, -1, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [1, -1, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # `extra` uses the original column-wise order
        # [X0_00, X1_00, X1_10, X1_01, X1_11].
        # Swap the two off-diagonal columns to match row-wise packing of X1.
        AEq[:16, 16:] = extra[:, [0, 1, 3, 2, 4]]

        # Last row: redundant trace/sum constraint
        AEq[16, [0, 5, 10, 15]] = 1

        return AEq


###############################
# ADMM Inequality Constraints #
###############################


class _FejerConstraints:
    """
    Class containing the data and operations for the discretized Fejer inequalities.

    Constructs the constraint matrices, bound, and update scale from
    representation weights, and provides the forward, adjoint, and
    residual operations used by both ADMM solvers.

    Note, C1 processes off-diagonal image pairs in chunks to limit peak memory use;
    the symmetric solver also applies the self-pair constraint.
    """

    def __init__(
        self,
        W0,
        W1,
        blocks,
        pair_batch_size,
        is_asymmetric,
    ):
        """
        Construct the Fejer inequalities from representation weights.

        Let G be the number of SO(3) grid points. Each list contains one
        weight array per representation degree, starting at degree one.

        :param W0: First representation weights; degree-k entries have
            shape (G, k, k).
        :param W1: Second representation weights; degree-k entries have
            shape (G, k+1, k+1).
        :param blocks: Fixed representation layout.
        :param pair_batch_size: Number of distinct image pairs per C1 batch.
        :param is_asymmetric: Whether to use the asymmetric C1 formulation.
        """
        self.idx_diag = blocks.idx_diag
        self.idx_offdiag = blocks.idx_offdiag
        self.D0 = blocks.D0
        self.n_packed = blocks.n_packed
        self.pair_batch_size = pair_batch_size
        self.is_asymmetric = is_asymmetric

        self._build_operators(W0, W1, blocks)

    def _build_operators(self, W0, W1, blocks):
        """
        Construct and store the Fejer operators, bound, and update scale.

        :param W0: First representation weights; degree-k entries have
            shape (G, k, k).
        :param W1: Second representation weights; degree-k entries have
            shape (G, k+1, k+1).
        :param blocks: Fixed representation layout defining the degree
            boundaries and representation row counts.
        """
        Lmax = blocks.Lmax
        d0, d1 = blocks.d0, blocks.d1
        D0, D1 = blocks.D0, blocks.D1
        Ngrid = W0[0].shape[0]

        # Build the operator for distinct image pairs.
        AI_mat_offdiag = np.zeros((Ngrid, D0 + D1), dtype=np.float64)
        for k_idx in range(Lmax):
            k = k_idx + 1
            scale = (Lmax - k + 2) * (Lmax - k + 1) * (k + 0.5)
            block0_cols = slice(d0[k_idx], d0[k_idx + 1])
            block1_cols = slice(D0 + d1[k_idx], D0 + d1[k_idx + 1])

            block0 = scale * W0[k_idx].reshape(Ngrid, -1)
            block1 = scale * W1[k_idx].reshape(Ngrid, -1)

            AI_mat_offdiag[:, block0_cols] = block0
            AI_mat_offdiag[:, block1_cols] = block1

        # Build the self-pair operator using the symmetric part of each weight matrix.
        AI_mat_diag = np.zeros((Ngrid, D0 + D1), dtype=np.float64)
        for k_idx in range(Lmax):
            k = k_idx + 1
            scale = (Lmax - k + 2) * (Lmax - k + 1) * (k + 0.5)
            block0_cols = slice(d0[k_idx], d0[k_idx + 1])
            block1_cols = slice(D0 + d1[k_idx], D0 + d1[k_idx + 1])

            W0_sym = 0.5 * (W0[k_idx] + W0[k_idx].transpose(0, 2, 1))
            W1_sym = 0.5 * (W1[k_idx] + W1[k_idx].transpose(0, 2, 1))

            block0 = scale * W0_sym.reshape(Ngrid, -1)
            block1 = scale * W1_sym.reshape(Ngrid, -1)

            AI_mat_diag[:, block0_cols] = block0
            AI_mat_diag[:, block1_cols] = block1

        self.AI_mat_diag = xp.asarray(AI_mat_diag)
        self.AI_mat_offdiag = xp.asarray(AI_mat_offdiag)
        self.bI = -(Lmax + 2) * (Lmax + 1) / 2

        # Scale used by the inequality-multiplier update.
        self.Lambda = self._largest_eigenvalue_gram(self.AI_mat_offdiag)

    @staticmethod
    def _largest_eigenvalue_gram(AI):
        """
        Compute the denominator for the Fejer inequality-multiplier update.

        The largest eigenvalue of AI @ AI.T is computed from the smaller
        AI.T @ AI Gram matrix. Retain the fixed 2000 offset used by the
        original implementation.

        :param AI: Real Fejer inequality operator of shape (Ngrid, D0 + D1).

        :return: Largest eigenvalue of AI @ AI.T, plus 2000.
        """
        return xp.linalg.eigvalsh(AI.T @ AI)[-1] + 2000

    def pair_chunks(self):
        """
        Yield distinct-pair column slices for bounded C1 GPU operations.
        """
        n_pairs = self.idx_offdiag.size
        for start in range(0, n_pairs, self.pair_batch_size):
            yield slice(start, min(start + self.pair_batch_size, n_pairs))

    def fun_AI(self, X0, X1):
        """
        Evaluate AI on the two representation blocks.

        C1 inputs contain one chunk of distinct pair columns. Symmetric inputs
        contain all packed pairs. The output has one row per SO(3) grid point
        and one column per input image pair.
        """
        tmp = xp.concatenate((X0, X1), axis=0)

        if self.is_asymmetric:
            # C1 passes a chunk containing only distinct image pairs.
            return self.AI_mat_offdiag @ tmp

        # The symmetric solver passes all packed pairs, including self-pairs.
        z = xp.zeros((self.AI_mat_offdiag.shape[0], self.n_packed), dtype=np.float64)
        z[:, self.idx_diag] = self.AI_mat_diag @ tmp[:, self.idx_diag]
        z[:, self.idx_offdiag] = self.AI_mat_offdiag @ tmp[:, self.idx_offdiag]
        return z

    def fun_AIT_chunk(self, yI_chunk):
        """
        Apply the distinct-pair Fejer adjoint to a C1 chunk.
        """
        return self.AI_mat_offdiag.T @ yI_chunk

    def fun_AIT(self, yI):
        """
        Apply the adjoint of AI to inequality multipliers.

        Return the two representation blocks in packed image-pair order. The
        self-pair columns are zero for C1.
        """
        Z = xp.zeros((self.AI_mat_offdiag.shape[1], self.n_packed), dtype=np.float64)

        if self.is_asymmetric:
            for columns in self.pair_chunks():
                packed_columns = self.idx_offdiag[columns]
                Z[:, packed_columns] = self.fun_AIT_chunk(yI[:, columns])
        else:
            Z[:, self.idx_diag] = self.AI_mat_diag.T @ yI[:, self.idx_diag]
            Z[:, self.idx_offdiag] = self.AI_mat_offdiag.T @ yI[:, self.idx_offdiag]

        return Z[: self.D0], Z[self.D0 :]

    def residual_norm(self, X0, X1):
        """
        Compute the norm of the positive Fejer inequality residual in chunks.
        This avoids storing a residual over the full Fejer grid.
        """
        if not self.is_asymmetric:
            return xp.linalg.norm(xp.maximum(self.bI - self.fun_AI(X0, X1), 0))

        squared_norm = xp.zeros((), dtype=np.float64)
        for columns in self.pair_chunks():
            packed_columns = self.idx_offdiag[columns]
            residual = xp.maximum(
                self.bI
                - self.fun_AI(
                    X0[:, packed_columns],
                    X1[:, packed_columns],
                ),
                0,
            )
            squared_norm += xp.sum(residual * residual)

        return xp.sqrt(squared_norm)


##############################################
# Representation Layout and Block Operations #
##############################################


class _RepresentationBlocks:
    """
    Fixed representation layout and operations used by ADMM.

    Stores degree-block row boundaries and image-pair indices. Representation
    arrays have one column per upper-triangular image pair, including self-pairs.
    Current representation and slack arrays are supplied to methods explicitly.

    In the docstrings below, N is the number of images, and D0/D1 are
    the row counts of the two representation arrays.
    """

    def __init__(self, n_img, Lmax):
        """
        Construct the degree and image-pair layout.

        :param n_img: Number of images.
        :param Lmax: Maximum representation degree.
        """
        self.n_img = n_img
        self.Lmax = Lmax

        self.d0 = [0]
        self.d1 = [0]
        for k in range(1, Lmax + 1):
            self.d0.append(self.d0[-1] + k**2)
            self.d1.append(self.d1[-1] + (k + 1) ** 2)

        self.D0 = self.d0[-1]
        self.D1 = self.d1[-1]
        self.n_pairs = n_img * (n_img - 1) // 2
        self.n_packed = n_img * (n_img + 1) // 2

        # Column indices for self-pairs (i, i) and distinct pairs (i, j), i < j.
        iu, ju = np.triu_indices(n_img)
        self.idx_diag = xp.asarray(np.flatnonzero(iu == ju))
        self.idx_offdiag = xp.asarray(np.flatnonzero(iu != ju))

        # Positions in the full N-by-N image-pair grid. Lower-triangular
        # positions follow the same pair order as idx_offdiag.
        iu_off, ju_off = np.triu_indices(n_img, k=1)
        self.IDX_upper = xp.asarray(iu * n_img + ju)
        self.IDX_lower = xp.asarray(ju_off * n_img + iu_off)

    def _project_representation_blocks(self, Z0, Z1):
        """
        Project the degree-wise representation arrays onto the PSD cone.

        Modifies both supplied arrays in place, using the stored representation
        layout.

        :param Z0: First representation array, of shape (D0, N(N+1)/2).
        :param Z1: Second representation array, of shape (D1, N(N+1)/2).

        :return: The modified arrays (Z0, Z1).
        """
        for k_idx in range(self.Lmax):
            k = k_idx + 1

            block0_rows = slice(
                self.d0[k_idx],
                self.d0[k_idx + 1],
            )
            block1_rows = slice(
                self.d1[k_idx],
                self.d1[k_idx + 1],
            )

            block0 = self.mat_block(Z0[block0_rows], k)
            block0 = self._psd_projection(block0)
            Z0[block0_rows] = self.vec_block(block0, k)

            block1 = self.mat_block(Z1[block1_rows], k + 1)
            block1 = self._psd_projection(block1)
            Z1[block1_rows] = self.vec_block(block1, k + 1)
        return Z0, Z1

    def _transform_coeff(self, A):
        """
        Convert full degree-wise matrices into the two representation arrays.

        :param A: List of full representation matrices for degrees 1 through Lmax.
            The degree-k matrix has shape (N*(2*k+1), N*(2*k+1)).

        :return: Tuple (A0, A1), with shapes (D0, N(N+1)/2) and
            (D1, N(N+1)/2).
        """
        A0 = xp.zeros((self.D0, self.n_packed), dtype=np.float64)
        A1 = xp.zeros((self.D1, self.n_packed), dtype=np.float64)

        for k_idx in range(self.Lmax):
            k = k_idx + 1

            block0_rows = slice(self.d0[k_idx], self.d0[k_idx + 1])
            block1_rows = slice(self.d1[k_idx], self.d1[k_idx + 1])

            block0, block1 = self._permutek(A[k_idx], k)

            A0[block0_rows] = self.vec_block(block0, k)
            A1[block1_rows] = self.vec_block(block1, k + 1)

        return A0, A1

    def _transform_coeff_back(self, A0, A1):
        """
        Reconstruct full degree-wise matrices from the two representation arrays.

        :param A0: First representation array, of shape (D0, N(N+1)/2).
        :param A1: Second representation array, of shape (D1, N(N+1)/2).

        :return: List of full representation matrices for degrees 1 through Lmax.
            The degree-k matrix has shape (N*(2*k+1), N*(2*k+1)).
        """
        N = self.n_img
        A = []

        for k_idx in range(self.Lmax):
            k = k_idx + 1
            dimension = 2 * k + 1

            block0_rows = slice(self.d0[k_idx], self.d0[k_idx + 1])
            block1_rows = slice(self.d1[k_idx], self.d1[k_idx + 1])

            Ak = xp.zeros((N * dimension, N * dimension), dtype=np.float64)
            Ak[: N * k, : N * k] = self.mat_block(A0[block0_rows], k)
            Ak[N * k :, N * k :] = self.mat_block(A1[block1_rows], k + 1)

            Ak = self._permutek_back(Ak, k)
            A.append(Ak)

        return A

    def _permutek(self, Ak, k):
        """
        Separate the two coordinate groups of a degree-k matrix.

        Each image contributes 2k + 1 coordinates. Within an image, the
        k-coordinate group occupies odd indices and the k + 1 group occupies
        even indices. Reorder both axes within each image, then place the
        k-coordinate group from every image before the (k + 1)-coordinate
        group. Return the two diagonal blocks so they can be processed
        separately by ADMM. Cross-group blocks are not returned.

        :param Ak: Matrix in image-wise coordinate order, with shape
            (N * (2k + 1), N * (2k + 1)).
        :param k: Representation degree.

        :return: The k-component and (k + 1)-component blocks, with shapes
            (N*k, N*k) and (N*(k + 1), N*(k + 1)).
        """
        N = self.n_img
        dk = 2 * k + 1

        # The real degree-k basis interleaves its k- and (k+1)-coordinate
        # components. Put the odd-indexed basis coordinates first.
        coordinate_order = xp.asarray(
            np.concatenate((np.arange(1, dk, 2), np.arange(0, dk, 2)))
        )

        # Reshape into blocks (image i, coordinate i, image j, coordinate j)
        # and reorder the coordinate axes.
        blocks = Ak.reshape(N, dk, N, dk)
        blocks = blocks[:, coordinate_order, :, :]
        blocks = blocks[:, :, :, coordinate_order]
        AkP = blocks.reshape(N * dk, N * dk)

        # Group the k-coordinate component of every image first, followed by
        # the (k + 1)-coordinate component of every image.
        image_offsets = dk * np.arange(N)[:, None]
        first_component = (image_offsets + np.arange(k)).reshape(-1)
        second_component = (image_offsets + np.arange(k, dk)).reshape(-1)
        group_order = xp.asarray(np.concatenate((first_component, second_component)))

        # Apply the same permutation to rows and columns.
        AkP = AkP[group_order[:, None], group_order[None, :]]

        return AkP[: N * k, : N * k], AkP[N * k :, N * k :]

    def _permutek_back(self, Ak, k):
        """
        Restore image-wise coordinate order after ADMM block processing.

        The input is in component-grouped order: all N*k coordinates of
        the first group precede all N*(k + 1) coordinates of the second.
        `_transform_coeff_back` places the two processed blocks on the
        diagonal of this matrix, with zeros elsewhere. This method then
        restores the original image-wise coordinate order.

        :param Ak: Full matrix in component-grouped order, with shape
            (N * (2k + 1), N * (2k + 1)).
        :param k: Representation degree.

        :return: Matrix in image-wise coordinate order, with the same shape
            as Ak.
        """
        dk = 2 * k + 1

        # Undo the grouping of all k-coordinate components before all
        # (k + 1)-coordinate components.
        image_offsets = dk * np.arange(self.n_img)[:, None]
        first_component = (image_offsets + np.arange(k)).reshape(-1)
        second_component = (image_offsets + np.arange(k, dk)).reshape(-1)
        group_order = np.concatenate((first_component, second_component))
        inverse_group_order = xp.asarray(np.argsort(group_order))
        AkB = Ak[inverse_group_order[:, None], inverse_group_order[None, :]]

        # Undo the within-image ordering of the degree-k basis coordinates.
        coordinate_order = np.concatenate((np.arange(1, dk, 2), np.arange(0, dk, 2)))
        inverse_coordinate_order = xp.asarray(np.argsort(coordinate_order))

        blocks = AkB.reshape(self.n_img, dk, self.n_img, dk)
        blocks = blocks[:, inverse_coordinate_order, :, :]
        blocks = blocks[:, :, :, inverse_coordinate_order]
        return blocks.reshape(self.n_img * dk, self.n_img * dk)

    def vec_block(self, A, sz):
        """
        Pack the upper-triangular image-pair blocks of a block matrix.

        Blocks are vectorized row-wise and stored as columns, using the
        image-pair layout constructed at initialization.

        :param A: Block matrix with shape (N * sz, N * sz).
        :param sz: Row and column size of each square block.

        :return: Packed block array with shape (sz**2, N * (N + 1) // 2).
        """
        vecA = (
            (A.reshape(self.n_img, sz, self.n_img, sz).transpose(0, 2, 1, 3))
            .reshape(self.n_img**2, sz**2)
            .T
        )
        return vecA[:, self.IDX_upper]

    def mat_block(self, vecA, sz):
        """
        Reconstruct a symmetric block matrix from image-pair columns.

        Each column is reshaped row-wise into a square block and placed in
        the upper triangle. Transposes of distinct-pair blocks are placed
        in the corresponding lower-triangular positions.

        :param vecA: Block array with shape (sz**2, N * (N + 1) // 2).
        :param sz: Row and column size of each square block.

        :return: Symmetric block matrix with shape (N * sz, N * sz).
        """
        tmp = vecA.T.reshape(self.n_packed, sz, sz)
        AA = xp.zeros((self.n_img**2, sz, sz), dtype=vecA.dtype)
        AA[self.IDX_upper] = tmp
        AA[self.IDX_lower] = tmp[self.idx_offdiag].transpose(0, 2, 1)
        return (
            AA.reshape(self.n_img, self.n_img, sz, sz).transpose(0, 2, 1, 3)
        ).reshape(self.n_img * sz, self.n_img * sz)

    @staticmethod
    def _psd_projection(B):
        """
        Project real square matrices onto the positive semidefinite cone.

        Each matrix is first symmetrized. Negative eigenvalues are then
        replaced with zero before the matrix is reconstructed. Leading
        dimensions are treated as stack dimensions.

        :param B: Matrix or stack of matrices with shape (..., n, n).

        :return: Positive semidefinite projection with same shape as B.
        """
        # compute the PSD part of a symmetric matrix
        B_sym = (B + B.swapaxes(-1, -2)) / 2
        evals, evecs = xp.linalg.eigh(B_sym)
        evals = xp.maximum(evals, 0)
        return (evecs * evals[..., None, :]) @ evecs.swapaxes(-1, -2)

    @staticmethod
    def transform_block(A, k, Pk):
        """
        Permute and pack the two components of degree-k matrices.

        Each input matrix is transformed as Pk @ A @ Pk.T. The leading k-by-k
        and trailing (k + 1)-by-(k + 1) diagonal blocks are then vectorized
        row-wise. Entries in the two off-diagonal blocks are not retained.

        :param A: Batch of degree-k matrices with shape (n_blocks, 2 * k + 1, 2 * k + 1).
        :param k: Wigner representation degree.
        :param Pk: Permutation matrix with shape (2 * k + 1, 2 * k + 1).

        :return: Packed leading and trailing block arrays with shapes
            (n_blocks, k**2) and (n_blocks, (k + 1)**2), respectively.
        """
        AT = Pk @ A @ Pk.T
        A0 = AT[:, :k, :k].reshape(A.shape[0], -1)
        A1 = AT[:, k:, k:].reshape(A.shape[0], -1)
        return A0, A1

    @staticmethod
    def transform_back_block(A0, A1, k, Pk):
        """
        Reconstruct degree-k matrices from two packed components.

        Each row of A0 and A1 is interpreted as a row-wise vectorization
        of a k-by-k or (k + 1)-by-(k + 1) block. The two blocks are placed
        on the diagonal of a (2 * k + 1)-by-(2 * k + 1) matrix, with zeros in the
        off-diagonal blocks. The permutation represented by Pk is then reversed.

        :param A0: Packed leading blocks with shape (n_blocks, k**2).
        :param A1: Packed trailing blocks with shape (n_blocks, (k + 1)**2).
        :param k: Wigner representation degree.
        :param Pk: Permutation matrix with shape (2 * k + 1, 2 * k + 1).

        :return: Reconstructed degree-k matrices with shape
            (n_blocks, 2 * k + 1, 2 * k + 1).
        """
        dk = 2 * k + 1
        A = xp.zeros((A0.shape[0], dk, dk), dtype=A0.dtype)
        A[:, :k, :k] = A0.reshape(-1, k, k)
        A[:, k:, k:] = A1.reshape(-1, k + 1, k + 1)
        return Pk.T @ A @ Pk


########################
# Prepared ADMM Inputs #
########################


@dataclass
class _ADMMData:
    """
    Data constructed for the C1 and symmetric ADMM solvers.

    Array fields use the active NumPy or CuPy backend. All fields are
    populated when preprocessing returns. Solvers take ownership of the
    initial working arrays during construction.

    Objective:
        C0, C1: Coefficients packed into the two representation blocks.
        normC: Combined norm of those coefficients.

    Fejer inequalities:
        fejer: Operators, bound, and multiplier-update scale for the
            discretized Fejer constraints.

    Block layout:
        blocks: Fixed degree-block boundaries, image-pair indices, and
            representation conversion and projection operations.

    ADMM variables:
        X0, X1, Xq: Primal variables for the two blocks and quaternion terms.
        S0, S1, Sq: Corresponding slack variables.
    """

    # Objective
    C0: xp.ndarray
    C1: xp.ndarray
    normC: float | xp.ndarray

    # Fejer inequalities
    fejer: _FejerConstraints

    # Fixed representation layout and block operations
    blocks: _RepresentationBlocks

    # Initial ADMM variables
    X0: xp.ndarray
    X1: xp.ndarray
    Xq: xp.ndarray
    S0: xp.ndarray
    S1: xp.ndarray
    Sq: xp.ndarray
