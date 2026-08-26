import logging

import numpy as np
from scipy.special import factorial

from aspire.abinitio import Orient3D
from aspire.abinitio.sync_voting import _syncrotations
from aspire.numeric import xp
from aspire.operators import PolarFT
from aspire.utils import Rotation, cart2sph, complex_type
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
        pr_iters=None,
        verbose=True,
        **kwargs,
    ):
        """
        Initialize the symmetric NUG orientation estimator. All default values match those used
        for publication results, with exception of `pr_iters`, which controls how many iterations
        of proximal refinement to be used. The default of None runs the algorithm without proximal
        refinement. Set `pr_iters=4` to match t3he proximal refinement workflow in the related
        publication.

        :param src: Source containing the input projection images.
        :param symmetry: Cyclic or dihedral symmetry specification, such as 'C3' or
            'D4'. If omitted, uses the source symmetry.
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
        :param pr_iters: Number of proximal refinement iterations. Default of None
            does not perform proximal refinement. Recommended value when using is 4.
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

        # Set up proximal refinement terms
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
        return self.rotations

    #######################
    # Compute Coeffs Step #
    #######################

    def compute_coeff(self):
        """
        Compute the truncated Fourier coefficient matrices of the pairwise common-line losses.
        """
        # Build degree-wise coefficient matrices C[k] for the linear SDP objective.
        N = self.n_img
        n_theta = self.n_theta
        Lmax = self.Lmax
        T = self.T

        def fij(alpha, gamma, i, j):
            """
            Evaluate the pairwise common-line loss used to approximate NUG Fourier coefficients.

            This function samples the corresponding polar Fourier rays from images i and j,
            compares them over all candidate 1D shifts, and returns the minimum shifted
            L1 mismatch.

            :param alpha: First ZYZ Euler angle of the relative orientation, in radians.
            :param gamma: Third ZYZ Euler angle of the relative orientation, in radians.
            :param i: Index of the first image.
            :param j: Index of the second image.

            :return: Minimum shifted L1 mismatch between the induced common-line rays.
            """

            Ii_hat = self.pf_full[i]
            Ij_hat = self.pf_full[j]
            idxi = np.round((alpha - np.pi / 2) * n_theta / 2 / np.pi) % n_theta
            idxj = np.round((-gamma - np.pi / 2) * n_theta / 2 / np.pi) % n_theta

            Si = Ii_hat[int(idxi)]
            Sj = Ij_hat[int(idxj)]

            # Apply shifts
            Sj_shifted = self.shift_phases * Sj
            norms = np.linalg.norm(Si[None] - Sj_shifted, 1, axis=1)
            return norms.min()

        # Quadrature grid used to sample the pairwise loss over SO(3) Euler angles.
        alpha_grid = np.arange(2 * T, dtype=np.float64) * np.pi / T
        beta_grid = (2 * np.arange(2 * T, dtype=np.float64) + 1) * np.pi / 4 / T
        gamma_grid = np.arange(2 * T, dtype=np.float64) * np.pi / T

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
            btk = np.sum(bT[:, None, None] * self.Wd(k, beta_grid), axis=0)
            BTK.append(btk.T)

        def fijhat_k(k, F):
            """
            Approximate the degree-k Fourier coefficient block of a sampled pairwise loss.
            """
            dk = 2 * k + 1

            exp_alpha_grid = np.zeros((2 * T, dk), dtype=complex_type(np.float64))
            for m in range(-k, k + 1):
                exp_alpha_grid[:, m + k] = np.exp(1j * m * alpha_grid)

            exp_gamma_grid = np.zeros((2 * T, dk), dtype=complex_type(np.float64))
            for m in range(-k, k + 1):
                exp_gamma_grid[:, m + k] = np.exp(1j * m * gamma_grid)

            S = (exp_alpha_grid.T @ F @ exp_gamma_grid).T
            fhat = BTK[k - 1] * S / 4 / T**2
            return fhat

        # Allocate one coefficient matrix per Wigner degree;
        # each is block-indexed by image pair.
        C = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            C.append(np.zeros((N * dk, N * dk), dtype=complex_type(np.float64)))

        # Compute off-diagonal image-pair losses and insert their degree-wise coefficients.
        for i in range(N):
            for j in range(i + 1, N):
                Fij = np.zeros((2 * T, 2 * T), dtype=np.float64)
                for j1 in range(2 * T):
                    for j2 in range(2 * T):
                        Fij[j1, j2] = fij(alpha_grid[j1], gamma_grid[j2], i, j)
                for k in range(1, Lmax + 1):
                    dk = 2 * k + 1
                    C[k - 1][j * dk : (j + 1) * dk, i * dk : (i + 1) * dk] = fijhat_k(
                        k, Fij
                    )  # *dk

        # Fill the conjugate transpose blocks so each coefficient matrix is Hermitian.
        for k in range(1, Lmax + 1):
            C[k - 1] = C[k - 1] + C[k - 1].conj().T

        # Diagonal blocks encode self-pair losses and are handled separately.
        for i in range(N):
            Fii = np.zeros((2 * T, 2 * T), dtype=np.float64)
            for j1 in range(2 * T):
                for j2 in range(2 * T):
                    Fii[j1, j2] = fij(alpha_grid[j1], gamma_grid[j2], i, i)
            for k in range(1, Lmax + 1):
                dk = 2 * k + 1
                C[k - 1][i * dk : (i + 1) * dk, i * dk : (i + 1) * dk] = fijhat_k(
                    k, Fii
                )  # *dk

        # Convert complex Wigner coefficients to the real representation basis used by ADMM.
        for k in range(1, Lmax + 1):
            [T, Tinv] = self.complex2real(k)
            C[k - 1] = np.real(
                np.kron(np.eye(N, dtype=np.float64), Tinv)
                @ C[k - 1]
                @ np.kron(np.eye(N, dtype=np.float64), T)
            )

        self.C = C

    def complex2real(self, ell):
        """
        Construct the transformation matrices between complex and real degree ell representations.

        :param ell: Wigner representation degree.

        :return: Forward and inverse change-of-basis matrices.
        """
        diml = 2 * ell + 1
        Tinv = np.zeros((diml, diml), dtype=complex_type(np.float64))
        for i in range(diml):
            if i < ell:
                Tinv[i, i] = 1j / np.sqrt(2)
                Tinv[i, diml - 1 - i] = -1j * (-1) ** (i - ell) / np.sqrt(2)
            if i == ell:
                Tinv[i, i] = 1
            if i > ell:
                Tinv[i, i] = (-1) ** (i - ell) / np.sqrt(2)
                Tinv[i, diml - 1 - i] = 1 / np.sqrt(2)

        T = np.zeros((diml, diml), dtype=complex_type(np.float64))
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
            if is_asymmetric:
                raise NotImplementedError(
                    "Proximal refinement is not yet implemented for asymmetric NUG."
                )

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

        :param C: Fourier coefficient matrices of the NUG objective.
        :param verbose: Whether to log ADMM progress.

        :return: Relaxed representation matrices.
        """
        Lmax = self.Lmax
        N = self.n_img
        max_iter = self.max_iter
        rho = self.rho
        ratio = self.ratio
        factor = self.factor
        mult = self.mult
        Nstep_yI = self.Nstep_yI

        (
            C0,
            C1,
            normC,
            AEq,
            bEq,
            AEqAEqtinv,
            AI_mat_diag,
            AI_mat_offdiag,
            bI,
            Lambda,
            d0,
            d1,
            D0,
            D1,
            idx_diag,
            idx_offdiag,
            IDX_upper,
            IDX_lower,
            X0,
            X1,
            Xq,
            S0,
            S1,
            Sq,
        ) = self.ADMM_preprocessing(C)

        Ngrid = self.Ngrid
        n_pairs = N * (N - 1) // 2
        n_packed = N * (N + 1) // 2

        # In the asymmetric formulation, diagonal representation blocks are identity.
        bE0 = xp.zeros(D0, dtype=np.float64)
        bE1 = xp.zeros(D1, dtype=np.float64)

        for k in range(1, Lmax + 1):
            bE0[d0[k - 1] : d0[k]] = xp.eye(k, dtype=np.float64).reshape(-1)
            bE1[d1[k - 1] : d1[k]] = xp.eye(k + 1, dtype=np.float64).reshape(-1)

        bE0 = xp.repeat(bE0[:, None], N, axis=1)
        bE1 = xp.repeat(bE1[:, None], N, axis=1)

        def fun_AE(X0, X1, Xq):
            """
            Apply the asymmetric equality constraints.
            """
            z0 = X0[:, idx_diag]
            z1 = X1[:, idx_diag]
            zq = AEq @ xp.concatenate(
                (Xq, X0[:1, idx_offdiag], X1[:4, idx_offdiag]), axis=0
            )
            return z0, z1, zq

        def fun_AET(yE0, yE1, yEq):
            """
            Apply the adjoint of the asymmetric equality operator.
            """
            Z0 = xp.zeros((D0, n_packed), dtype=np.float64)
            Z1 = xp.zeros((D1, n_packed), dtype=np.float64)
            Z0[:, idx_diag] = yE0
            Z1[:, idx_diag] = yE1

            Zq = AEq.T @ yEq
            Z0[:1, idx_offdiag] = Zq[16:17]
            Z1[:4, idx_offdiag] = Zq[17:]
            return Z0, Z1, Zq[:16]

        def fun_AI(X0, X1):
            """
            Apply the Fejer inequality operator to off-diagonal image pairs.

            The plain formulation does not apply these inequalities to the fixed
            diagonal representation blocks.
            """
            tmp = xp.concatenate((X0[:, idx_offdiag], X1[:, idx_offdiag]), axis=0)
            return AI_mat_offdiag @ tmp

        def fun_AIT(yI):
            """
            Apply the adjoint of the asymmetric Fejer operator.
            """
            Z = xp.zeros((D0 + D1, n_packed), dtype=np.float64)
            Z[:, idx_offdiag] = AI_mat_offdiag.T @ yI
            return Z[:D0], Z[D0:]

        def update_S(C0, C1, yE0, yE1, yEq, yI, X0, X1, Xq, rho):
            """
            Update and project the asymmetric PSD slack variables.
            """
            Z0, Z1, Zq = fun_AET(yE0, yE1, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)

            S0 = C0 - Z0 - AIT_yI0 - X0 / rho
            S1 = C1 - Z1 - AIT_yI1 - X1 / rho
            Sq = -Zq - Xq / rho

            for k in range(1, Lmax + 1):
                tmp = self.mat_block(
                    S0[d0[k - 1] : d0[k]],
                    N,
                    k,
                    IDX_upper,
                    IDX_lower,
                    idx_offdiag,
                )
                tmp = self.psd_projection(tmp)
                S0[d0[k - 1] : d0[k]] = self.vec_block(tmp, N, k, IDX_upper)

                tmp = self.mat_block(
                    S1[d1[k - 1] : d1[k]],
                    N,
                    k + 1,
                    IDX_upper,
                    IDX_lower,
                    idx_offdiag,
                )
                tmp = self.psd_projection(tmp)
                S1[d1[k - 1] : d1[k]] = self.vec_block(tmp, N, k + 1, IDX_upper)

            # This is the established batched equivalent of projecting each 4x4
            # quaternion block and storing tmp.T.reshape(16) in each column.
            Sq = self.psd_projection(Sq.T.reshape(n_pairs, 4, 4))
            Sq = Sq.T.reshape(16, n_pairs)
            return S0, S1, Sq

        def update_yE(C0, C1, S0, S1, Sq, yI, X0, X1, Xq, rho):
            """
            Update the asymmetric equality multipliers.
            """
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            z0, z1, zq = fun_AE(
                -X0 / rho + C0 - S0 - AIT_yI0,
                -X1 / rho + C1 - S1 - AIT_yI1,
                -Xq / rho - Sq,
            )
            yE0 = bE0 / rho + z0
            yE1 = bE1 / rho + z1
            yEq = AEqAEqtinv @ (bEq / rho + zq)
            return yE0, yE1, yEq

        def update_yI(
            C0,
            C1,
            S0,
            S1,
            yE0,
            yE1,
            yEq,
            yI,
            X0,
            X1,
            rho,
            Lambda,
        ):
            """
            Update the nonnegative Fejer inequality multipliers.
            """
            AET_yE0, AET_yE1, _ = fun_AET(yE0, yE1, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            tmp = fun_AI(
                -X0 / rho + C0 - S0 - AIT_yI0 - AET_yE0,
                -X1 / rho + C1 - S1 - AIT_yI1 - AET_yE1,
            )
            yI = yI + bI / rho / Lambda + tmp / Lambda
            yI = xp.maximum(yI, 0)
            return yI

        def update_X(
            C0,
            C1,
            S0,
            S1,
            Sq,
            yE0,
            yE1,
            yEq,
            yI,
            X0,
            X1,
            Xq,
            rho,
        ):
            """
            Update the asymmetric primal multipliers.
            """
            Z0, Z1, Zq = fun_AET(yE0, yE1, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)

            tmp = S0 + AIT_yI0 + Z0 - C0
            X0 = X0 + mult * rho * tmp
            resX0 = xp.linalg.norm(tmp)

            tmp = S1 + AIT_yI1 + Z1 - C1
            X1 = X1 + mult * rho * tmp
            resX1 = xp.linalg.norm(tmp)

            tmp = Sq + Zq
            Xq = Xq + mult * rho * tmp
            resXq = xp.linalg.norm(tmp)

            res_X = xp.sqrt(resX0**2 + resX1**2 + resXq**2)
            return X0, X1, Xq, res_X

        def update_rho(X0, X1, Xq, res_X, rho):
            """
            Update the ADMM penalty parameter.
            """
            z0, z1, zq = fun_AE(X0, X1, Xq)
            res_eq = (
                xp.linalg.norm(bE0 - z0) / (1 + xp.linalg.norm(bE0))
                + xp.linalg.norm(bE1 - z1) / (1 + xp.linalg.norm(bE1))
                + xp.linalg.norm(bEq - zq) / (1 + xp.linalg.norm(bEq))
            )
            res_inq = xp.linalg.norm(xp.maximum(bI - fun_AI(X0, X1), 0)) / (
                1 + abs(bI) * xp.sqrt(Ngrid * n_pairs)
            )

            p_resnorm = res_eq + res_inq
            d_resnorm = res_X / (1 + normC)
            if d_resnorm > ratio * p_resnorm:
                rho = rho * factor
            if d_resnorm < ratio * p_resnorm:
                rho = rho / factor
            return rho, p_resnorm, d_resnorm

        def print_updates(verbose):
            """
            Log diagnostics for the current asymmetric ADMM iterate.
            """
            if not verbose:
                return

            obj_p = (
                xp.vdot(C0[:, idx_diag], X0[:, idx_diag])
                + xp.vdot(C1[:, idx_diag], X1[:, idx_diag])
                + 2 * xp.vdot(C0[:, idx_offdiag], X0[:, idx_offdiag])
                + 2 * xp.vdot(C1[:, idx_offdiag], X1[:, idx_offdiag])
            )
            obj_d = (
                xp.vdot(yE0, bE0)
                + xp.vdot(yE1, bE1)
                + 2 * xp.vdot(yEq, bEq)
                + 2 * bI * xp.sum(yI)
            )

            z0, z1, zq = fun_AE(X0, X1, Xq)
            res_eq = (
                xp.linalg.norm(bE0 - z0) / (1 + xp.linalg.norm(bE0))
                + xp.linalg.norm(bE1 - z1) / (1 + xp.linalg.norm(bE1))
                + xp.linalg.norm(bEq - zq) / (1 + xp.linalg.norm(bEq))
            )
            res_inq = xp.linalg.norm(xp.maximum(bI - fun_AI(X0, X1), 0)) / (
                1 + abs(bI) * xp.sqrt(Ngrid * n_pairs)
            )

            res_psdX = 0
            for k in range(1, Lmax + 1):
                tmp = self.mat_block(
                    X0[d0[k - 1] : d0[k]],
                    N,
                    k,
                    IDX_upper,
                    IDX_lower,
                    idx_offdiag,
                )
                res_psdX += xp.linalg.norm(self.psd_projection(-tmp))

                tmp = self.mat_block(
                    X1[d1[k - 1] : d1[k]],
                    N,
                    k + 1,
                    IDX_upper,
                    IDX_lower,
                    idx_offdiag,
                )
                res_psdX += xp.linalg.norm(self.psd_projection(-tmp))

            res_psdX /= 1 + xp.linalg.norm(X0) + xp.linalg.norm(X1)

            Xq_blocks = Xq.T.reshape(n_pairs, 4, 4)
            res_psdQ = xp.linalg.norm(
                self.psd_projection(-Xq_blocks), axis=(-2, -1)
            ).sum()
            res_psdQ /= 1 + xp.linalg.norm(Xq)

            normS = xp.sqrt(
                xp.linalg.norm(S0) ** 2
                + xp.linalg.norm(S1) ** 2
                + xp.linalg.norm(Sq) ** 2
            )
            normX = xp.sqrt(
                xp.linalg.norm(X0) ** 2
                + xp.linalg.norm(X1) ** 2
                + xp.linalg.norm(Xq) ** 2
            )
            p_res = res_eq + res_inq + res_psdX + res_psdQ
            d_res = res_X / (1 + normC)

            logger.info(
                "Iter %i" % t
                + ": p_res=%1.5f" % p_res
                + ", d_res=%1.5f" % d_res
                + ", obj_primal=%1.2f" % obj_p
                + ", obj_dual=%1.2f" % obj_d
                + ", duality gap=%1.2f" % (obj_p - obj_d)
                + "\n        eq_res=%1.5f" % res_eq
                + ", inq_res=%1.5f" % res_inq
                + ", psd_res=%1.5f" % (res_psdX + res_psdQ)
                + ", |S|=%1.2f" % normS
                + ", |X|=%1.2f" % normX
            )

        yI = xp.zeros((Ngrid, n_pairs), dtype=np.float64)
        yE0 = xp.zeros(bE0.shape, dtype=np.float64)
        yE1 = xp.zeros(bE1.shape, dtype=np.float64)
        yEq = xp.zeros(bEq.shape, dtype=np.float64)

        IDX = np.arange(3)
        for t in range(max_iter):
            # np.random.shuffle(IDX)
            for idx in IDX:
                if idx == 0:
                    S0, S1, Sq = update_S(C0, C1, yE0, yE1, yEq, yI, X0, X1, Xq, rho)
                elif idx == 1:
                    yE0, yE1, yEq = update_yE(C0, C1, S0, S1, Sq, yI, X0, X1, Xq, rho)
                else:
                    for _ in range(Nstep_yI):
                        yI = update_yI(
                            C0,
                            C1,
                            S0,
                            S1,
                            yE0,
                            yE1,
                            yEq,
                            yI,
                            X0,
                            X1,
                            rho,
                            Lambda,
                        )

            X0, X1, Xq, res_X = update_X(
                C0, C1, S0, S1, Sq, yE0, yE1, yEq, yI, X0, X1, Xq, rho
            )
            if t % 100 == 0:
                print_updates(verbose)
            rho, p_resnorm, d_resnorm = update_rho(X0, X1, Xq, res_X, rho)

        X_admm = self.transform_coeff_back(X0, X1, IDX_upper, IDX_lower, idx_offdiag)
        for k in range(Lmax):
            X_admm[k] = xp.asnumpy(X_admm[k])
        return X_admm

    def admm_sym_J(self, C, verbose):
        """
        Solve the symmetry-constrained NUG semidefinite relaxation using ADMM.

        :param C: Fourier coefficient matrices of the NUG objective.
        :param verbose: Whether to log ADMM progress.

        :return: Relaxed representation matrices.
        """
        Lmax = self.Lmax
        N = self.n_img
        max_iter = self.max_iter
        rho = self.rho
        ratio = self.ratio
        factor = self.factor
        mult = self.mult
        Nstep_yI = self.Nstep_yI

        # Solve the symmetry-constrained SDP relaxation in the real, packed
        # representation basis prepared by ADMM_preprocessing.
        (
            C0,
            C1,
            normC,
            AEq,
            bEq,
            AEqAEqtinv,
            AI_mat_diag,
            AI_mat_offdiag,
            bI,
            Lambda,
            d0,
            d1,
            D0,
            D1,
            idx_diag,
            idx_offdiag,
            IDX_upper,
            IDX_lower,
            X0,
            X1,
            Xq,
            S0,
            S1,
            Sq,
        ) = self.ADMM_preprocessing(C)

        n_pairs = N * (N - 1) // 2
        Ngrid = self.Ngrid
        rank_Ak, _ = self.compute_rank()
        logger.info(f"Rank of Ak: {rank_Ak}")

        # Build per-degree equality operators coupling the k and k+1 invariant
        # blocks. These encode the representation constraints imposed by symmetry.
        AE = []
        AEAETinv = []
        for k in range(1, Lmax + 1):
            s0 = k**2
            s1 = (k + 1) ** 2
            AEk = xp.zeros((1 + s0 + s1, 2 * (s0 + s1)), dtype=np.float64)
            AEk[0, :s0] = xp.eye(k, dtype=np.float64).T.reshape(-1)
            AEk[0, s0 : s0 + s1] = xp.eye(k + 1, dtype=np.float64).T.reshape(-1)
            for count in range(1, 1 + s0):
                AEk[count, count - 1] = 1
                AEk[count, count - 1 + s0 + s1] = 1
            for count in range(1 + s0, 1 + s0 + s1):
                AEk[count, count - 1] = 1
                AEk[count, count - 1 + s1 + s0] = 1
            AE.append(AEk)
            AEAETinv.append(xp.linalg.pinv(AEk @ AEk.T))

        # Right-hand sides for the equality constraints: symmetry projector ranks
        # and identity constraints on diagonal representation blocks.
        bE = xp.zeros((Lmax + D0 + D1), dtype=np.float64)
        for k in range(Lmax):
            bE[k + d0[k] + d1[k] :] = rank_Ak[k]
            bE[k + 1 + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k]] = xp.eye(
                k + 1
            ).T.reshape(-1)
            bE[k + 1 + d0[k + 1] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]] = xp.eye(
                k + 2
            ).T.reshape(-1)
        bE = xp.repeat(bE[:, None], N, axis=1)

        # Degree-wise permutations used to move between full Wigner blocks and the
        # two invariant block variables used by the relaxation.
        P = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            Pk = xp.eye(dk, dtype=np.float64)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
            P.append(Pk)

        def fun_AE(X0, X1, Xd0, Xd1, Xq):
            """
            Apply the equality constraint operator to the current primal variables.

            z contains the per-image, per-degree equality constraints that couple
            diagonal X0/X1 blocks to auxiliary diagonal variables Xd0/Xd1.

            zq contains the off-diagonal quaternion constraints built from Xq and
            the lowest-degree off-diagonal representation blocks.
            """
            z = xp.zeros((Lmax + D0 + D1, N), dtype=np.float64)
            for k in range(Lmax):
                z[k + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]] = AE[
                    k
                ] @ xp.concatenate(
                    (
                        X0[d0[k] : d0[k + 1], idx_diag],
                        X1[d1[k] : d1[k + 1], idx_diag],
                        Xd0[d0[k] : d0[k + 1]],
                        Xd1[d1[k] : d1[k + 1]],
                    ),
                    axis=0,
                )
            zq = AEq @ xp.concatenate(
                (Xq, X0[:1, idx_offdiag], X1[:4, idx_offdiag]), axis=0
            )
            return z, zq

        def fun_AET(yE, yEq):
            """
            Apply the adjoint of the equality constraint operator.

            yE contains multipliers for the per-image, per-degree equality
            constraints produced as z by fun_AE.

            yEq contains multipliers for the off-diagonal quaternion constraints
            produced as zq by fun_AE.

            Returns the contribution of these equality multipliers back into each
            packed primal variable block: X0, X1, Xd0, Xd1, and Xq.
            """
            Z0 = xp.zeros((D0, N * (N + 1) // 2), dtype=np.float64)
            Z1 = xp.zeros((D1, N * (N + 1) // 2), dtype=np.float64)
            Zd0 = xp.zeros((D0, N), dtype=np.float64)
            Zd1 = xp.zeros((D1, N), dtype=np.float64)
            for k in range(Lmax):
                s0 = (k + 1) ** 2
                s1 = (k + 2) ** 2
                Ztmp = AE[k].T @ yE[k + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]]
                Z0[d0[k] : d0[k + 1], idx_diag] = Ztmp[:s0]
                Z1[d1[k] : d1[k + 1], idx_diag] = Ztmp[s0 : s0 + s1]
                Zd0[d0[k] : d0[k + 1]] = Ztmp[s0 + s1 : 2 * s0 + s1]
                Zd1[d1[k] : d1[k + 1]] = Ztmp[2 * s0 + s1 : 2 * s0 + 2 * s1]
            Zq = AEq.T @ yEq
            Z0[:1, idx_offdiag] = Zq[16:17]
            Z1[:4, idx_offdiag] = Zq[17:]
            return Z0, Z1, Zd0, Zd1, Zq[:16]

        def fun_AI(X0, X1):
            """
            Evaluate the Fejer inequality operator on the packed representation blocks.

            Each output column corresponds to one packed image pair. Each row
            corresponds to one sampled SO(3) grid point.

            The result is compared against bI to enforce the discretized
            nonnegativity constraints of the truncated SDP relaxation.
            """
            z = xp.zeros((Ngrid, N * (N + 1) // 2), dtype=np.float64)
            tmp = xp.concatenate((X0, X1), axis=0)
            z[:, idx_diag] = AI_mat_diag @ tmp[:, idx_diag]
            z[:, idx_offdiag] = AI_mat_offdiag @ tmp[:, idx_offdiag]
            return z

        def fun_AIT(yI):
            """
            Apply the adjoint of the Fejer inequality operator.
            """
            Z = xp.zeros((D0 + D1, N * (N + 1) // 2), dtype=np.float64)
            Z[:, idx_diag] = AI_mat_diag.T @ yI[:, idx_diag]
            Z[:, idx_offdiag] = AI_mat_offdiag.T @ yI[:, idx_offdiag]
            return Z[:D0, :], Z[D0:, :]

        def update_S(C0, C1, yE, yEq, yI, X0, X1, Xd0, Xd1, Xq, rho, Lmax, N):
            """
            Update PSD slack variables for the current ADMM iterate.

            The slack variables S0/S1/Sd0/Sd1/Sq carry the semidefinite constraints.
            This step forms the unconstrained slack minimizers and projects each
            matrix block onto the positive semidefinite cone.
            """
            # Equality multipliers mapped back to each primal variable block.
            Z0, Z1, Zd0, Zd1, Zq = fun_AET(yE, yEq)

            # Inequality multipliers mapped back to the X0/X1 variable blocks.
            AIT_yI0, AIT_yI1 = fun_AIT(yI)

            # Unconstrained slack updates before PSD projection.
            S0 = C0 - Z0 - AIT_yI0 - X0 / rho
            S1 = C1 - Z1 - AIT_yI1 - X1 / rho
            Sd0 = -Zd0 - Xd0 / rho
            Sd1 = -Zd1 - Xd1 / rho
            Sq = -Zq - Xq / rho

            # Project packed X0/X1 blocks degree by degree.
            for k in range(1, Lmax + 1):
                tmp = self.mat_block(
                    S0[d0[k - 1] : d0[k], :], N, k, IDX_upper, IDX_lower, idx_offdiag
                )
                tmp = self.psd_projection(tmp)
                S0[d0[k - 1] : d0[k], :] = self.vec_block(tmp, N, k, IDX_upper)

                tmp = self.mat_block(
                    S1[d1[k - 1] : d1[k], :],
                    N,
                    k + 1,
                    IDX_upper,
                    IDX_lower,
                    idx_offdiag,
                )
                tmp = self.psd_projection(tmp)
                S1[d1[k - 1] : d1[k], :] = self.vec_block(tmp, N, k + 1, IDX_upper)

            # Project the diagonal coupling blocks.
            Sd0 = Sd0.T
            Sd1 = Sd1.T
            for k in range(1, Lmax + 1):
                tmp = self.transform_back_block(
                    Sd0[:, d0[k - 1] : d0[k]],
                    Sd1[:, d1[k - 1] : d1[k]],
                    k,
                    P[k - 1],
                )
                tmp = self.psd_projection(tmp)
                Sd0[:, d0[k - 1] : d0[k]], Sd1[:, d1[k - 1] : d1[k]] = (
                    self.transform_block(tmp, k, P[k - 1])
                )
            Sd0 = Sd0.T
            Sd1 = Sd1.T

            # Project each 4x4 quaternion slack block for off-diagonal image pairs.
            Sq = self.psd_projection(Sq.T.reshape(n_pairs, 4, 4))
            Sq = Sq.T.reshape(-1, n_pairs)
            return S0, S1, Sd0, Sd1, Sq

        def update_yE(C0, C1, X0, X1, Xd0, Xd1, Xq, S0, S1, Sd0, Sd1, Sq, yI, rho):
            """
            Update equality multipliers for the ADMM iterate.

            yE enforces the per-degree symmetry/diagonal constraints.
            yEq enforces the off-diagonal quaternion constraints.
            """
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            z, zq = fun_AE(
                -X0 / rho + C0 - S0 - AIT_yI0,
                -X1 / rho + C1 - S1 - AIT_yI1,
                -Xd0 / rho - Sd0,
                -Xd1 / rho - Sd1,
                -Xq / rho - Sq,
            )
            yE = xp.zeros((Lmax + D0 + D1, N), dtype=np.float64)
            for k in range(Lmax):
                yE[k + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]] = AEAETinv[k] @ (
                    bE[k + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]] / rho
                    + z[k + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]]
                )
            yEq = AEqAEqtinv @ (bEq / rho + zq)
            return yE, yEq

        def update_yI(C0, C1, X0, X1, S0, S1, yE, yEq, yI, rho, Lambda):
            """
            Update nonnegative multipliers for the Fejer inequality constraints.

            yI corresponds to the discretized constraints fun_AI(X0, X1) >= bI,
            one multiplier per SO(3) grid point and packed image-pair column.
            """
            Z0, Z1, _, _, _ = fun_AET(yE, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            tmp = fun_AI(
                -X0 / rho + C0 - S0 - Z0 - AIT_yI0, -X1 / rho + C1 - S1 - Z1 - AIT_yI1
            )
            yI = yI + bI / rho / Lambda + tmp / Lambda
            yI = xp.maximum(yI, 0)
            return yI

        def update_X(
            C0, C1, X0, X1, Xd0, Xd1, Xq, yE, yEq, yI, S0, S1, Sd0, Sd1, Sq, rho
        ):
            """
            Update the primal ADMM multipliers and compute their combined residual.

            Each tmp below is the current residual for one block family.
            The multiplier is advanced by mult * rho * tmp.
            """
            Z0, Z1, Zd0, Zd1, Zq = fun_AET(yE, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            tmp = S0 + Z0 + AIT_yI0 - C0
            X0 = X0 + mult * rho * tmp
            resX0 = xp.linalg.norm(tmp)
            tmp = S1 + Z1 + AIT_yI1 - C1
            X1 = X1 + mult * rho * tmp
            resX1 = xp.linalg.norm(tmp)
            tmp = Sd0 + Zd0
            Xd0 = Xd0 + mult * rho * tmp
            resXd0 = xp.linalg.norm(tmp)
            tmp = Sd1 + Zd1
            Xd1 = Xd1 + mult * rho * tmp
            resXd1 = xp.linalg.norm(tmp)
            tmp = Sq + Zq
            Xq = Xq + mult * rho * tmp
            resXq = xp.linalg.norm(tmp)
            return (
                X0,
                X1,
                Xd0,
                Xd1,
                Xq,
                xp.sqrt(resX0**2 + resX1**2 + resXd0**2 + resXd1**2 + resXq**2),
            )

        def update_rho(X0, X1, Xd0, Xd1, Xq, bE, bEq, bI, res_X, rho, factor, normC):
            """
            Update the ADMM penalty parameter rho.
            """
            z, zq = fun_AE(X0, X1, Xd0, Xd1, Xq)
            res_eq = xp.linalg.norm(z - bE) / (1 + xp.linalg.norm(bE)) + xp.linalg.norm(
                zq - bEq
            ) / (1 + xp.linalg.norm(bEq))
            res_inq = xp.linalg.norm(xp.maximum(bI - fun_AI(X0, X1), 0)) / (
                1 + abs(bI) * xp.sqrt(Ngrid * N**2)
            )
            p_resnorm = res_eq + res_inq
            d_resnorm = res_X / (1 + normC)
            if d_resnorm > ratio * p_resnorm:
                rho = rho * factor
            if d_resnorm < ratio * p_resnorm:
                rho = rho / factor
            return rho, p_resnorm, d_resnorm

        def print_updates(verbose):
            if verbose:
                obj_p = (
                    xp.vdot(C0[:, idx_diag], X0[:, idx_diag])
                    + xp.vdot(C1[:, idx_diag], X1[:, idx_diag])
                    + 2 * xp.vdot(C0[:, idx_offdiag], X0[:, idx_offdiag])
                    + 2 * xp.vdot(C1[:, idx_offdiag], X1[:, idx_offdiag])
                )
                obj_d = (
                    xp.vdot(yE, bE)
                    + 2 * xp.vdot(yEq, bEq)
                    + xp.vdot(yI[:, idx_diag], bI * xp.ones((Ngrid, N)))
                    + 2
                    * xp.vdot(
                        yI[:, idx_offdiag], bI * xp.ones((Ngrid, N * (N - 1) // 2))
                    )
                )

                z, zq = fun_AE(X0, X1, Xd0, Xd1, Xq)
                res_eq = xp.linalg.norm(z - bE) / (
                    1 + xp.linalg.norm(bE)
                ) + xp.linalg.norm(zq - bEq) / (1 + xp.linalg.norm(bEq))
                res_inq = xp.linalg.norm(xp.maximum(bI - fun_AI(X0, X1), 0)) / (
                    1 + abs(bI) * xp.sqrt(Ngrid * N * (N + 1) / 2)
                )
                res_psdX = 0
                for k in range(1, Lmax + 1):
                    tmp = self.mat_block(
                        X0[d0[k - 1] : d0[k], :],
                        N,
                        k,
                        IDX_upper,
                        IDX_lower,
                        idx_offdiag,
                    )
                    res_psdX += xp.linalg.norm(self.psd_projection(-tmp))
                    tmp = self.mat_block(
                        X1[d1[k - 1] : d1[k], :],
                        N,
                        k + 1,
                        IDX_upper,
                        IDX_lower,
                        idx_offdiag,
                    )
                    res_psdX += xp.linalg.norm(self.psd_projection(-tmp))
                res_psdX = res_psdX / (1 + xp.linalg.norm(X0) + xp.linalg.norm(X1))
                res_psdD = 0
                for k in range(1, Lmax + 1):
                    tmp = self.transform_back_block(
                        Xd0[d0[k - 1] : d0[k]].T,
                        Xd1[d1[k - 1] : d1[k]].T,
                        k,
                        P[k - 1],
                    )
                    res_psdD += xp.linalg.norm(
                        self.psd_projection(-tmp), axis=(-2, -1)
                    ).sum()
                res_psdD = res_psdD / (1 + xp.linalg.norm(Xd0) + xp.linalg.norm(Xd1))
                res_psdQ = 0
                for count in range(N * (N - 1) // 2):
                    tmp = Xq[:, count].reshape(4, 4).T
                    res_psdQ += xp.linalg.norm(self.psd_projection(-tmp))
                res_psdQ = res_psdQ / (1 + xp.linalg.norm(Xq))

                normS = xp.sqrt(
                    xp.linalg.norm(S0) ** 2
                    + xp.linalg.norm(S1) ** 2
                    + xp.linalg.norm(Sd0) ** 2
                    + xp.linalg.norm(Sd1) ** 2
                    + xp.linalg.norm(Sq) ** 2
                )
                normX = xp.sqrt(
                    xp.linalg.norm(X0) ** 2
                    + xp.linalg.norm(X1) ** 2
                    + xp.linalg.norm(Xd0) ** 2
                    + xp.linalg.norm(Xd1) ** 2
                    + xp.linalg.norm(Xq) ** 2
                )
                p_res = res_eq + res_inq + res_psdX + res_psdD + res_psdQ
                d_res = res_X / (1 + normC)
                logger.info(
                    "Iter %i" % t
                    + ": p_res=%1.5f" % p_res
                    + ", d_res=%1.5f" % d_res
                    + ", obj_primal=%1.2f" % obj_p
                    + ", obj_dual=%1.2f" % obj_d
                    + ", duality gap=%1.2f" % (obj_p - obj_d)
                    + "\n        eq_res=%1.5f" % res_eq
                    + ", inq_res=%1.5f" % res_inq
                    + ", psd_res=%1.5f" % (res_psdX + res_psdD + res_psdQ)
                    + ", |S|=%1.2f" % normS
                    + ", |X|=%1.2f" % normX
                )

        Xd0 = xp.zeros((D0, N), dtype=xp.float64)
        Xd1 = xp.zeros((D1, N), dtype=xp.float64)
        Sd0 = xp.zeros(Xd0.shape, dtype=xp.float64)
        Sd1 = xp.zeros(Xd1.shape, dtype=xp.float64)
        yI = xp.zeros((Ngrid, N * (N + 1) // 2), dtype=xp.float64)
        yE = xp.zeros(bE.shape, dtype=xp.float64)
        yEq = xp.zeros(bEq.shape, dtype=xp.float64)

        # Run ADMM iterations, randomly ordering the block updates before each primal
        # multiplier update and penalty adjustment.
        IDX = np.arange(3)
        for t in range(max_iter):
            np.random.shuffle(IDX)
            for idx in IDX:
                if idx == 0:
                    S0, S1, Sd0, Sd1, Sq = update_S(
                        C0, C1, yE, yEq, yI, X0, X1, Xd0, Xd1, Xq, rho, Lmax, N
                    )
                if idx == 1:
                    yE, yEq = update_yE(
                        C0, C1, X0, X1, Xd0, Xd1, Xq, S0, S1, Sd0, Sd1, Sq, yI, rho
                    )
                if idx == 2:
                    for _ in range(Nstep_yI):
                        yI = update_yI(C0, C1, X0, X1, S0, S1, yE, yEq, yI, rho, Lambda)
            X0, X1, Xd0, Xd1, Xq, res_X = update_X(
                C0, C1, X0, X1, Xd0, Xd1, Xq, yE, yEq, yI, S0, S1, Sd0, Sd1, Sq, rho
            )
            if t % 100 == 0:
                print_updates(verbose)
            rho, p_resnorm, d_resnorm = update_rho(
                X0, X1, Xd0, Xd1, Xq, bE, bEq, bI, res_X, rho, factor, normC
            )

        # Convert the optimized packed block variables back to full degree-wise
        # representation matrices for Euler-angle recovery.
        X_admm = self.transform_coeff_back(X0, X1, IDX_upper, IDX_lower, idx_offdiag)
        for k in range(Lmax):
            X_admm[k] = xp.asnumpy(X_admm[k])
        return X_admm

    def ADMM_preprocessing(self, C):
        """
        Construct the transformed coefficients, constraints, indices, and initial variables used by ADMM.

        :param C: Fourier coefficient matrices of the NUG objective.

        :return: Quantities required by the ADMM solver.
        """
        Lmax = self.Lmax
        N = self.n_img

        # Indices into the packed upper-triangular image-pair columns:
        # diagonal self-pairs (i, i) and off-diagonal pairs (i, j), i < j.
        iu, ju = np.triu_indices(N)
        idx_diag = xp.asarray(np.flatnonzero(iu == ju))
        idx_offdiag = xp.asarray(np.flatnonzero(iu != ju))

        # Linear indices into the full N x N image-pair grid. IDX_upper selects
        # upper-triangular pairs (i, j), i <= j; IDX_lower selects the matching
        # lower-triangular pairs (j, i), i < j, in the same order as idx_offdiag.
        iu_off, ju_off = np.triu_indices(N, k=1)
        IDX_upper = xp.asarray(iu * N + ju)
        IDX_lower = xp.asarray(ju_off * N + iu_off)

        # normalize C matrix
        Cnorm = 0
        Xnorm = 0
        for k in range(Lmax):
            dk = 2 * k + 1
            Cnorm += np.linalg.norm(C[k]) ** 2
            Xnorm += dk * N**2
        Cnorm = np.sqrt(Cnorm)
        Xnorm = np.sqrt(Xnorm)
        for k in range(Lmax):
            C[k] = xp.asarray(Xnorm / Cnorm * C[k])
        C0, C1 = self.transform_coeff(C, IDX_upper)
        normC = xp.sqrt(xp.linalg.norm(C0) ** 2 + xp.linalg.norm(C1) ** 2)
        del C

        # compute the block sizes for X
        d0 = [0]
        d1 = [0]
        for k in range(1, Lmax + 1):
            d0.append(d0[-1] + k**2)
            d1.append(d1[-1] + (k + 1) ** 2)
        D0 = d0[-1]
        D1 = d1[-1]

        # AE and bE for quaternion constraints
        AEq = xp.asarray(self.construct_AEq())
        AEqAEqtinv = xp.linalg.pinv(AEq @ AEq.T)

        bEq = xp.zeros(17, dtype=np.float64)
        bEq[:16] = xp.eye(4, dtype=np.float64).reshape(-1) / 4
        bEq[-1] = 1
        bEq = xp.repeat(bEq[:, None], N * (N - 1) // 2, axis=1)

        # Compute AI and bI
        W0, W1, Ngrid = self.compute_fejer_weights()

        # AI_mat_offdiag computation has been vectorized, but originally
        # had the note by block1: this needs double checking (Ruiyi)
        AI_mat_offdiag = np.zeros((Ngrid, D0 + D1), dtype=np.float64)
        for k in range(1, Lmax + 1):
            scale = (Lmax - k + 2) * (Lmax - k + 1) * (k + 0.5)

            block0 = scale * W0[k - 1].transpose(0, 2, 1).reshape(Ngrid, -1)
            block1 = scale * W1[k - 1].transpose(0, 2, 1).reshape(Ngrid, -1)

            AI_mat_offdiag[:, d0[k - 1] : d0[k]] = block0
            AI_mat_offdiag[:, d0[-1] + d1[k - 1] : d0[-1] + d1[k]] = block1

        # AI_mat_diag computation has been vectorized, but originally
        # had the note by block1: this needs double checking (Ruiyi)
        AI_mat_diag = np.zeros((Ngrid, D0 + D1), dtype=np.float64)
        for k in range(1, Lmax + 1):
            scale = (Lmax - k + 2) * (Lmax - k + 1) * (k + 0.5)

            W0_sym = 0.5 * (W0[k - 1] + W0[k - 1].transpose(0, 2, 1))
            W1_sym = 0.5 * (W1[k - 1] + W1[k - 1].transpose(0, 2, 1))

            block0 = scale * W0_sym.transpose(0, 2, 1).reshape(Ngrid, -1)
            block1 = scale * W1_sym.transpose(0, 2, 1).reshape(Ngrid, -1)

            AI_mat_diag[:, d0[k - 1] : d0[k]] = block0
            AI_mat_diag[:, d0[-1] + d1[k - 1] : d0[-1] + d1[k]] = block1

        AI_mat_diag = xp.asarray(AI_mat_diag)
        AI_mat_offdiag = xp.asarray(AI_mat_offdiag)
        bI = -(Lmax + 2) * (Lmax + 1) / 2

        # largest eigenvalue for AIAIT
        Lambda = self.largest_eigenvalue(AI_mat_offdiag, Ngrid, N)

        # initialization
        II = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            II.append(xp.eye(N * dk, dtype=np.float64))
        I0, I1 = self.transform_coeff(II, IDX_upper)
        X0 = xp.zeros((D0, N * (N + 1) // 2), dtype=np.float64)
        X1 = xp.zeros((D1, N * (N + 1) // 2), dtype=np.float64)
        Xq = xp.zeros((16, N * (N - 1) // 2), dtype=np.float64)
        S0 = xp.copy(I0)
        S1 = xp.copy(I1)
        Sq = xp.zeros(Xq.shape, dtype=np.float64)

        self.Ngrid = Ngrid

        return (
            C0,
            C1,
            normC,
            AEq,
            bEq,
            AEqAEqtinv,
            AI_mat_diag,
            AI_mat_offdiag,
            bI,
            Lambda,
            d0,
            d1,
            D0,
            D1,
            idx_diag,
            idx_offdiag,
            IDX_upper,
            IDX_lower,
            X0,
            X1,
            Xq,
            S0,
            S1,
            Sq,
        )

    def compute_fejer_weights(self):
        """
        Evaluate the real Wigner representation blocks used by the discretized Fejer inequality constraints.

        :return: Two sets of block weights and the number of SO(3) grid points.
        """
        SO3_grid = self.discretize_SO3()
        Ngrid = SO3_grid.shape[0]
        start = 1

        TT = []
        TTI = []
        for ell in range(start, self.Lmax + 1):
            T, Tinv = self.complex2real(ell)
            TT.append(T)
            TTI.append(Tinv)

        def permutek_block(Ak, k):
            dk = 2 * k + 1
            Pk = np.eye(dk, dtype=np.float64)
            for m in range(k):
                for ell in range(k - m):
                    Pk[(m + 2 * ell, m + 2 * ell + 1), :] = Pk[
                        (m + 2 * ell + 1, m + 2 * ell), :
                    ]
            AkP = Pk @ Ak @ Pk.T
            return AkP[..., :k, :k], AkP[..., k:, k:]

        W0 = []
        W1 = []
        for k in range(start, self.Lmax + 1):
            W0k = np.zeros((Ngrid, k, k), dtype=np.float64)
            W1k = np.zeros((Ngrid, k + 1, k + 1), dtype=np.float64)

            TkT = TT[k - start].T
            TinvkT = TTI[k - start].T

            w = np.real(TkT @ self.WD(k, SO3_grid).conj() @ TinvkT)
            W0k, W1k = permutek_block(w, k)

            W0.append(W0k)
            W1.append(W1k)
        return W0, W1, Ngrid

    def discretize_SO3(self):
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
        N = self.n_img
        C = self.C

        def Ak(J, Euler):
            """
            Average the degree-J representation over the symmetry
            group to form the symmetry projector Ak.
            """
            order = Euler.shape[0]
            A = self.WD(J, Euler).sum(axis=0)
            return np.round(A / order, 10)

        def rel_change(A, B, eps=1e-12):
            """
            Measure relative change across all degree-wise relaxed matrices.
            """
            num = 0.0
            den = 0.0
            for k in range(len(A)):
                num += np.linalg.norm(A[k] - B[k]) ** 2
                den += np.linalg.norm(B[k]) ** 2
            return np.sqrt(num) / max(np.sqrt(den), eps)

        # The symmetry projector ranks give the expected low-rank structure
        # for each degree of a valid orbit solution.
        rank_Ak = np.zeros(self.Lmax, dtype=np.float64)

        # Keep an unmodified copy of the original objective coefficients; each
        # refinement step adds a temporary proximal term to these.
        C_base = [None] * self.Lmax
        for k in range(self.Lmax):
            C_base[k] = xp.asnumpy(C[k]).copy()
            rank_Ak[k] = np.linalg.matrix_rank(
                Ak(k + 1, self.sym_euler), tol=1e-6, hermitian=True
            )

        def low_rank_proj(X, r_step):
            """
            Project diagonal image blocks toward the rank expected from symmetry.
            """
            Xproj = []
            for k in range(self.Lmax):
                dk = 2 * k + 3
                rk = min(int(rank_Ak[k] * 2) + r_step, dk)
                tmp = np.copy(X[k])
                for i in range(N):
                    u, s, v = np.linalg.svd(
                        tmp[i * dk : (i + 1) * dk, i * dk : (i + 1) * dk]
                    )
                    tmp[i * dk : (i + 1) * dk, i * dk : (i + 1) * dk] = (
                        u[:, :rk] @ np.diag(s[:rk]) @ v[:rk]
                    )
                Xproj.append(tmp)
            return Xproj

        # CC stores the modified objective coefficients for the next proximal ADMM solve.
        CC = [None] * self.Lmax
        current = [np.copy(Xk) for Xk in X_admm]

        # Proximal refinement step.
        # Repeatedly bias the SDP objective toward the low-rank projection and resolve.
        for step in range(self.pr_iters):
            X_proj = low_rank_proj(current, r[step])

            for k in range(self.Lmax):
                CC[k] = (
                    C_base[k]
                    - Penalty[step] * weight[k] * (X_proj[k] + X_proj[k].T) / 2
                )

            X_next = self.admm_sym_J(CC, verbose=False)

            if self.verbose:
                logger.info(
                    "Proximal refine step %d/%d: relative update %.3e",
                    step + 1,
                    self.pr_iters,
                    rel_change(X_next, current),
                )

            # Use the refined SDP solution as the starting point for the next projection step
            current = [np.copy(Xk) for Xk in X_next]

        return current

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

    def euler_est_Cm(self, X1, XS):
        """
        Recover Euler angles for cyclic symmetry.

        :param X1: Relaxed degree-one representation matrix.
        :param XS: Relaxed representation matrix at the symmetry order.

        :return: Estimated rotation matrices.
        """
        S = self.n_sym
        N = self.n_img

        # Convert the relaxed NUG matrices from the real representation used by
        # the SDP solver to the complex Wigner-D basis used in the recovery formulas.
        X1 = self._real_to_complex_representation(X1, degree=1)
        XS = self._real_to_complex_representation(XS, degree=S)

        def find_beta(X1):
            B1 = np.zeros((N, N), dtype=np.float64)
            B2 = np.zeros((N, N), dtype=np.float64)

            # From equation (36) in the paper, we have
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

        def find_alpha(X1):
            ZZbar = np.zeros((N, N), dtype=complex_type(np.float64))
            ZZ = np.zeros((N, N), dtype=complex_type(np.float64))

            # Equation (36) also gives the pairwise phase measurements
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

        alpha_est = find_alpha(X1)
        beta_est = find_beta(X1)

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
        N = self.n_img

        X2 = self._real_to_complex_representation(X_est[1], degree=2)
        XS = self._real_to_complex_representation(X_est[S - 1], degree=S)

        def find_alpha_beta(X2):
            B1 = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    Xij = X2[5 * i : 5 * (i + 1), 5 * j : 5 * (j + 1)]
                    B1[i, j] = np.real((Xij[2, 2] - 2 * abs(Xij[0, 0]) + 0.5) / 3 * 2)
            e1, v1 = np.linalg.eigh(B1)
            idx = np.argmax(e1)
            b1 = v1[:, idx] * np.sqrt(e1[idx]) * np.sign(v1[0, idx])
            beta_est = np.arccos(np.clip(np.sqrt(b1), -1, 1)) % np.pi

            Aminus = np.zeros((N, N), dtype=complex_type(np.float64))
            Aplus = np.zeros((N, N), dtype=complex_type(np.float64))
            for i in range(N):
                for j in range(N):
                    Xij = X2[5 * i : 5 * (i + 1), 5 * j : 5 * (j + 1)]
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

            evals, evecs = np.linalg.eigh(Aminus)
            idx = np.argmax(abs(evals))
            Z = evecs[:, idx] * np.sqrt(abs(evals[idx]))
            c = self._find_phase(Z[:, None] @ Z[:, None].T, Aplus)
            Z = np.sqrt(c) * Z
            alpha_est = (np.angle(Z)) % (2 * np.pi)

            return alpha_est, beta_est

        alpha_est, beta_est = find_alpha_beta(X2)
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

    ############################
    # Euler Estimation Helpers #
    ############################
    def _real_to_complex_representation(self, X, degree):
        """
        Convert a degree-wise representation matrix from the real basis used
        by ADMM to the complex Wigner basis.

        :param X: Representation matrix of shape
            (n_img * (2 * degree + 1), n_img * (2 * degree + 1)).
        :param degree: Wigner representation degree.

        :return: Representation matrix in the complex Wigner basis.
        """
        T, Tinv = self.complex2real(degree)
        identity = np.eye(self.n_img, dtype=np.float64)

        return np.kron(identity, T) @ X @ np.kron(identity, Tinv)

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

        C = np.zeros((N, N), dtype=complex_type(np.float64))
        Jk = self._handedness_matrix(degree)
        ws = self.Wd(degree, beta)

        for i in range(N):
            wi = ws[i]
            Di = np.exp(-1j * modes * alpha[i])

            for j in range(i + 1, N):
                wj = ws[j]
                Dj = np.exp(-1j * modes * alpha[j])

                Xij = Xm[
                    dk * i : dk * (i + 1),
                    dk * j : dk * (j + 1),
                ]

                DXijD = np.diag(Di.conj()) @ Xij @ np.diag(Dj)

                C[i, j] = pair_estimator(
                    wi,
                    wj,
                    DXijD,
                    Jk,
                    degree,
                )

        C += C.T.conj() + np.eye(N, dtype=np.float64)

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        leading_idx = np.argmax(eigenvalues)
        phase_vector = eigenvectors[:, leading_idx] * np.sqrt(eigenvalues[leading_idx])

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

    ####################
    # Helper Functions #
    ####################
    def transform_coeff(self, A, IDX_upper):
        """
        Convert representation matrices to the block-vector form used by ADMM.

        :param A: Representation matrices.
        :param IDX_upper: Indices of upper-triangular image pair blocks.

        :return: The two block-vector coefficient arrays.
        """
        d0 = [0]
        d1 = [0]
        for k in range(1, self.Lmax + 1):
            d0.append(d0[-1] + k**2)
            d1.append(d1[-1] + (k + 1) ** 2)
        A0 = xp.zeros((d0[-1], self.n_img * (self.n_img + 1) // 2), dtype=np.float64)
        A1 = xp.zeros((d1[-1], self.n_img * (self.n_img + 1) // 2), dtype=np.float64)
        for k in range(1, self.Lmax + 1):
            a0, a1 = self.permutek(A[k - 1], k, self.n_img)
            A0[d0[k - 1] : d0[k], :] = self.vec_block(a0, self.n_img, k, IDX_upper)
            A1[d1[k - 1] : d1[k], :] = self.vec_block(a1, self.n_img, k + 1, IDX_upper)
        return A0, A1

    def transform_coeff_back(self, A0, A1, IDX_upper, IDX_lower, idx_offdiag):
        """
        Reconstruct representation matrices from their ADMM block-vector form.

        :param A0: First block-vector array.
        :param A1: Second block-vector array.
        :param IDX_upper: Indices of upper-triangular image-pair blocks.
        :param IDX_lower: Indices of lower-triangular image-pair blocks.
        :param idx_offdiag: Indices of off-diagonal image pairs.

        :return: Reconstructed representation matrices.
        """
        d0 = [0]
        d1 = [0]
        N = self.n_img
        for k in range(1, self.Lmax + 1):
            d0.append(d0[-1] + k**2)
            d1.append(d1[-1] + (k + 1) ** 2)
        A = []
        for k in range(1, self.Lmax + 1):
            dk = 2 * k + 1
            Ak = xp.zeros((N * dk, N * dk), dtype=np.float64)
            Ak[: N * k, : N * k] = self.mat_block(
                A0[d0[k - 1] : d0[k], :], N, k, IDX_upper, IDX_lower, idx_offdiag
            )
            Ak[N * k :, N * k :] = self.mat_block(
                A1[d1[k - 1] : d1[k], :], N, k + 1, IDX_upper, IDX_lower, idx_offdiag
            )
            Ak = self.permutek_back(Ak, k, N)
            A.append(Ak)
        return A

    @staticmethod
    def permutek(Ak, k, N):
        """
        Permute and split a degree-k matrix into blocks of sizes k and k + 1.

        :param Ak: Degree-k block matrix.
        :param k: Representation degree.
        :param N: Number of images.

        :return: The two permuted matrix blocks.
        """
        AkP = xp.copy(Ak)
        dk = 2 * k + 1
        Pk = xp.eye(dk, dtype=AkP.dtype)
        for m in range(k):
            for n in range(k - m):
                Pk[(m + 2 * n, m + 2 * n + 1), :] = Pk[(m + 2 * n + 1, m + 2 * n), :]
        AkP = (
            xp.kron(xp.eye(N, dtype=AkP.dtype), Pk)
            @ Ak
            @ xp.kron(xp.eye(N, dtype=AkP.dtype), Pk.T)
        )

        Pk = xp.eye(N * dk, dtype=AkP.dtype)
        idx = xp.concatenate((xp.arange(dk - k, dk), xp.arange(k + 1)))
        for m in range(N - 1):
            for n in range(N - 1 - m):
                Pk[k * (m + 1) + n * dk : k * (m + 1) + (n + 1) * dk] = Pk[
                    k * (m + 1) + n * dk : k * (m + 1) + (n + 1) * dk
                ][idx, :]
        AkP = Pk @ AkP @ Pk.T
        return AkP[: N * k, : N * k], AkP[N * k :, N * k :]

    @staticmethod
    def permutek_back(Ak, k, N):
        """
        Undo the degree-k block permutation and reconstruct the full matrix.

        :param Ak: Permuted degree-k matrix.
        :param k: Representation degree.
        :param N: Number of images.

        :return: Matrix in the original block ordering.
        """
        dk = 2 * k + 1
        Pk = xp.eye(N * dk, dtype=Ak.dtype)
        idx = xp.concatenate((xp.arange(dk - k, dk), xp.arange(k + 1)))
        for m in range(N - 1):
            for n in range(N - 1 - m):
                Pk[k * (m + 1) + n * dk : k * (m + 1) + (n + 1) * dk] = Pk[
                    k * (m + 1) + n * dk : k * (m + 1) + (n + 1) * dk
                ][idx, :]
        AkB = Pk.T @ Ak @ Pk
        dk = 2 * k + 1
        Pk = xp.eye(dk, dtype=Ak.dtype)
        for m in range(k):
            for n in range(k - m):
                Pk[(m + 2 * n, m + 2 * n + 1), :] = Pk[(m + 2 * n + 1, m + 2 * n), :]
        AkB = (
            xp.kron(xp.eye(N, dtype=Ak.dtype), Pk.T)
            @ AkB
            @ xp.kron(xp.eye(N, dtype=Ak.dtype), Pk)
        )
        return AkB

    @staticmethod
    def vec_block(A, N, sz, IDX_upper):
        """
        Vectorize the upper-triangular image-pair blocks of a block matrix.
        """
        vecA = (A.reshape(N, sz, N, sz).transpose(0, 2, 3, 1)).reshape(N**2, sz**2).T
        return vecA[:, IDX_upper]

    @staticmethod
    def largest_eigenvalue(AI, Ngrid, N):
        """
        Estimate the largest eigenvalue of the Fejér constraint operator.
        """
        # find the largest eigenvalue of the operator AI
        z = xp.random.normal(0, 1, (Ngrid, N**2))
        Lambda = 0

        while abs(Lambda - xp.linalg.norm(z)) > 500:
            Lambda = xp.linalg.norm(z)
            z = z / xp.linalg.norm(z)
            z = AI @ (AI.T @ z)
        Lambda += 2000
        logger.info("Largest eigenvalue of AIAIT is approximately %1.2f" % Lambda)
        return Lambda

    def compute_rank(self):
        """
        Compute the ranks and matrices of the symmetry-averaging projectors at each degree.

        :param Lmax: Maximum representation degree.

        :return: Ranks and symmetry-averaging matrices for each degree.
        """
        rk = xp.zeros(self.Lmax, dtype=np.float64)
        A = []
        for k in range(1, self.Lmax + 1):
            Ak = np.sum(self.WD(k, self.sym_euler), axis=0)
            Ak = np.round(Ak / self.n_sym, 6)
            A.append(Ak)
            rk[k - 1] = np.linalg.matrix_rank(Ak)
        return rk, A

    def WD(self, J, euler):
        """
        Evaluate degree-J Wigner D matrices at the supplied ZYZ Euler angles.
        """
        # compute Wigner D matrix
        alpha = euler[:, 0]
        beta = euler[:, 1]
        gamma = euler[:, 2]
        d = self.Wd(J, beta)

        m = np.arange(-J, J + 1)
        left = np.exp(-1j * alpha[:, None] * m[None, :])
        right = np.exp(-1j * gamma[:, None] * m[None, :])
        D = left[:, :, None] * d * right[:, None, :]

        return D

    @staticmethod
    def Wd(J, beta):
        """
        Evaluate degree-J Wigner small-d matrices at the supplied polar angles.
        """
        # compute Wigner small d matrix
        d = np.zeros((len(beta), 2 * J + 1, 2 * J + 1), dtype=beta.dtype)
        for m in range(-J, J + 1):
            for n in range(-J, J + 1):
                smin = max(0, m - n)
                smax = min(J + m, J - n)
                for s in range(smin, smax + 1):
                    mul = (
                        np.sqrt(factorial(J + m))
                        / factorial(J + m - s)
                        * np.sqrt(factorial(J + n))
                        / factorial(s)
                        * np.sqrt(factorial(J - m))
                        / factorial(n - m + s)
                        * np.sqrt(factorial(J - n))
                        / factorial(J - n - s)
                    )
                    d[:, n + J, m + J] += (
                        mul
                        * (-1) ** (n - m + s)
                        * (np.cos(beta / 2)) ** (2 * J + m - n - 2 * s)
                        * (np.sin(beta / 2)) ** (n - m + 2 * s)
                    )
        return d

    @staticmethod
    def mat_block(vecA, N, sz, IDX_upper, IDX_lower, idx_offdiag):
        """
        Reconstruct a symmetric block matrix from its vectorized upper-triangular blocks.
        """
        tmp = vecA.T.reshape(N * (N + 1) // 2, sz, sz).transpose(0, 2, 1)
        AA = xp.zeros((N**2, sz, sz), dtype=vecA.dtype)
        AA[IDX_upper] = tmp
        AA[IDX_lower] = tmp[idx_offdiag].transpose(0, 2, 1)
        return (AA.reshape(N, N, sz, sz).transpose(0, 2, 1, 3)).reshape(N * sz, N * sz)

    @staticmethod
    def psd_projection(B):
        """
        Project one or more symmetric matrices onto the positive semidefinite cone.
        """
        # compute the PSD part of a symmstric matrix
        B_sym = (B + B.swapaxes(-1, -2)) / 2
        evals, evecs = xp.linalg.eigh(B_sym)
        evals = xp.maximum(evals, 0)
        return (evecs * evals[..., None, :]) @ evecs.swapaxes(-1, -2)

    @staticmethod
    def transform_block(A, k, Pk):
        """
        Permute and vectorize the two invariant blocks of a degree-k matrix.
        """
        AT = Pk @ A @ Pk.T
        A0 = AT[:, :k, :k].swapaxes(-1, -2).reshape(A.shape[0], -1)
        A1 = AT[:, k:, k:].swapaxes(-1, -2).reshape(A.shape[0], -1)
        return A0, A1

    @staticmethod
    def transform_back_block(A0, A1, k, Pk):
        """
        Reconstruct a degree-k matrix from its two invariant block vectors.
        """
        dk = 2 * k + 1
        A = xp.zeros((A0.shape[0], dk, dk), dtype=A0.dtype)
        A[:, :k, :k] = A0.reshape(-1, k, k).swapaxes(-1, -2)
        A[:, k:, k:] = A1.reshape(-1, k + 1, k + 1).swapaxes(-1, -2)
        return Pk.T @ A @ Pk

    def construct_AEq(self):
        """
        Construct the linear equality operator encoding the quaternion constraints.
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

        AEq[:16, 16:] = extra

        # Last row: redundant trace/sum constraint
        AEq[16, [0, 5, 10, 15]] = 1

        return AEq

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
            wigner = self.WD(k, euler_angles)

            # Average the degree-k representation over the molecular symmetry
            # group. For the asymmetric case, self.sym_euler contains only the
            # identity and Ak is therefore the identity matrix.
            Ak = np.mean(self.WD(k, self.sym_euler), axis=0)

            # Construct the globally handedness-conjugated solution.
            Jk = np.ones(dk)
            Jk[k + 1 :: 2] = -1
            Jk[k - 1 :: -2] = -1
            Jk = np.diag(Jk)

            wigner_J = Jk @ wigner @ Jk

            # Convert each representation to the real basis used by ADMM.
            _, Tinv = self.complex2real(k)
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
