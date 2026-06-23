import logging
import time

import numpy as np
from scipy.special import factorial

from aspire.abinitio import Orient3D
from aspire.numeric import xp
from aspire.operators import PolarFT
from aspire.utils import Rotation, cart2sph, complex_type
from aspire.volume import CnSymmetryGroup, DnSymmetryGroup, SymmetryGroup

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
        perform_pr=False,
        verbose=True,
        **kwargs,
    ):
        """
        Initialize the symmetric NUG orientation estimator.

        :param src: Source containing the input projection images.
        :param symmetry: Cyclic or dihedral symmetry specification, such as 'C3' or
            'D4'. If omitted, uses the source symmetry.
        :param n_rad: Number of radial samples in the polar Fourier transform.
        :param n_theta: Number of angular samples in the polar Fourier transform.
        :param max_shift: Maximum shift considered when comparing common lines.
        :param shift_step: Sampling interval for candidate shifts.
        :param mask: Whether to apply a circular mask to the input images.
        :param Lmax: Maximum Wigner representation degree used in the relaxation.
        :param T: Quadrature resolution used to compute the Fourier coefficients.
        :param max_iter: Number of ADMM iterations.
        :param rho: Initial ADMM penalty parameter.
        :param ratio: Residual ratio used when updating the ADMM penalty.
        :param factor: Scaling factor used when updating the ADMM penalty.
        :param mult: Step-size multiplier for the ADMM primal update.
        :param S2_grid: Number of sphere samples used to discretize SO(3).
        :param Nstep_yI: Number of inequality-multiplier updates per ADMM iteration.
        :param perform_pr: Whether to apply proximal refinement after ADMM.
        :param verbose: Whether to log ADMM progress.
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
        self.perform_pr = perform_pr
        self.verbose = verbose

        # Handle symmetry
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
        self.sym_euler = self.sym_grp.rotations.angles
        self.n_sym = len(self.sym_euler)

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
        Estimate rotations by computing NUG coefficients, solving the SDP relaxation, and recovering Euler angles.

        :return: Estimated rotation matrices.
        """
        self.compute_coeff()
        self.perform_admm()
        self.euler_est()
        return self.rotations

    #######################
    # Compute Coeffs Step #
    #######################

    def compute_coeff(self):
        """
        Compute the truncated Fourier coefficient matrices of the pairwise common-line losses.
        """
        # compute the coefficient matrix
        N = self.n_img
        n_theta = self.n_theta
        Lmax = self.Lmax
        T = self.T

        def fij(alpha, gamma, i, j):
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

        alpha_grid = np.arange(2 * T, dtype=np.float64) * np.pi / T
        beta_grid = (2 * np.arange(2 * T, dtype=np.float64) + 1) * np.pi / 4 / T
        gamma_grid = np.arange(2 * T, dtype=np.float64) * np.pi / T

        bT = np.zeros(2 * T, dtype=np.float64)
        for n in range(2 * T):
            ss = 0
            for m in range(T):
                ss = ss + np.sin(beta_grid[n] * (2 * m + 1)) / (2 * m + 1)
            bT[n] = 2 / T * np.sin(beta_grid[n]) * ss

        BTK = []
        for k in range(1, Lmax + 1):
            btk = np.sum(bT[:, None, None] * self.Wd(k, beta_grid), axis=0)
            BTK.append(btk.T)

        def fijhat_k(k, F):
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

        C = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            C.append(np.zeros((N * dk, N * dk), dtype=complex_type(np.float64)))
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
        for k in range(1, Lmax + 1):
            C[k - 1] = C[k - 1] + C[k - 1].conj().T

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
        Solve the symmetric NUG relaxation and optionally apply proximal refinement.
        """
        X_est = self.admm_sym_J(self.C, self.verbose)

        if self.perform_pr:
            weight = 1 / (1 + np.arange(self.Lmax))
            Penalty = [1, 1, 1, 1]
            r = [3, 2, 1, 0]
            X_est = self.proximal_refine(
                X_est,
                weight,
                Penalty,
                r,
            )
        self.X_est = X_est

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

        # admm for symmetric case
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
        bE = xp.zeros((Lmax + D0 + D1), dtype=np.float64)
        for k in range(Lmax):
            bE[k + d0[k] + d1[k] :] = rank_Ak[k]
            bE[k + 1 + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k]] = xp.eye(
                k + 1
            ).T.reshape(-1)
            bE[k + 1 + d0[k + 1] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]] = xp.eye(
                k + 2
            ).T.reshape(-1)
        bE = xp.repeat(bE[:, np.newaxis], N, axis=1)
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
            return z, AEq @ xp.concatenate(
                (Xq, X0[:1, idx_offdiag], X1[:4, idx_offdiag]), axis=0
            )

        def fun_AET(yE, yEq):
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
            z = xp.zeros((Ngrid, N * (N + 1) // 2), dtype=np.float64)
            tmp = xp.concatenate((X0, X1), axis=0)
            z[:, idx_diag] = AI_mat_diag @ tmp[:, idx_diag]
            z[:, idx_offdiag] = AI_mat_offdiag @ tmp[:, idx_offdiag]
            return z

        def fun_AIT(yI):
            Z = xp.zeros((D0 + D1, N * (N + 1) // 2), dtype=np.float64)
            Z[:, idx_diag] = AI_mat_diag.T @ yI[:, idx_diag]
            Z[:, idx_offdiag] = AI_mat_offdiag.T @ yI[:, idx_offdiag]
            return Z[:D0, :], Z[D0:, :]

        def update_S(C0, C1, yE, yEq, yI, X0, X1, Xd0, Xd1, Xq, rho, Lmax, N):
            Z0, Z1, Zd0, Zd1, Zq = fun_AET(yE, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            S0 = C0 - Z0 - AIT_yI0 - X0 / rho
            S1 = C1 - Z1 - AIT_yI1 - X1 / rho
            Sd0 = -Zd0 - Xd0 / rho
            Sd1 = -Zd1 - Xd1 / rho
            Sq = -Zq - Xq / rho
            tic0 = time.perf_counter()
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
            toc0 = time.perf_counter()
            Time[0] += toc0 - tic0

            Sd0 = Sd0.T
            Sd1 = Sd1.T
            tic1 = time.perf_counter()
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
            toc1 = time.perf_counter()
            Time[1] += toc1 - tic1

            tic2 = time.perf_counter()
            Sq = self.psd_projection(Sq.T.reshape(n_pairs, 4, 4))
            Sq = Sq.T.reshape(-1, n_pairs)
            toc2 = time.perf_counter()
            Time[2] += toc2 - tic2
            return S0, S1, Sd0, Sd1, Sq

        def update_yE(C0, C1, X0, X1, Xd0, Xd1, Xq, S0, S1, Sd0, Sd1, Sq, yI, rho):
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
                res_eq = np.linalg.norm(z - bE) / (
                    1 + np.linalg.norm(bE)
                ) + np.linalg.norm(zq - bEq) / (1 + np.linalg.norm(bEq))
                res_inq = np.linalg.norm(np.maximum(bI - fun_AI(X0, X1), 0)) / (
                    1 + abs(bI) * np.sqrt(Ngrid * N * (N + 1) / 2)
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
                    res_psdX += np.linalg.norm(self.psd_projection(-tmp))
                    tmp = self.mat_block(
                        X1[d1[k - 1] : d1[k], :],
                        N,
                        k + 1,
                        IDX_upper,
                        IDX_lower,
                        idx_offdiag,
                    )
                    res_psdX += np.linalg.norm(self.psd_projection(-tmp))
                res_psdX = res_psdX / (1 + np.linalg.norm(X0) + np.linalg.norm(X1))
                res_psdD = 0
                for k in range(1, Lmax + 1):
                    tmp = self.transform_back_block(
                        Xd0[d0[k - 1] : d0[k]].T,
                        Xd1[d1[k - 1] : d1[k]].T,
                        k,
                        P[k - 1],
                    )
                    res_psdD += np.linalg.norm(
                        self.psd_projection(-tmp), axis=(-2, -1)
                    ).sum()
                res_psdD = res_psdD / (1 + np.linalg.norm(Xd0) + np.linalg.norm(Xd1))
                res_psdQ = 0
                for count in range(N * (N - 1) // 2):
                    tmp = Xq[:, count].reshape(4, 4).T
                    res_psdQ += np.linalg.norm(self.psd_projection(-tmp))
                res_psdQ = res_psdQ / (1 + np.linalg.norm(Xq))

                normS = np.sqrt(
                    np.linalg.norm(S0) ** 2
                    + np.linalg.norm(S1) ** 2
                    + np.linalg.norm(Sd0) ** 2
                    + np.linalg.norm(Sd1) ** 2
                    + np.linalg.norm(Sq) ** 2
                )
                normX = np.sqrt(
                    np.linalg.norm(X0) ** 2
                    + np.linalg.norm(X1) ** 2
                    + np.linalg.norm(Xd0) ** 2
                    + np.linalg.norm(Xd1) ** 2
                    + np.linalg.norm(Xq) ** 2
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

        Xd0 = xp.zeros((D0, N), dtype=np.float64)
        Xd1 = xp.zeros((D1, N), dtype=np.float64)
        Sd0 = xp.zeros(Xd0.shape, dtype=np.float64)
        Sd1 = xp.zeros(Xd1.shape, dtype=np.float64)
        yI = xp.zeros((Ngrid, N * (N + 1) // 2), dtype=np.float64)
        yE = xp.zeros(bE.shape, dtype=np.float64)
        yEq = xp.zeros(bEq.shape, dtype=np.float64)

        IDX = np.arange(3)
        Time = np.zeros(4)
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
        count = 0
        idx_diag = []
        idx_offdiag = []
        for i in range(N):
            for j in range(i, N):
                if j == i:
                    idx_diag.append(count)
                else:
                    idx_offdiag.append(count)
                count += 1
        IDX_upper = []
        IDX_lower = []
        for i in range(N):
            for j in range(N):
                if j >= i:
                    IDX_upper.append(i * N + j)
        for j in range(N):
            for i in range(N):
                if j < i:
                    IDX_lower.append(i * N + j)

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
        normC = np.sqrt(np.linalg.norm(C0) ** 2 + np.linalg.norm(C1) ** 2)
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
        bEq = xp.repeat(bEq[:, xp.newaxis], N * (N - 1) // 2, axis=1)

        # AI and bI
        W0, W1, Ngrid = self.compute_fejer_weights()
        AI_mat_offdiag = np.zeros((Ngrid, D0 + D1), dtype=np.float64)
        for p in range(Ngrid):
            w0 = np.zeros(D0, dtype=np.float64)
            w1 = np.zeros(D1, dtype=np.float64)
            for k in range(1, Lmax + 1):
                w0[d0[k - 1] : d0[k]] = (
                    (Lmax - k + 2)
                    * (Lmax - k + 1)
                    * (k + 0.5)
                    * W0[k - 1][p].T.reshape(-1)
                )
                w1[d1[k - 1] : d1[k]] = (
                    (Lmax - k + 2)
                    * (Lmax - k + 1)
                    * (k + 0.5)
                    * W1[k - 1][p].T.reshape(-1)
                )
                # this needs double checking (Ruiyi)
            AI_mat_offdiag[p, : d0[-1]] = w0
            AI_mat_offdiag[p, d0[-1] :] = w1

        # Vectorized version, added by Josh
        # AI_mat_offdiag_new = np.zeros((Ngrid, D0 + D1))
        # for k in range(1, Lmax + 1):
        #     scale = (Lmax - k + 2) * (Lmax - k + 1) * (k + 0.5)

        #     block0 = scale * W0[k - 1].transpose(0, 2, 1).reshape(Ngrid, -1)
        #     block1 = scale * W1[k - 1].transpose(0, 2, 1).reshape(Ngrid, -1)

        #     AI_mat_offdiag_new[:, d0[k - 1]:d0[k]] = block0
        #     AI_mat_offdiag_new[:, d0[-1] + d1[k - 1]:d0[-1] + d1[k]] = block1

        AI_mat_diag = np.zeros((Ngrid, D0 + D1), dtype=np.float64)
        for p in range(Ngrid):
            w0 = np.zeros(D0, dtype=np.float64)
            w1 = np.zeros(D1, dtype=np.float64)
            for k in range(1, Lmax + 1):
                w0[d0[k - 1] : d0[k]] = (
                    (Lmax - k + 2)
                    * (Lmax - k + 1)
                    * (k + 0.5)
                    * (0.5 * W0[k - 1][p] + 0.5 * W0[k - 1][p].T).T.reshape(-1)
                )
                w1[d1[k - 1] : d1[k]] = (
                    (Lmax - k + 2)
                    * (Lmax - k + 1)
                    * (k + 0.5)
                    * (0.5 * W1[k - 1][p] + 0.5 * W1[k - 1][p].T).T.reshape(-1)
                )
                # this needs double checking (Ruiyi)
            AI_mat_diag[p, : d0[-1]] = w0
            AI_mat_diag[p, d0[-1] :] = w1
        AI_mat_diag = xp.asarray(AI_mat_diag) / 1
        AI_mat_offdiag = xp.asarray(AI_mat_offdiag) / 1
        bI = -(Lmax + 2) * (Lmax + 1) / 2 / 1

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
            # compute Ak matrix
            order = Euler.shape[0]
            A = self.WD(J, Euler).sum(axis=0)
            return np.round(A / order, 10)

        def rel_change(A, B, eps=1e-12):
            num = 0.0
            den = 0.0
            for k in range(len(A)):
                num += np.linalg.norm(A[k] - B[k]) ** 2
                den += np.linalg.norm(B[k]) ** 2
            return np.sqrt(num) / max(np.sqrt(den), eps)

        rank_Ak = np.zeros(self.Lmax, dtype=np.float64)
        C_base = [None] * self.Lmax
        for k in range(self.Lmax):
            C_base[k] = xp.asnumpy(C[k]).copy()
            rank_Ak[k] = np.linalg.matrix_rank(
                Ak(k + 1, self.sym_euler), tol=1e-6, hermitian=True
            )

        def low_rank_proj(X, r_step):
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

        Niter = len(r)
        CC = [None] * self.Lmax
        current = [np.copy(Xk) for Xk in X_admm]

        for step in range(Niter):
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
                    Niter,
                    rel_change(X_next, current),
                )

            current = [np.copy(Xk) for Xk in X_next]

        return current

    #########################
    # Euler Estimation Step #
    #########################

    def euler_est(self):
        """
        Recover rotations and Euler angles using the estimator for the configured symmetry group.
        """
        X_est = self.X_est
        if isinstance(self.sym_grp, CnSymmetryGroup):
            R_est, Euler_est = self.euler_est_Cm(X_est[0], X_est[self.n_sym - 1])
        elif isinstance(self.sym_grp, DnSymmetryGroup):
            R_est, Euler_est = self.euler_est_Dm(X_est)

        self.Euler_est = Euler_est
        self.rotations = R_est.astype(self.dtype)

    def euler_est_Cm(self, X1, XS):
        """
        Recover Euler angles for cyclic symmetry.

        :param X1: Relaxed degree-one representation matrix.
        :param XS: Relaxed representation matrix at the symmetry order.

        :return: Estimated rotation matrices and Euler angles.
        """
        S = self.n_sym
        N = self.n_img
        sym_euler = np.zeros((S, 3), dtype=np.float64)
        for s in range(S):
            sym_euler[s] = [2 * np.pi * s / S, 0, 0]
        [T, Tinv] = self.complex2real(1)
        X1 = (
            np.kron(np.eye(N, dtype=np.float64), T)
            @ X1
            @ np.kron(np.eye(N, dtype=np.float64), Tinv)
        )
        [T, Tinv] = self.complex2real(S)
        XS = (
            np.kron(np.eye(N, dtype=np.float64), T)
            @ XS
            @ np.kron(np.eye(N, dtype=np.float64), Tinv)
        )

        def find_phase(A, B):
            # find a number c that minimizes ||cA-B||_F
            Ar = np.real(A)
            Ai = np.imag(A)
            Br = np.real(B)
            Bi = np.imag(B)
            c = (np.vdot(Ar, Br) + np.vdot(Ai, Bi)) / (
                np.vdot(Ar, Ar) + np.vdot(Ai, Ai)
            ) + 1j * (np.vdot(Ar, Bi) - np.vdot(Ai, Br)) / (
                np.vdot(Ar, Ar) + np.vdot(Ai, Ai)
            )
            return c / abs(c)

        def find_beta(X1):
            B1 = np.zeros((N, N), dtype=np.float64)
            B2 = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    Xij = X1[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)]
                    B1[i, j] = abs(Xij[0, 0]) * 2
                    B2[i, j] = np.real(Xij[1, 1])
            e1, v1 = np.linalg.eigh(B1)
            idx = np.argmax(e1)
            b1 = -v1[:, idx] * np.sqrt(e1[idx])
            e2, v2 = np.linalg.eigh(B2)
            idx = np.argmax(e2)
            b2 = v2[:, idx] * np.sqrt(e2[idx])
            beta = np.arctan(b1 / b2) % np.pi
            return beta

        def find_alpha(X1):
            ZZbar = np.zeros((N, N), dtype=complex_type(np.float64))
            ZZ = np.zeros((N, N), dtype=complex_type(np.float64))
            for i in range(N):
                for j in range(N):
                    z = X1[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)][0, 0]
                    ZZbar[i, j] = z / abs(z)

                    z = X1[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)][0, 2]
                    ZZ[i, j] = -z / abs(z)

            evals, evecs = np.linalg.eigh(ZZbar)
            idx = np.argmax(abs(evals))
            Z = evecs[:, idx] * np.sqrt(abs(evals[idx]))

            c = find_phase(Z[:, None] @ Z[:, None].T, ZZ)
            Z = np.sqrt(c) * Z
            return np.angle(Z).astype(np.float64)

        dk = 2 * S + 1

        def find_gamma(Xm, beta, alpha):
            C = np.zeros((N, N), dtype=complex_type(np.float64))
            Jk = np.ones(dk)
            Jk[S + 1 :: 2] = -1
            Jk[S - 1 :: -2] = -1
            Jk = np.diag(Jk)
            ws = self.Wd(S, beta)
            for i in range(N):
                wi = ws[i]
                for j in range(i + 1, N):
                    Di = np.exp(-1j * np.arange(-S, S + 1) * alpha[i])
                    Dj = np.exp(-1j * np.arange(-S, S + 1) * alpha[j])
                    Xijm = Xm[dk * i : dk * (i + 1), dk * j : dk * (j + 1)]
                    DXijmD = np.diag(Di.conj()) @ Xijm @ np.diag(Dj)
                    wj = ws[j]
                    C1 = (
                        wi[:, 0][:, None] @ (wj[:, 0][:, None].T)
                        + Jk @ wi[:, 0][:, None] @ (wj[:, 0][:, None].T) @ Jk
                    ) / 2
                    C2 = (
                        wi[:, -1][:, None] @ (wj[:, -1][:, None].T)
                        + Jk @ wi[:, -1][:, None] @ (wj[:, -1][:, None].T) @ Jk
                    ) / 2
                    C3 = (
                        DXijmD
                        - (
                            wi[:, S][:, None] @ (wj[:, S][:, None].T)
                            + Jk @ wi[:, S][:, None] @ (wj[:, S][:, None].T) @ Jk
                        )
                        / 2
                    )
                    C[i, j] = np.vdot(C1 + C2, np.real(C3)) / np.vdot(
                        C1 + C2, C1 + C2
                    ) + 1j * np.vdot(C1 - C2, np.imag(C3)) / np.vdot(C1 - C2, C1 - C2)
            C += C.T.conj() + np.eye(N, dtype=np.float64)
            evals, evecs = np.linalg.eigh(C)
            idx = np.argmax(evals)
            c = evecs[:, idx] * np.sqrt(evals[idx])
            return np.angle(c) / S

        Euler_est = np.zeros((N, 3), dtype=np.float64)
        Euler_est[:, 0] = find_alpha(X1)
        Euler_est[:, 1] = find_beta(X1)
        Euler_est[:, 2] = find_gamma(XS, Euler_est[:, 1], Euler_est[:, 0])
        R_est = Rotation.from_euler(Euler_est).matrices.transpose(0, 2, 1)
        return R_est, Euler_est

    def euler_est_Dm(self, X_est):
        """
        Recover Euler angles for dihedral symmetry.

        :param X_est: Relaxed representation matrices.

        :return: Estimated rotation matrices and Euler angles.
        """
        X2 = X_est[1]
        S = self.sym_grp.order
        N = self.n_img
        XS = X_est[S - 1]
        dk = 2 * S + 1

        def find_alpha_beta(X2):
            def find_phase(A, B):
                # find a number c that minimizes ||cA-B||_F
                Ar = np.real(A)
                Ai = np.imag(A)
                Br = np.real(B)
                Bi = np.imag(B)
                c = (np.vdot(Ar, Br) + np.vdot(Ai, Bi)) / (
                    np.vdot(Ar, Ar) + np.vdot(Ai, Ai)
                ) + 1j * (np.vdot(Ar, Bi) - np.vdot(Ai, Br)) / (
                    np.vdot(Ar, Ar) + np.vdot(Ai, Ai)
                )
                return c / abs(c)

            T, Tinv = self.complex2real(2)
            X2 = (
                np.kron(np.eye(N, dtype=np.float64), T)
                @ X2
                @ np.kron(np.eye(N, dtype=np.float64), Tinv)
            )

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
            c = find_phase(Z[:, None] @ Z[:, None].T, Aplus)
            Z = np.sqrt(c) * Z
            alpha_est = (np.angle(Z)) % (2 * np.pi)

            return alpha_est, beta_est

        def find_gamma(Xm, alpha, beta):
            def LS_D(W1, W2, W3, W4, Br, Bi):
                A = np.array(
                    [
                        [np.vdot(W1 + W4, W1 + W4), np.vdot(W1 + W4, W2 + W3)],
                        [np.vdot(W2 + W3, W1 + W4), np.vdot(W2 + W3, W2 + W3)],
                    ]
                )
                B = np.array([np.vdot(W1 + W4, Br), np.vdot(W2 + W3, Br)])
                a, c = np.linalg.lstsq(A, B)[0]

                A = np.array(
                    [
                        [np.vdot(W1 - W4, W1 - W4), np.vdot(W1 - W4, W3 - W2)],
                        [np.vdot(W1 - W4, W3 - W2), np.vdot(W3 - W2, W3 - W2)],
                    ]
                )
                B = np.array([np.vdot(W1 - W4, Bi), np.vdot(W3 - W2, Bi)])
                b, d = np.linalg.lstsq(A, B)[0]
                return a + 1j * b

            [T, Tinv] = self.complex2real(S)
            Xm = (
                np.kron(np.eye(N, dtype=np.float64), T)
                @ Xm
                @ np.kron(np.eye(N, dtype=np.float64), Tinv)
            )
            C = np.zeros((N, N), dtype=complex_type(np.float64))
            Jk = np.ones(dk)
            Jk[S + 1 :: 2] = -1
            Jk[S - 1 :: -2] = -1
            Jk = np.diag(Jk)
            ws = self.Wd(S, beta)
            for i in range(N):
                wi = ws[i]
                for j in range(i + 1, N):
                    Di = np.exp(-1j * np.arange(-S, S + 1) * alpha[i])
                    Dj = np.exp(-1j * np.arange(-S, S + 1) * alpha[j])
                    Xijm = Xm[dk * i : dk * (i + 1), dk * j : dk * (j + 1)]
                    DXijmD = np.diag(Di.conj()) @ Xijm @ np.diag(Dj)
                    wj = ws[j]
                    W1 = (
                        wi[:, 0][:, np.newaxis] @ wj[:, 0][:, np.newaxis].T
                        + Jk @ wi[:, 0][:, np.newaxis] @ wj[:, 0][:, np.newaxis].T @ Jk
                    )
                    W2 = (
                        wi[:, -1][:, np.newaxis] @ wj[:, 0][:, np.newaxis].T
                        + Jk @ wi[:, -1][:, np.newaxis] @ wj[:, 0][:, np.newaxis].T @ Jk
                    )
                    W3 = (
                        wi[:, 0][:, np.newaxis] @ wj[:, -1][:, np.newaxis].T
                        + Jk @ wi[:, 0][:, np.newaxis] @ wj[:, -1][:, np.newaxis].T @ Jk
                    )
                    W4 = (
                        wi[:, -1][:, np.newaxis] @ wj[:, -1][:, np.newaxis].T
                        + Jk
                        @ wi[:, -1][:, np.newaxis]
                        @ wj[:, -1][:, np.newaxis].T
                        @ Jk
                    )
                    Br = np.real(4 * DXijmD)
                    Bi = np.imag(4 * DXijmD)
                    C[i, j] = LS_D(W1, W2, W3, W4, Br, Bi)
            C += C.T.conj() + np.eye(N, dtype=np.float64)
            evals, evecs = np.linalg.eigh(C)
            idx = np.argmax(evals)
            c = evecs[:, idx] * np.sqrt(evals[idx])
            return (np.angle(c) / S) % (2 * np.pi)

        alpha_est, beta_est = find_alpha_beta(X2)
        gamma_est = find_gamma(XS, alpha_est, beta_est)
        Euler_est = np.zeros((N, 3), dtype=np.float64)
        Euler_est[:, 0] = alpha_est
        Euler_est[:, 1] = beta_est
        Euler_est[:, 2] = gamma_est
        R_est = Rotation.from_euler(Euler_est).matrices.transpose(0, 2, 1)

        return R_est, Euler_est

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
        np.random.seed(0)
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
    def transform_block(A, k, Pk=None):
        """
        Permute and vectorize the two invariant blocks of a degree-k matrix.
        """
        single = A.ndim == 2
        if single:
            A = A[None, :, :]

        if Pk is None:
            dk = 2 * k + 1
            Pk = xp.eye(dk, dtype=A.dtype)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
        AT = Pk @ A @ Pk.T
        A0 = AT[:, :k, :k].swapaxes(-1, -2).reshape(A.shape[0], -1)
        A1 = AT[:, k:, k:].swapaxes(-1, -2).reshape(A.shape[0], -1)

        if single:
            return A0[0], A1[0]

        return A0, A1

    @staticmethod
    def transform_back_block(A0, A1, k, Pk=None):
        """
        Reconstruct a degree-k matrix from its two invariant block vectors.
        """
        dk = 2 * k + 1
        single = A0.ndim == 1

        if single:
            A0 = A0[None, :]
            A1 = A1[None, :]

        A = xp.zeros((A0.shape[0], dk, dk), dtype=A0.dtype)
        A[:, :k, :k] = A0.reshape(-1, k, k).swapaxes(-1, -2)
        A[:, k:, k:] = A1.reshape(-1, k + 1, k + 1).swapaxes(-1, -2)
        if Pk is None:
            Pk = xp.eye(dk, dtype=A0.dtype)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
        out = Pk.T @ A @ Pk
        return out[0] if single else out

    def construct_AEq(self):
        """
        Construct the linear equality operator encoding the quaternion constraints.
        """
        AEq = np.zeros((17, 21), np.float64)

        # First 16 rows: identity constraints on first 16 variables
        AEq[:16, :16] = np.eye(16, dtype=np.float64)

        # Extra columns 16:21
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
