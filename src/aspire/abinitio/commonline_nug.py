import logging
import time

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as spr
from scipy.special import factorial

from aspire.abinitio import CLOrient3D
from aspire.nufft import nufft
from aspire.numeric import fft, xp
from aspire.operators import PolarFT, wemd_embed
from aspire.utils import cart2sph
from aspire.volume import SymmetryGroup

logger = logging.getLogger(__name__)


class CommonlineNUG(CLOrient3D):
    """
    Class to estimate 3D orientations using non-uqique games.
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
        loss="l1",
        T=36,
        max_iter=501,
        rho=0.05,
        ratio=1,
        factor=1.0,
        mult=1.5,
        Ngrid=16317,
        Nstep_yI=10,
        **kwargs,
    ):
        """
        Initialize object for estimating 3D orientations for symmetric molecules.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param symmetry: A string, ie. 'C3', indicating the symmetry type.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction. Default = 360.
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
        self.loss = loss
        self.T = T
        self.max_iter = max_iter
        self.rho = rho
        self.ratio = ratio
        self.factor = factor
        self.mult = mult
        self.Ngrid = Ngrid
        self.Nstep_yI = Nstep_yI

        # Handle symmetry
        self.sym_grp = SymmetryGroup.parse(symmetry)
        self.sym_euler = self.sym_grp.rotations.angles
        self.n_sym = len(self.sym_euler)

        self._build_full_pft()

    def _build_full_pft(self):
        pf = self.pf
        self.pf_full = PolarFT.half_to_full(pf)

    def estimate_rotations(self):
        imgs = self.src.images[:]
        C = self.compute_coeff(imgs, self.loss, self.Lmax, T=self.T)
        X_est = self.admm_sym_J(
            C,
            self.Lmax,
            self.n_img,
            self.Ngrid,
            self.max_iter,
            self.rho,
            self.ratio,
            self.factor,
            self.mult,
            self.Nstep_yI,
        )

        R_est, Euler_est = self.euler_est(X_est[0], X_est[self.n_sym - 1])
        self.rotations = R_est

        return R_est

    #######################
    # Compute Coeffs Step #
    #######################

    def compute_coeff(self, Img, loss, Lmax, T):
        # compute the coefficient matrix
        N, L, _ = Img.shape
        n_theta = 360
        angular_sampling = np.arange(0, 360, 1)
        line_proj = np.zeros((L, n_theta, N))
        Img_pft = np.zeros((L, n_theta, N), dtype=complex)

        # Replace with Image.project() later
        Img = Img.asnumpy()
        for n in range(N):
            line_proj[:, :, n], Img_pft[:, :, n] = self.fast_radon_transform(
                Img[n], angular_sampling
            )

        dim_wave = len(wemd_embed(line_proj[:, 0, 0]))
        WE = np.zeros((dim_wave, n_theta, N))
        for i in range(N):
            for theta in range(n_theta):
                WE[:, theta, i] = wemd_embed(line_proj[:, theta, i])

        def fij(alpha, gamma, i, j, loss):
            if loss == "l1":
                Ii_hat = Img_pft[:, :, i]
                Ij_hat = Img_pft[:, :, j]
                idxi = np.round((alpha - np.pi / 2) * n_theta / 2 / np.pi) % n_theta
                idxj = np.round((-gamma - np.pi / 2) * n_theta / 2 / np.pi) % n_theta

                Si = Ii_hat[:, int(idxi)]
                Sj = Ij_hat[:, int(idxj)]

                # Using aspire PolarFT. Replace later
                # Ii_hat = self.pf_full[i]
                # Ij_hat = self.pf_full[j]
                # idxi = np.round((alpha - np.pi / 2) * n_theta / 2 / np.pi) % n_theta
                # idxj = np.round((-gamma - np.pi / 2) * n_theta / 2 / np.pi) % n_theta

                # Si = Ii_hat[int(idxi)]
                # Sj = Ij_hat[int(idxj)]
                # norm_new = np.linalg.norm(Si - Sj, 1)

                return np.linalg.norm(Si - Sj, 1)

            if loss == "wemd":
                idxi = np.round((alpha - np.pi / 2) * n_theta / 2 / np.pi) % n_theta
                idxj = np.round((-gamma - np.pi / 2) * n_theta / 2 / np.pi) % n_theta

                Si = WE[:, int(idxi), i]
                Sj = WE[:, int(idxj), j]
                return np.linalg.norm(Si - Sj, 1)

        alpha_grid = np.arange(2 * T) * np.pi / T
        beta_grid = (2 * np.arange(2 * T) + 1) * np.pi / 4 / T
        gamma_grid = np.arange(2 * T) * np.pi / T

        bT = np.zeros(2 * T)
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

            exp_alpha_grid = np.zeros((2 * T, dk), dtype=complex)
            for m in range(-k, k + 1):
                exp_alpha_grid[:, m + k] = np.exp(1j * m * alpha_grid)

            exp_gamma_grid = np.zeros((2 * T, dk), dtype=complex)
            for m in range(-k, k + 1):
                exp_gamma_grid[:, m + k] = np.exp(1j * m * gamma_grid)

            S = (exp_alpha_grid.T @ F @ exp_gamma_grid).T
            fhat = BTK[k - 1] * S / 4 / T**2
            return fhat

        C = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            C.append(np.zeros((N * dk, N * dk), dtype=complex))
        for i in range(N):
            for j in range(i + 1, N):
                Fij = np.zeros((2 * T, 2 * T))
                for j1 in range(2 * T):
                    for j2 in range(2 * T):
                        Fij[j1, j2] = fij(alpha_grid[j1], gamma_grid[j2], i, j, loss)
                for k in range(1, Lmax + 1):
                    dk = 2 * k + 1
                    C[k - 1][j * dk : (j + 1) * dk, i * dk : (i + 1) * dk] = fijhat_k(
                        k, Fij
                    )  # *dk
        for k in range(1, Lmax + 1):
            C[k - 1] = C[k - 1] + C[k - 1].conj().T

        for i in range(N):
            Fii = np.zeros((2 * T, 2 * T))
            for j1 in range(2 * T):
                for j2 in range(2 * T):
                    Fii[j1, j2] = fij(alpha_grid[j1], gamma_grid[j2], i, i, loss)
            for k in range(1, Lmax + 1):
                dk = 2 * k + 1
                C[k - 1][i * dk : (i + 1) * dk, i * dk : (i + 1) * dk] = fijhat_k(
                    k, Fii
                )  # *dk

        for k in range(1, Lmax + 1):
            [T, Tinv] = self.complex2real(k)
            C[k - 1] = np.real(
                np.kron(np.eye(N), Tinv) @ C[k - 1] @ np.kron(np.eye(N), T)
            )
            C[k - 1] = np.round(C[k - 1], 10)

        return C

    @staticmethod
    def fast_radon_transform(array, angles, use_ramp=False):

        angles = np.array(angles).flatten()
        img_size = array.shape[1]
        rads = angles / 180 * np.pi
        y_idx = np.arange(-img_size / 2, img_size / 2) / img_size * 2
        x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
        y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]

        pts = np.pi * np.vstack(
            [
                x_theta.flatten(),
                y_theta.flatten(),
            ]
        )
        pts = pts.astype(array.dtype)

        # array = array.astype(np.float32)
        lines_f = nufft(array, pts).reshape((img_size, -1))

        if img_size % 2 == 0:
            lines_f[0, :] = 0

        if use_ramp:
            freqs = np.abs(np.pi * y_idx)
            lines_f *= freqs[:, np.newaxis]

        projections = np.real(
            xp.asnumpy(fft.centered_ifft(xp.asarray(lines_f), axis=0))
        )

        return projections, lines_f

    @staticmethod
    def complex2real(ell):
        # compute transformation matrices that convert complex representations to real ones
        diml = 2 * ell + 1
        Tinv = np.zeros((diml, diml), dtype=complex)
        for i in range(diml):
            if i < ell:
                Tinv[i, i] = 1j / np.sqrt(2)
                Tinv[i, diml - 1 - i] = -1j * (-1) ** (i - ell) / np.sqrt(2)
            if i == ell:
                Tinv[i, i] = 1
            if i > ell:
                Tinv[i, i] = (-1) ** (i - ell) / np.sqrt(2)
                Tinv[i, diml - 1 - i] = 1 / np.sqrt(2)

        T = np.zeros((diml, diml), dtype=complex)
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

    def admm_sym_J(
        self,
        C,
        Lmax,
        N,
        Ngrid,
        max_iter,
        rho,
        ratio,
        factor,
        mult=1,
        Nstep_yI=20,
        verbose=True,
    ):
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
        ) = self.ADMM_preprocessing(C, Lmax, N, Ngrid)

        rank_Ak, _ = self.compute_rank(Lmax)
        logger.info(f"Rank of Ak: {rank_Ak}")

        # rank_Ak=cp.zeros(Lmax)
        # for ell in range(Lmax): rank_Ak[ell]=np.linalg.matrix_rank(Ak(ell+1,sym_euler))

        AE = []
        AEAETinv = []
        for k in range(1, Lmax + 1):
            s0 = k**2
            s1 = (k + 1) ** 2
            AEk = xp.zeros((1 + s0 + s1, 2 * (s0 + s1)))
            AEk[0, :s0] = xp.eye(k).T.reshape(-1)
            AEk[0, s0 : s0 + s1] = xp.eye(k + 1).T.reshape(-1)
            for count in range(1, 1 + s0):
                AEk[count, count - 1] = 1
                AEk[count, count - 1 + s0 + s1] = 1
            for count in range(1 + s0, 1 + s0 + s1):
                AEk[count, count - 1] = 1
                AEk[count, count - 1 + s1 + s0] = 1
            AE.append(AEk)
            AEAETinv.append(np.linalg.pinv(AEk @ AEk.T))
        bE = xp.zeros((Lmax + D0 + D1))
        for k in range(Lmax):
            bE[k + d0[k] + d1[k] :] = rank_Ak[k]
            bE[k + 1 + d0[k] + d1[k] : k + 1 + d0[k + 1] + d1[k]] = xp.eye(
                k + 1
            ).T.reshape(-1)
            bE[k + 1 + d0[k + 1] + d1[k] : k + 1 + d0[k + 1] + d1[k + 1]] = xp.eye(
                k + 2
            ).T.reshape(-1)
        bE = np.repeat(bE[:, np.newaxis], N, axis=1)
        P = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            Pk = xp.eye(dk)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
            P.append(Pk)

        def fun_AE(X0, X1, Xd0, Xd1, Xq):
            z = xp.zeros((Lmax + D0 + D1, N))
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
            Z0 = xp.zeros((D0, N * (N + 1) // 2))
            Z1 = xp.zeros((D1, N * (N + 1) // 2))
            Zd0 = xp.zeros((D0, N))
            Zd1 = xp.zeros((D1, N))
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
            z = xp.zeros((Ngrid, N * (N + 1) // 2))
            tmp = xp.concatenate((X0, X1), axis=0)
            z[:, idx_diag] = AI_mat_diag @ tmp[:, idx_diag]
            z[:, idx_offdiag] = AI_mat_offdiag @ tmp[:, idx_offdiag]
            # return AI_mat@xp.concatenate((X0,X1),axis=0)
            return z

        def fun_AIT(yI):
            # Z=AI_mat.T@yI
            # return Z[:D0,:], Z[D0:,:]
            Z = xp.zeros((D0 + D1, N * (N + 1) // 2))
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
            tic1 = time.perf_counter()
            for k in range(1, Lmax + 1):
                for n in range(N):
                    tmp = self.transform_back_block(
                        Sd0[d0[k - 1] : d0[k], n],
                        Sd1[d1[k - 1] : d1[k], n],
                        k,
                        P[k - 1],
                    )
                    tmp = self.psd_projection(tmp)
                    Sd0[d0[k - 1] : d0[k], n], Sd1[d1[k - 1] : d1[k], n] = (
                        self.transform_block(tmp, k, P[k - 1])
                    )
            toc1 = time.perf_counter()
            Time[1] += toc1 - tic1
            tic2 = time.perf_counter()
            for count in range(N * (N - 1) // 2):
                tmp = self.psd_projection(Sq[:, count].reshape(4, 4).T)
                Sq[:, count] = tmp.T.reshape(16)
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
            yE = xp.zeros((Lmax + D0 + D1, N))
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
            yI = np.maximum(yI, 0)
            return yI

        def update_X(
            C0, C1, X0, X1, Xd0, Xd1, Xq, yE, yEq, yI, S0, S1, Sd0, Sd1, Sq, rho
        ):
            Z0, Z1, Zd0, Zd1, Zq = fun_AET(yE, yEq)
            AIT_yI0, AIT_yI1 = fun_AIT(yI)
            tmp = S0 + Z0 + AIT_yI0 - C0
            X0 = X0 + mult * rho * tmp
            resX0 = np.linalg.norm(tmp)
            tmp = S1 + Z1 + AIT_yI1 - C1
            X1 = X1 + mult * rho * tmp
            resX1 = np.linalg.norm(tmp)
            tmp = Sd0 + Zd0
            Xd0 = Xd0 + mult * rho * tmp
            resXd0 = np.linalg.norm(tmp)
            tmp = Sd1 + Zd1
            Xd1 = Xd1 + mult * rho * tmp
            resXd1 = np.linalg.norm(tmp)
            tmp = Sq + Zq
            Xq = Xq + mult * rho * tmp
            resXq = np.linalg.norm(tmp)
            return (
                X0,
                X1,
                Xd0,
                Xd1,
                Xq,
                np.sqrt(resX0**2 + resX1**2 + resXd0**2 + resXd1**2 + resXq**2),
            )

        def update_rho(X0, X1, Xd0, Xd1, Xq, bE, bEq, bI, res_X, rho, factor, normC):
            z, zq = fun_AE(X0, X1, Xd0, Xd1, Xq)
            res_eq = np.linalg.norm(z - bE) / (1 + np.linalg.norm(bE)) + np.linalg.norm(
                zq - bEq
            ) / (1 + np.linalg.norm(bEq))
            res_inq = np.linalg.norm(np.maximum(bI - fun_AI(X0, X1), 0)) / (
                1 + abs(bI) * np.sqrt(Ngrid * N**2)
            )
            p_resnorm = res_eq + res_inq
            d_resnorm = res_X / (1 + normC)
            if d_resnorm > ratio * p_resnorm:
                rho = rho * factor
            if d_resnorm < ratio * p_resnorm:
                rho = rho / factor
            return rho, p_resnorm, d_resnorm

        def print_updates(verbose=True):
            # X_admm=transform_coeff_back(X0,X1,Lmax,N); obj_p=0
            # for k in range(Lmax): obj_p+=xp.trace(C[k]@X_admm[k])
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
                    # res_psdX+=norm(self.psd_projection(-tmp))/(1+norm(tmp))
                    tmp = self.mat_block(
                        X1[d1[k - 1] : d1[k], :],
                        N,
                        k + 1,
                        IDX_upper,
                        IDX_lower,
                        idx_offdiag,
                    )
                    res_psdX += np.linalg.norm(self.psd_projection(-tmp))
                    # res_psdX+=norm(self.psd_projection(-tmp))/(1+norm(tmp))
                res_psdX = res_psdX / (1 + np.linalg.norm(X0) + np.linalg.norm(X1))
                res_psdD = 0
                for k in range(1, Lmax + 1):
                    for n in range(N):
                        tmp = self.transform_back_block(
                            Xd0[d0[k - 1] : d0[k], n],
                            Xd1[d1[k - 1] : d1[k], n],
                            k,
                            P[k - 1],
                        )
                        res_psdD += np.linalg.norm(self.psd_projection(-tmp))
                        # res_psdD+=norm(self.psd_projection(-tmp))/(1+norm(tmp))
                res_psdD = res_psdD / (1 + np.linalg.norm(Xd0) + np.linalg.norm(Xd1))
                res_psdQ = 0
                for count in range(N * (N - 1) // 2):
                    tmp = Xq[:, count].reshape(4, 4).T
                    res_psdQ += np.linalg.norm(self.psd_projection(-tmp))
                    # res_psdQ+=norm(self.psd_projection(-tmp))/(1+norm(tmp))
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

        Xd0 = xp.zeros((D0, N))
        Xd1 = xp.zeros((D1, N))
        Sd0 = xp.zeros(Xd0.shape)
        Sd1 = xp.zeros(Xd1.shape)
        yI = xp.zeros((Ngrid, N * (N + 1) // 2))
        yE = xp.zeros(bE.shape)
        yEq = xp.zeros(bEq.shape)

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

        X_admm = self.transform_coeff_back(
            X0, X1, Lmax, N, IDX_upper, IDX_lower, idx_offdiag
        )
        # if self.GPU:
        #     for k in range(Lmax):
        #         X_admm[k] = X_admm[k].get()
        for k in range(Lmax):
            X_admm[k] = xp.asnumpy(X_admm[k])
        return X_admm

    def ADMM_preprocessing(self, C, Lmax, N, Ngrid):
        # compute necessary quantities for ADMM
        # compute some useful index sets
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
        C0, C1 = self.transform_coeff(C, Lmax, N, IDX_upper)
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
        AEq = xp.asarray(loadmat("data/Eq_constraints/AEqJ.mat")["AEq"])
        AEqAEqtinv = xp.asarray(loadmat("data/Eq_constraints/AEqJ.mat")["AEqAEqtinv"])

        bEq = xp.zeros(17)
        bEq[:16] = xp.eye(4).reshape(-1) / 4
        bEq[-1] = 1
        bEq = xp.repeat(bEq[:, xp.newaxis], N * (N - 1) // 2, axis=1)

        # AI and bI
        W0, W1 = self.compute_fejer_weights()

        # AI_mat=np.zeros((Ngrid,D0+D1))
        # for p in range(Ngrid):
        #     w0=np.zeros(D0); w1=np.zeros(D1)
        #     for k in range(1,Lmax+1):
        #         w0[d0[k-1]:d0[k]]=(Lmax-k+2)*(Lmax-k+1)*(k+0.5)*W0[k-1][p].T.reshape(-1)
        #         w1[d1[k-1]:d1[k]]=(Lmax-k+2)*(Lmax-k+1)*(k+0.5)*W1[k-1][p].T.reshape(-1)
        #         # this needs double checking
        #     AI_mat[p,:d0[-1]]=w0; AI_mat[p,d0[-1]:]=w1
        # AI_mat=xp.asarray(AI_mat) / 10;
        # bI=-(Lmax+2)*(Lmax+1)/2 / 10

        AI_mat_offdiag = np.zeros((Ngrid, D0 + D1))
        for p in range(Ngrid):
            w0 = np.zeros(D0)
            w1 = np.zeros(D1)
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
                # this needs double checking
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

        AI_mat_diag = np.zeros((Ngrid, D0 + D1))
        for p in range(Ngrid):
            w0 = np.zeros(D0)
            w1 = np.zeros(D1)
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
                # this needs double checking
            AI_mat_diag[p, : d0[-1]] = w0
            AI_mat_diag[p, d0[-1] :] = w1
        AI_mat_diag = xp.asarray(AI_mat_diag) / 1
        AI_mat_offdiag = xp.asarray(AI_mat_offdiag) / 1
        bI = -(Lmax + 2) * (Lmax + 1) / 2 / 1
        np.save("AI_mat_offdiag_aspire.npy", AI_mat_offdiag)
        # largest eigenvalue for AIAIT
        Lambda = self.largest_eigenvalue(AI_mat_offdiag, Ngrid, N)

        # initialization
        II = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            II.append(xp.eye(N * dk))
        I0, I1 = self.transform_coeff(II, Lmax, N, IDX_upper)
        X0 = xp.zeros((D0, N * (N + 1) // 2))
        X1 = xp.zeros((D1, N * (N + 1) // 2))
        Xq = xp.zeros((16, N * (N - 1) // 2))
        # X0,X1=transform_coeff(II,Lmax,N); Xq=xp.zeros((16,N*(N-1))); Xq[0,:]=1;
        S0 = xp.copy(I0)
        S1 = xp.copy(I1)
        Sq = xp.zeros(Xq.shape)
        # S0=xp.zeros(X0.shape); S1=xp.zeros(X1.shape); Sq=xp.zeros(Xq.shape)

        # return C0,C1,normC,AEq,bEq,AEqAEqtinv,AI_mat,bI,Lambda,d0,d1,D0,D1,idx_diag,idx_offdiag,IDX_upper,IDX_lower,X0,X1,Xq,S0,S1,Sq
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
            Pk = np.eye(dk)
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
            W0k = np.zeros((Ngrid, k, k))
            W1k = np.zeros((Ngrid, k + 1, k + 1))

            TkT = TT[k - start].T
            TinvkT = TTI[k - start].T

            w = np.real(TkT @ self.WD(k, SO3_grid).conj() @ TinvkT)
            W0k, W1k = permutek_block(w, k)

            W0.append(W0k)
            W1.append(W1k)
        return W0, W1

    def discretize_SO3(self):
        S2 = loadmat("design20.mat")["design"]
        S2_size = S2.shape[0]

        # discretize S1
        S1_size = round(np.sqrt(np.pi * S2_size))
        alpha = np.linspace(0, 2 * np.pi, S1_size)

        # discretize S2
        gamma, beta, _ = cart2sph(S2[:, 0], S2[:, 1], S2[:, 2])
        beta = np.pi / 2 - beta
        gamma = gamma + np.pi

        # SO(3) in Euler ZYZ
        SO3 = np.zeros((S2_size * S1_size, 3))
        count = 0
        for i in range(S1_size):
            for j in range(S2_size):
                SO3[count] = [alpha[i], beta[j], gamma[j]]
                count += 1

        return SO3

    #########################
    # Euler Estimation Step #
    #########################

    def euler_est(self, X1, XS):
        S = self.n_sym
        N = self.n_img
        sym_euler = np.zeros((S, 3))
        for s in range(S):
            sym_euler[s] = [2 * np.pi * s / S, 0, 0]
        [T, Tinv] = self.complex2real(1)
        X1 = np.kron(np.eye(N), T) @ X1 @ np.kron(np.eye(N), Tinv)
        [T, Tinv] = self.complex2real(S)
        XS = np.kron(np.eye(N), T) @ XS @ np.kron(np.eye(N), Tinv)

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
            B1 = np.zeros((N, N))
            B2 = np.zeros((N, N))
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
            ZZbar = np.zeros((N, N), dtype=complex)
            ZZ = np.zeros((N, N), dtype=complex)
            for i in range(N):
                for j in range(N):
                    z = X1[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)][0, 0]
                    ZZbar[i, j] = z / abs(z)
                    # ZZbar[i,j]=z*2/sin(beta[i])/sin(beta[j])
                    z = X1[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)][0, 2]
                    ZZ[i, j] = -z / abs(z)
                    # ZZ[i,j]=-z*2/sin(beta[i])/sin(beta[j])
            evals, evecs = np.linalg.eigh(ZZbar)
            idx = np.argmax(abs(evals))
            Z = evecs[:, idx] * np.sqrt(abs(evals[idx]))

            c = find_phase(Z[:, None] @ Z[:, None].T, ZZ)
            Z = np.sqrt(c) * Z
            return np.angle(Z)

        dk = 2 * S + 1

        def find_gamma(Xm, beta, alpha):
            C = np.zeros((N, N), dtype=complex)
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
            C += C.T.conj() + np.eye(N)
            evals, evecs = np.linalg.eigh(C)
            idx = np.argmax(evals)
            c = evecs[:, idx] * np.sqrt(evals[idx])
            return np.angle(c) / S

        Euler_est = np.zeros((N, 3))
        Euler_est[:, 0] = find_alpha(X1)
        Euler_est[:, 1] = find_beta(X1)
        Euler_est[:, 2] = find_gamma(XS, Euler_est[:, 1], Euler_est[:, 0])
        R_est = np.zeros((N, 3, 3))
        for n in range(N):
            R_est[n] = spr.from_euler("zyz", np.flip(Euler_est[n])).as_matrix().T
            # note that the order of alpha and gamma are swapped due to convention

        return R_est, Euler_est

    ####################
    # Helper Functions #
    ####################
    def transform_coeff(self, A, Lmax, N, IDX_upper):
        d0 = [0]
        d1 = [0]
        for k in range(1, Lmax + 1):
            d0.append(d0[-1] + k**2)
            d1.append(d1[-1] + (k + 1) ** 2)
        A0 = xp.zeros((d0[-1], N * (N + 1) // 2))
        A1 = xp.zeros((d1[-1], N * (N + 1) // 2))
        for k in range(1, Lmax + 1):
            a0, a1 = self.permutek(A[k - 1], k, N)
            A0[d0[k - 1] : d0[k], :] = self.vec_block(a0, N, k, IDX_upper)
            A1[d1[k - 1] : d1[k], :] = self.vec_block(a1, N, k + 1, IDX_upper)
        return A0, A1

    def transform_coeff_back(self, A0, A1, Lmax, N, IDX_upper, IDX_lower, idx_offdiag):
        d0 = [0]
        d1 = [0]
        for k in range(1, Lmax + 1):
            d0.append(d0[-1] + k**2)
            d1.append(d1[-1] + (k + 1) ** 2)
        A = []
        for k in range(1, Lmax + 1):
            dk = 2 * k + 1
            Ak = xp.zeros((N * dk, N * dk))
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
        AkP = xp.copy(Ak)
        dk = 2 * k + 1
        Pk = xp.eye(dk)
        for m in range(k):
            for n in range(k - m):
                Pk[(m + 2 * n, m + 2 * n + 1), :] = Pk[(m + 2 * n + 1, m + 2 * n), :]
        AkP = xp.kron(xp.eye(N), Pk) @ Ak @ xp.kron(xp.eye(N), Pk.T)

        Pk = xp.eye(N * dk)
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
        dk = 2 * k + 1
        Pk = xp.eye(N * dk)
        idx = xp.concatenate((xp.arange(dk - k, dk), xp.arange(k + 1)))
        for m in range(N - 1):
            for n in range(N - 1 - m):
                Pk[k * (m + 1) + n * dk : k * (m + 1) + (n + 1) * dk] = Pk[
                    k * (m + 1) + n * dk : k * (m + 1) + (n + 1) * dk
                ][idx, :]
        AkB = Pk.T @ Ak @ Pk
        dk = 2 * k + 1
        Pk = xp.eye(dk)
        for m in range(k):
            for n in range(k - m):
                Pk[(m + 2 * n, m + 2 * n + 1), :] = Pk[(m + 2 * n + 1, m + 2 * n), :]
        AkB = xp.kron(xp.eye(N), Pk.T) @ AkB @ xp.kron(xp.eye(N), Pk)
        return AkB

    @staticmethod
    def vec_block(A, N, sz, IDX_upper):
        vecA = (A.reshape(N, sz, N, sz).transpose(0, 2, 3, 1)).reshape(N**2, sz**2).T
        return vecA[:, IDX_upper]

    @staticmethod
    def largest_eigenvalue(AI, Ngrid, N):
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
        # Lambda=xp.linalg.eigvalsh(AI@AI.T)[-1]; print(Lambda)
        return Lambda

    def compute_rank(self, Lmax):
        rk = xp.zeros(Lmax)
        A = []
        for k in range(1, Lmax + 1):
            Ak = np.sum(self.WD(k, self.sym_euler), axis=0)
            Ak = np.round(Ak / self.n_sym, 6)
            A.append(Ak)
            rk[k - 1] = np.linalg.matrix_rank(Ak)
        return rk, A

    def WD(self, J, euler):
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
        # compute Wigner small d matrix
        d = np.zeros((len(beta), 2 * J + 1, 2 * J + 1))
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
        tmp = vecA.T.reshape(N * (N + 1) // 2, sz, sz).transpose(0, 2, 1)
        AA = xp.zeros((N**2, sz, sz))
        AA[IDX_upper] = tmp
        AA[IDX_lower] = tmp[idx_offdiag].transpose(0, 2, 1)
        return (AA.reshape(N, N, sz, sz).transpose(0, 2, 1, 3)).reshape(N * sz, N * sz)

    @staticmethod
    def psd_projection(B):
        # compute the PSD part of a symmstric matrix
        evals, evecs = xp.linalg.eigh((B + B.T) / 2)
        evals = xp.maximum(evals, 0)
        return (evecs * evals) @ evecs.T

    @staticmethod
    def transform_block(A, k, Pk=None):
        if Pk is None:
            dk = 2 * k + 1
            Pk = xp.eye(dk)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
        AT = Pk @ A @ Pk.T
        return AT[:k, :k].T.reshape(-1), AT[k:, k:].T.reshape(-1)

    @staticmethod
    def transform_back_block(A0, A1, k, Pk=None):
        dk = 2 * k + 1
        A = xp.zeros((dk, dk))
        A[:k, :k] = A0.reshape(k, k).T
        A[k:, k:] = A1.reshape(k + 1, k + 1).T
        if Pk is None:
            Pk = xp.eye(dk)
            for m in range(k):
                for el in range(k - m):
                    Pk[(m + 2 * el, m + 2 * el + 1), :] = Pk[
                        (m + 2 * el + 1, m + 2 * el), :
                    ]
        return Pk.T @ A @ Pk
