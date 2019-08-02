import logging
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import solve
from numpy.linalg import inv

from aspire.utils import ensure
from aspire.utils.matlab_compat import m_reshape

from aspire.utils.blk_diag_func import *
from aspire.utils.cell import Cell2D

from aspire.utils.optimize import fill_struct
from aspire.utils.optimize import conj_grad
from aspire.denoise.covar2d import RotCov2D


logger = logging.getLogger(__name__)

class Cov2DCTF(RotCov2D):
    """
    Define a derived class for denoising 2D images using CTF and Wiener Cov2D method
    """

    def get_mean_ctf(self, coeffs, ctf_fb, ctf_idx):
        """
        Calculate the mean vector from the expansion coefficient.
        param b_coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        :return: The mean value vector for all images.
        """
        if coeffs is None:
            raise RuntimeError('The coefficients need to be calculated!')

        b = np.zeros((self.basis.basis_count, 1), dtype=self.as_type)

        A = blk_diag_zeros(blk_diag_partition(ctf_fb[0]), dtype=self.as_type)
        for k in np.unique(ctf_idx[:]).T:
            coeff_k = coeffs[:, ctf_idx == k]
            weight = np.size(coeff_k, 1)/np.size(coeffs, 1)
            mean_coeff_k = self.get_mean(coeff_k)
            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            b = b + weight*blk_diag_apply(ctf_fb_k_t, mean_coeff_k)
            A = blk_diag_add(A, blk_diag_mult(weight, blk_diag_mult(ctf_fb_k_t, ctf_fb_k)))

        mean_coeff = blk_diag_solve(A, b)

        return mean_coeff

    def get_covar_ctf(self, coeffs, ctf_fb, ctf_idx, mean_coeff=None, noise_var=1, covar_est_opt=None):

        def identity(x):
            return x

        if covar_est_opt is None:
            covar_est_opt = {'shrinker': 'none', 'verbose': 0, 'max_iter': 250, 'iter_callback': [],
                             'store_iterates': False, 'rel_tolerance': 1e-12, 'precision': 'float64',
                             'preconditioner': 'identity'}

        if mean_coeff is None:
            mean_coeff = self.get_mean_ctf(coeffs, ctf_fb, ctf_idx)

        block_partition = blk_diag_partition(ctf_fb[0])
        b_coeff = blk_diag_zeros(block_partition, dtype=self.as_type)
        b_noise = blk_diag_zeros(block_partition, dtype=self.as_type)
        A = []
        for k in range(0, len(ctf_fb)):
            A.append(blk_diag_zeros(block_partition, dtype=self.as_type))

        M = blk_diag_zeros(block_partition, dtype=self.as_type)

        for k in np.unique(ctf_idx[:]):

            coeff_k = coeffs[:, ctf_idx == k]
            weight = np.size(coeff_k, 1)/np.size(coeffs, 1)

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            mean_coeff_k = blk_diag_apply(ctf_fb_k, mean_coeff)
            covar_coeff_k = self.get_covar(coeff_k, mean_coeff_k)

            b_coeff = blk_diag_add(b_coeff, blk_diag_mult(ctf_fb_k_t,
                blk_diag_mult(covar_coeff_k, blk_diag_mult(ctf_fb_k, weight))))

            b_noise = blk_diag_add(b_noise, blk_diag_mult(weight,
                blk_diag_mult(ctf_fb_k_t, ctf_fb_k)))

            A[k] = blk_diag_mult(ctf_fb_k_t, blk_diag_mult(ctf_fb_k, np.sqrt(weight)))

            M = blk_diag_add(M, A[k])

        if covar_est_opt['shrinker'] == 'none':
            b = blk_diag_add(b_coeff, blk_diag_mult(-noise_var, b_noise))
        else:
            b = self.shrink_covar_backward(b_coeff, b_noise, np.size(coeffs, 1),
                                           noise_var, covar_est_opt['shrinker'])

        cg_opt = covar_est_opt
        covar_coeff = blk_diag_zeros(block_partition, dtype=self.as_type)

        def precond_fun(S, x):
            p = np.size(S, 0)
            x = m_reshape(x, p*np.ones((1, 2)))
            y = S @ x @ S
            return y

        def apply(A, x):
            p = np.size(A[0], 0)
            x = m_reshape(x, p*np.ones((1, 2)))
            y = np.zeros_like(x)
            for k in range(0, len(A)):
                    y = y + A[k] @ x @ A[k].T
            return y

        for ell in range(0, len(b)):
            A_ell = []
            for k in range(0, len(A)):
                A_ell.append(A[k][ell])
            b_ell = b[ell]
            S = inv(M[ell])
            cg_opt["preconditioner"] = lambda x: precond_fun(S, x)
            covar_coeff[ell], _, _ = conj_grad(lambda x: apply(A_ell, x), b_ell, cg_opt)

        return covar_coeff

    def shrink_covar_backward(self, b, b_noise, n, noise_var, shrinker):
        b_out = b
        for ell in range(0, b.size()):
            b_ell = b[ell]
            p = np.size(b_ell, 1)
            S = sqrtm(b_noise[ell])
            # from Matlab b_ell = S \ b_ell /S
            b_ell = np.divide(solve(S,b_ell), S)
            b_ell = shrink_covar(b_ell, noise_var, p/n, shrinker)
            b_ell = S @ b_ell @ S
            b_out[ell] = b_ell
        return b_out

    def get_wiener_ctf(self, coeffs, ctf_fb, ctf_idx, mean_coeff=None, covar_coeff=None, noise_var=1):
        if mean_coeff is None:
            mean_coeff = self.get_mean_ctf(coeffs, ctf_fb, ctf_idx)
        if covar_coeff is None:
            covar_coeff = self.get_covar_ctf(coeffs, ctf_fb, ctf_idx, mean_coeff, noise_var=noise_var)

        blk_partition = blk_diag_partition(ctf_fb[0])

        noise_covar_coeff = blk_diag_mult(noise_var, blk_diag_eye(blk_partition, dtype=self.as_type))

        coeffs_est = np.zeros_like(coeffs, dtype=self.as_type)

        for k in np.unique(ctf_idx[:]):
            coeff_k = coeffs[:, ctf_idx == k]
            ctf_fb_k = ctf_fb[k]
            print("ctf_fb_k", ctf_fb_k)
            print("covar_coeff", covar_coeff)
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            sig_covar_coeff = blk_diag_mult(ctf_fb_k, blk_diag_mult(covar_coeff, ctf_fb_k_t))
            sig_noise_covar_coeff = blk_diag_add(sig_covar_coeff, noise_covar_coeff)

            mean_coeff_k = blk_diag_apply(ctf_fb_k, mean_coeff)

            coeff_est_k = coeff_k - mean_coeff_k   # blk_diag_minus(coeff_k, mean_coeff_k)
            coeff_est_k = blk_diag_solve(sig_noise_covar_coeff, coeff_est_k)
            coeff_est_k = blk_diag_apply(blk_diag_mult(covar_coeff, ctf_fb_k_t), coeff_est_k)
            coeff_est_k = coeff_est_k + mean_coeff  # = blk_dia_add(coeff_est_k + mean_coeff)

            coeffs_est[:, ctf_idx == k] = coeff_est_k

        return coeffs_est
