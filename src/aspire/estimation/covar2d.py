import logging
from scipy.linalg import sqrtm
from scipy.linalg import solve
from numpy.linalg import inv
from sys import float_info

from aspire.utils.matlab_compat import m_reshape
from aspire.utils.blk_diag_func import *
from aspire.utils.matrix import shrink_covar
from aspire.utils.optimize import conj_grad
from aspire.utils import ensure
from aspire.utils.filters import RadialCTFFilter

logger = logging.getLogger(__name__)


class RotCov2D:
    """
    Define a class for performing Cov2D analysis with CTF information described in

    T. Bhamre, T. Zhang, and A. Singer, "Denoising and covariance estimation of single particle cryo-EM images",
    J. Struct. Biol. 195, 27-81 (2016). DOI: 10.1016/j.jsb.2016.04.013
    """

    def __init__(self, basis):
        """
        constructor of an object for 2D covariance analysis
        """
        self.basis = basis
        ensure(basis.ndim == 2, 'Only two-dimensional basis functions are needed.')
        self.dtype = 'double'

    def _get_mean(self, coeffs):
        """
        Calculate the mean vector from the expansion coefficients of 2D images without CTF information.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) to be averaged.
        :return: The mean value vector for all images.
        """
        if coeffs.size == 0:
            raise RuntimeError('The coefficients need to be calculated first!')
        self.dtype = coeffs.dtype
        mask = self.basis._indices["ells"] == 0
        mean_coeff = np.zeros((self.basis.count, 1), dtype=self.dtype)
        mean_coeff[mask, 0] = np.mean(coeffs[mask, ...], axis=1)

        return mean_coeff.flatten()

    def _get_covar(self, coeffs, mean_coeff=None,  do_refl=True):
        """
        Calculate the covariance matrix from the expansion coefficients without CTF information.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) calculated from 2D images.
        :param mean_coeff: The mean vector calculated from the `coeffs`.
        :param do_refl: If true, enforce invariance to reflection (default false).
        :return: The covariance matrix of coefficients for all images.
        """
        if coeffs.size == 0:
            raise RuntimeError('The coefficients need to be calculated first!')
        if mean_coeff is None:
            mean_coeff = self._get_mean(coeffs)
        mean_coeff = mean_coeff.reshape((self.basis.count, 1))

        covar_coeff = []
        ind = 0
        ell = 0
        mask = self.basis._indices["ells"] == ell
        coeff_ell = coeffs[mask, ...] - mean_coeff[mask, ...]
        covar_ell = np.array(coeff_ell @ coeff_ell.T/np.size(coeffs, 1))
        covar_coeff.append(covar_ell)
        ind += 1

        for ell in range(1, self.basis.ell_max+1):
            mask = self.basis._indices["ells"] == ell
            mask_pos = [mask[i] and (self.basis._indices['sgns'][i] == +1) for i in range(len(mask))]
            mask_neg = [mask[i] and (self.basis._indices['sgns'][i] == -1) for i in range(len(mask))]
            covar_ell_diag = np.array(coeffs[mask_pos, :] @ coeffs[mask_pos, :].T +
                coeffs[mask_neg, :] @ coeffs[mask_neg, :].T) / (2 * np.size(coeffs, 1))

            if do_refl:
                covar_coeff.append(covar_ell_diag)
                covar_coeff.append(covar_ell_diag)
                ind = ind + 2
            else:
                covar_ell_off = np.array((coeffs[mask_pos, :] @ coeffs[mask_neg, :].T / np.size(coeffs, 1) -
                                 coeffs[mask_neg, :] @ coeffs[mask_pos, :].T)/(2 * np.size(coeffs, 1)))
                hsize = np.size(covar_ell_diag, 0)
                covar_coeff_blk = np.zeros((2 * hsize, 2 * hsize))

                fsize = np.size(covar_coeff_blk, 0)
                covar_coeff_blk[0:hsize, 0:hsize] = covar_ell_diag[0:hsize, 0:hsize]
                covar_coeff_blk[hsize:fsize, hsize:fsize] = covar_ell_diag[0:hsize, 0:hsize]
                covar_coeff_blk[0:hsize, hsize:fsize] = covar_ell_off[0:hsize, 0:hsize]
                covar_coeff_blk[hsize:fsize, 0:hsize] = covar_ell_off.T[0:hsize, 0:hsize]
                covar_coeff.append(covar_coeff_blk)
                ind = ind + 1

        return covar_coeff

    def get_mean(self, coeffs, ctf_fb=None, ctf_idx=None):
        """
        Calculate the mean vector from the expansion coefficients with CTF information.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) to be averaged.
        :param ctf_fb: The CFT functions in the FB expansion.
        :param ctf_idx: An array of the CFT function indices for all 2D images.
            If ctf_fb or ctf_idx is None, the identity filter will be applied.
        :return: The mean value vector for all images.
        """
        if coeffs.size == 0:
            raise RuntimeError('The coefficients need to be calculated!')

        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[1], dtype=int)
            ctf_fb = [blk_diag_eye(blk_diag_partition(RadialCTFFilter().fb_mat(self.basis)))]

        b = np.zeros((self.basis.count, 1), dtype=self.dtype)

        A = blk_diag_zeros(blk_diag_partition(ctf_fb[0]), dtype=self.dtype)
        for k in np.unique(ctf_idx[:]).T:
            coeff_k = coeffs[:, ctf_idx == k]
            weight = np.size(coeff_k, 1)/np.size(coeffs, 1)
            mean_coeff_k = self._get_mean(coeff_k).reshape((self.basis.count, 1))
            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            b = b + weight*blk_diag_apply(ctf_fb_k_t, mean_coeff_k)
            A = blk_diag_add(A, blk_diag_mult(weight, blk_diag_mult(ctf_fb_k_t, ctf_fb_k)))

        mean_coeff = blk_diag_solve(A, b)

        return mean_coeff.flatten()

    def get_covar(self, coeffs, ctf_fb=None, ctf_idx=None, mean_coeff=None,
                  do_refl=True, noise_var=1, covar_est_opt=None):
        """
        Calculate the covariance matrix from the expansion coefficients and CTF information.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) to be calculated.
        :param ctf_fb: The CFT functions in the FB expansion.
        :param ctf_idx: An array of the CFT function indices for all 2D images.
            If ctf_fb or ctf_idx is None, the identity filter will be applied.
        :param mean_coeff: The mean value vector from all images.
        :param noise_var: The estimated variance of noise. The value should be zero for `coeffs`
            from clean images of simulation data.
        :param covar_est_opt: The optimization parameter list for obtaining the Cov2D matrix.
        :return: The basis coefficients of the covariance matrix in
            the form of cell array representing a block diagonal matrix. These
            block diagonal matrices may be manipulated using the `blk_diag_*` functions.
            The covariance is calculated from the images represented by the coeffs array,
            along with all possible rotations and reflections. As a result, the computed covariance
            matrix is invariant to both reflection and rotation. The effect of the filters in ctf_fb
            are accounted for and inverted to yield a covariance estimate of the unfiltered images.
        """

        if coeffs.size == 0:
            raise RuntimeError('The coefficients need to be calculated!')

        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[1], dtype=int)
            ctf_fb = [blk_diag_eye(blk_diag_partition(RadialCTFFilter().fb_mat(self.basis)))]

        def identity(x):
            return x

        if covar_est_opt is None:
            covar_est_opt = {'shrinker': 'None', 'verbose': 0, 'max_iter': 250, 'iter_callback': [],
                             'store_iterates': False, 'rel_tolerance': 1e-12, 'precision': 'float64',
                             'preconditioner': 'identity'}

        if mean_coeff is None:
            mean_coeff = self.get_mean(coeffs, ctf_fb, ctf_idx)
        mean_coeff = mean_coeff.reshape((self.basis.count, 1))

        block_partition = blk_diag_partition(ctf_fb[0])
        b_coeff = blk_diag_zeros(block_partition, dtype=self.dtype)
        b_noise = blk_diag_zeros(block_partition, dtype=self.dtype)
        A = []
        for k in range(0, len(ctf_fb)):
            A.append(blk_diag_zeros(block_partition, dtype=self.dtype))

        M = blk_diag_zeros(block_partition, dtype=self.dtype)

        for k in np.unique(ctf_idx[:]):

            coeff_k = coeffs[:, ctf_idx == k]
            weight = np.size(coeff_k, 1)/np.size(coeffs, 1)

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            mean_coeff_k = blk_diag_apply(ctf_fb_k, mean_coeff)
            covar_coeff_k = self._get_covar(coeff_k, mean_coeff_k.flatten())

            b_coeff = blk_diag_add(b_coeff, blk_diag_mult(ctf_fb_k_t,
                blk_diag_mult(covar_coeff_k, blk_diag_mult(ctf_fb_k, weight))))

            b_noise = blk_diag_add(b_noise, blk_diag_mult(weight,
                blk_diag_mult(ctf_fb_k_t, ctf_fb_k)))

            A[k] = blk_diag_mult(ctf_fb_k_t, blk_diag_mult(ctf_fb_k, np.sqrt(weight)))

            M = blk_diag_add(M, A[k])

        if covar_est_opt['shrinker'] == 'None':
            b = blk_diag_add(b_coeff, blk_diag_mult(-noise_var, b_noise))
        else:
            b = self.shrink_covar_backward(b_coeff, b_noise, np.size(coeffs, 1),
                                           noise_var, covar_est_opt['shrinker'])

        cg_opt = covar_est_opt
        covar_coeff = blk_diag_zeros(block_partition, dtype=self.dtype)

        def precond_fun(S, x):
            p = np.size(S, 0)
            ensure(np.size(x) == p*p, 'The sizes of S and x are not consistent.')
            x = m_reshape(x, (p, p))
            y = S @ x @ S
            return y

        def apply(A, x):
            p = np.size(A[0], 0)
            x = m_reshape(x, (p, p))
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
        """
        Apply the shrinking method to the 2D covariance of coefficients.

        :param b: An input coefficient covariance.
        :param b_noise: The noise covariance.
        :param noise_var: The estimated variance of noise.
        :param shrinker: The shrinking method.
        :return: The shrinked 2D covariance coefficients.
        """
        b_out = b
        for ell in range(0, len(b)):
            b_ell = b[ell]
            p = np.size(b_ell, 1)
            S = sqrtm(b_noise[ell])
            # from Matlab b_ell = S \ b_ell /S
            b_ell = solve(S, b_ell) @ inv(S)
            b_ell = shrink_covar(b_ell, noise_var, p/n, shrinker)
            b_ell = S @ b_ell @ S
            b_out[ell] = b_ell
        return b_out

    def get_cwf_coeffs(self, coeffs, ctf_fb=None, ctf_idx=None, mean_coeff=None, covar_coeff=None, noise_var=1):
        """
        Estimate the expansion coefficients using the Covariance Wiener Filtering (CWF) method.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) to be calculated.
        :param ctf_fb: The CFT functions in the FB expansion.
        :param ctf_idx: An array of the CFT function indices for all 2D images.
            If ctf_fb or ctf_idx is None, the identity filter will be applied.
        :param mean_coeff: The mean value vector from all images.
        :param covar_coeff: The block diagonal covariance matrix of the clean coefficients represented by a cell array.
        :param noise_var: The estimated variance of noise. The value should be zero for `coeffs`
            from clean images of simulation data.
        :return: The estimated coefficients of the unfiltered images in certain math basis.
            These are obtained using a Wiener filter with the specified covariance for the clean images
            and white noise of variance `noise_var` for the noise.
        """
        if mean_coeff is None:
            mean_coeff = self.get_mean(coeffs, ctf_fb, ctf_idx)
        mean_coeff = mean_coeff.reshape((self.basis.count, 1))

        if covar_coeff is None:
            covar_coeff = self.get_covar(coeffs, ctf_fb, ctf_idx, mean_coeff, noise_var=noise_var)

        blk_partition = blk_diag_partition(covar_coeff)

        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[1], dtype=int)
            ctf_fb = [blk_diag_eye(blk_partition)]

        noise_covar_coeff = blk_diag_mult(noise_var, blk_diag_eye(blk_partition, dtype=self.dtype))

        coeffs_est = np.zeros_like(coeffs, dtype=self.dtype)

        for k in np.unique(ctf_idx[:]):
            coeff_k = coeffs[:, ctf_idx == k]
            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            sig_covar_coeff = blk_diag_mult(ctf_fb_k, blk_diag_mult(covar_coeff, ctf_fb_k_t))
            sig_noise_covar_coeff = blk_diag_add(sig_covar_coeff, noise_covar_coeff)

            mean_coeff_k = blk_diag_apply(ctf_fb_k, mean_coeff)

            coeff_est_k = coeff_k - mean_coeff_k
            coeff_est_k = blk_diag_solve(sig_noise_covar_coeff, coeff_est_k)
            coeff_est_k = blk_diag_apply(blk_diag_mult(covar_coeff, ctf_fb_k_t), coeff_est_k)
            coeff_est_k = coeff_est_k + mean_coeff
            coeffs_est[:, ctf_idx == k] = coeff_est_k

        return coeffs_est
