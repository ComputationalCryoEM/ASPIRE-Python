import logging

import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import solve, sqrtm

from aspire.operators import BlkDiagMatrix, RadialCTFFilter
from aspire.optimization import conj_grad, fill_struct
from aspire.utils import ensure, make_symmat
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


def shrink_covar(covar_in, noise_var, gamma, shrinker=None):
    """
    Shrink the covariance matrix
    :param covar_in: An input covariance matrix
    :param noise_var: The estimated variance of noise
    :param gamma: An input parameter to specify the maximum values of eigen values to be neglected.
    :param shrinker: An input parameter to select different shrinking methods.
    :return: The shrinked covariance matrix
    """

    if shrinker is None:
        shrinker = "frobenius_norm"
    ensure(
        shrinker in ("frobenius_norm", "operator_norm", "soft_threshold"),
        "Unsupported shrink method",
    )

    covar = covar_in / noise_var

    lambs, eig_vec = eig(make_symmat(covar))

    lambda_max = (1 + np.sqrt(gamma)) ** 2

    lambs[lambs < lambda_max] = 0

    if shrinker == "operator_norm":
        lambdas = lambs[lambs > lambda_max]
        lambdas = (
            1
            / 2
            * (lambdas - gamma + 1 + np.sqrt((lambdas - gamma + 1) ** 2 - 4 * lambdas))
            - 1
        )
        lambs[lambs > lambda_max] = lambdas
    elif shrinker == "frobenius_norm":
        lambdas = lambs[lambs > lambda_max]
        lambdas = (
            1
            / 2
            * (lambdas - gamma + 1 + np.sqrt((lambdas - gamma + 1) ** 2 - 4 * lambdas))
            - 1
        )
        c = np.divide(
            (1 - np.divide(gamma, lambdas ** 2)), (1 + np.divide(gamma, lambdas))
        )
        lambdas = lambdas * c
        lambs[lambs > lambda_max] = lambdas
    else:
        # for the case of shrinker == 'soft_threshold'
        lambdas = lambs[lambs > lambda_max]
        lambs[lambs > lambda_max] = lambdas - lambda_max

    diag_lambs = np.zeros_like(covar)
    np.fill_diagonal(diag_lambs, lambs)

    shrinked_covar = eig_vec @ diag_lambs @ eig_vec.conj().T
    shrinked_covar *= noise_var

    return shrinked_covar


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
        self.dtype = self.basis.dtype
        ensure(basis.ndim == 2, "Only two-dimensional basis functions are needed.")

    def _get_mean(self, coeffs):
        """
        Calculate the mean vector from the expansion coefficients of 2D images without CTF information.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) to be averaged.
        :return: The mean value vector for all images.
        """
        if coeffs.size == 0:
            raise RuntimeError("The coefficients need to be calculated first!")
        mask = self.basis._indices["ells"] == 0
        mean_coeff = np.zeros(self.basis.count, dtype=coeffs.dtype)
        mean_coeff[mask] = np.mean(coeffs[..., mask], axis=0)

        return mean_coeff

    def _get_covar(self, coeffs, mean_coeff=None, do_refl=True):
        """
        Calculate the covariance matrix from the expansion coefficients without CTF information.

        :param coeffs: A coefficient vector (or an array of coefficient vectors) calculated from 2D images.
        :param mean_coeff: The mean vector calculated from the `coeffs`.
        :param do_refl: If true, enforce invariance to reflection (default false).
        :return: The covariance matrix of coefficients for all images.
        """
        if coeffs.size == 0:
            raise RuntimeError("The coefficients need to be calculated first!")
        if mean_coeff is None:
            mean_coeff = self._get_mean(coeffs)

        # Initialize a totally empty BlkDiagMatrix, build incrementally.
        covar_coeff = BlkDiagMatrix.empty(0, dtype=coeffs.dtype)
        ell = 0
        mask = self.basis._indices["ells"] == ell
        coeff_ell = coeffs[..., mask] - mean_coeff[mask]
        covar_ell = np.array(coeff_ell.T @ coeff_ell / coeffs.shape[0])
        covar_coeff.append(covar_ell)

        for ell in range(1, self.basis.ell_max + 1):
            mask = self.basis._indices["ells"] == ell
            mask_pos = [
                mask[i] and (self.basis._indices["sgns"][i] == +1)
                for i in range(len(mask))
            ]
            mask_neg = [
                mask[i] and (self.basis._indices["sgns"][i] == -1)
                for i in range(len(mask))
            ]
            covar_ell_diag = np.array(
                coeffs[:, mask_pos].T @ coeffs[:, mask_pos]
                + coeffs[:, mask_neg].T @ coeffs[:, mask_neg]
            ) / (2 * coeffs.shape[0])

            if do_refl:
                covar_coeff.append(covar_ell_diag)
                covar_coeff.append(covar_ell_diag)
            else:
                covar_ell_off = np.array(
                    (
                        coeffs[:, mask_pos] @ coeffs[:, mask_neg].T / coeffs.shape[0]
                        - coeffs[:, mask_pos].T @ coeffs[:, mask_neg]
                    )
                    / (2 * coeffs.shape[0])
                )

                hsize = covar_ell_diag.shape[0]
                covar_coeff_blk = np.zeros((2, hsize, 2, hsize))

                covar_coeff_blk[0:2, :, 0:2, :] = covar_ell_diag[:hsize, :hsize]
                covar_coeff_blk[0, :, 1, :] = covar_ell_off[:hsize, :hsize]
                covar_coeff_blk[1, :, 0, :] = covar_ell_off.T[:hsize, :hsize]

                covar_coeff.append(covar_coeff_blk.reshape(2 * hsize, 2 * hsize))

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
            raise RuntimeError("The coefficients need to be calculated!")

        # should assert we require none or both...
        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[0], dtype=int)
            ctf_fb = [BlkDiagMatrix.eye_like(RadialCTFFilter().fb_mat(self.basis))]

        b = np.zeros(self.basis.count, dtype=coeffs.dtype)

        A = BlkDiagMatrix.zeros_like(ctf_fb[0])
        for k in np.unique(ctf_idx[:]).T:
            coeff_k = coeffs[ctf_idx == k]
            weight = coeff_k.shape[0] / coeffs.shape[0]
            mean_coeff_k = self._get_mean(coeff_k)

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = ctf_fb_k.T
            b += weight * ctf_fb_k_t.apply(mean_coeff_k)
            A += weight * (ctf_fb_k_t @ ctf_fb_k)

        mean_coeff = A.solve(b)
        return mean_coeff

    def get_covar(
        self,
        coeffs,
        ctf_fb=None,
        ctf_idx=None,
        mean_coeff=None,
        do_refl=True,
        noise_var=1,
        covar_est_opt=None,
    ):
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
            block diagonal matrices are implemented as BlkDiagMatrix instances.
            The covariance is calculated from the images represented by the coeffs array,
            along with all possible rotations and reflections. As a result, the computed covariance
            matrix is invariant to both reflection and rotation. The effect of the filters in ctf_fb
            are accounted for and inverted to yield a covariance estimate of the unfiltered images.
        """

        if coeffs.size == 0:
            raise RuntimeError("The coefficients need to be calculated!")

        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[0], dtype=int)
            ctf_fb = [BlkDiagMatrix.eye_like(RadialCTFFilter().fb_mat(self.basis))]

        def identity(x):
            return x

        default_est_opt = {
            "shrinker": "None",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
            "preconditioner": identity,
        }

        covar_est_opt = fill_struct(covar_est_opt, default_est_opt)

        if mean_coeff is None:
            mean_coeff = self.get_mean(coeffs, ctf_fb, ctf_idx)

        b_coeff = BlkDiagMatrix.zeros_like(ctf_fb[0])
        b_noise = BlkDiagMatrix.zeros_like(ctf_fb[0])
        A = []
        for _ in range(len(ctf_fb)):
            A.append(BlkDiagMatrix.zeros_like(ctf_fb[0]))

        M = BlkDiagMatrix.zeros_like(ctf_fb[0])

        for k in np.unique(ctf_idx[:]):

            coeff_k = coeffs[ctf_idx == k]
            weight = coeff_k.shape[0] / coeffs.shape[0]

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = ctf_fb_k.T
            mean_coeff_k = ctf_fb_k.apply(mean_coeff)
            covar_coeff_k = self._get_covar(coeff_k, mean_coeff_k)

            b_coeff += weight * (ctf_fb_k_t @ covar_coeff_k @ ctf_fb_k)

            ctf_fb_k_sq = ctf_fb_k_t @ ctf_fb_k
            b_noise += weight * ctf_fb_k_sq

            A[k] = np.sqrt(weight) * ctf_fb_k_sq
            M += A[k]

        if covar_est_opt["shrinker"] == "None":
            b = b_coeff - noise_var * b_noise
        else:
            b = self.shrink_covar_backward(
                b_coeff,
                b_noise,
                np.size(coeffs, 1),
                noise_var,
                covar_est_opt["shrinker"],
            )

        # RCOPT okay, this looks like a big batch, come back later

        cg_opt = covar_est_opt

        covar_coeff = BlkDiagMatrix.zeros_like(ctf_fb[0])

        def precond_fun(S, x):
            p = np.size(S, 0)
            ensure(np.size(x) == p * p, "The sizes of S and x are not consistent.")
            x = m_reshape(x, (p, p))
            y = S @ x @ S
            y = m_reshape(y, (p ** 2,))
            return y

        def apply(A, x):
            p = np.size(A[0], 0)
            x = m_reshape(x, (p, p))
            y = np.zeros_like(x)
            for k in range(0, len(A)):
                y = y + A[k] @ x @ A[k].T
            y = m_reshape(y, (p ** 2,))
            return y

        for ell in range(0, len(b)):
            A_ell = []
            for k in range(0, len(A)):
                A_ell.append(A[k][ell])
            p = np.size(A_ell[0], 0)
            b_ell = m_reshape(b[ell], (p ** 2,))
            S = inv(M[ell])
            cg_opt["preconditioner"] = lambda x: precond_fun(S, x)
            covar_coeff_ell, _, _ = conj_grad(lambda x: apply(A_ell, x), b_ell, cg_opt)
            covar_coeff[ell] = m_reshape(covar_coeff_ell, (p, p))

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
            b_ell = shrink_covar(b_ell, noise_var, p / n, shrinker)
            b_ell = S @ b_ell @ S
            b_out[ell] = b_ell
        return b_out

    def get_cwf_coeffs(
        self,
        coeffs,
        ctf_fb=None,
        ctf_idx=None,
        mean_coeff=None,
        covar_coeff=None,
        noise_var=1,
    ):
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

        if covar_coeff is None:
            covar_coeff = self.get_covar(
                coeffs, ctf_fb, ctf_idx, mean_coeff, noise_var=noise_var
            )

        # should be none or both
        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[0], dtype=int)
            ctf_fb = [BlkDiagMatrix.eye_like(covar_coeff)]

        noise_covar_coeff = noise_var * BlkDiagMatrix.eye_like(covar_coeff)

        coeffs_est = np.zeros_like(coeffs)

        for k in np.unique(ctf_idx[:]):
            coeff_k = coeffs[ctf_idx == k]
            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = ctf_fb_k.T
            sig_covar_coeff = ctf_fb_k @ covar_coeff @ ctf_fb_k_t

            sig_noise_covar_coeff = sig_covar_coeff + noise_covar_coeff

            mean_coeff_k = ctf_fb_k.apply(mean_coeff)

            coeff_est_k = coeff_k - mean_coeff_k
            coeff_est_k = sig_noise_covar_coeff.solve(coeff_est_k.T).T
            coeff_est_k = (covar_coeff @ ctf_fb_k_t).apply(coeff_est_k.T).T
            coeff_est_k = coeff_est_k + mean_coeff
            coeffs_est[ctf_idx == k] = coeff_est_k

        return coeffs_est


class BatchedRotCov2D(RotCov2D):
    """
    Perform batchwise rotationally equivariant 2D covariance estimation from an
    `ImageSource` objects. This is done with a single pass through the data,
    processing moderately-sized batches one at a time. The rotational
    equivariance is achieved by decomposing images in a steerable Fourier–Bessel
    basis. For more information, see

        T. Bhamre, T. Zhang, and A. Singer, "Denoising and covariance estimation
        of single particle cryo-EM images", J. Struct. Biol. 195, 27-81 (2016).
        DOI: 10.1016/j.jsb.2016.04.013

        :param src: The `ImageSource` object from which the sample images are to
        be extracted.
        :param basis: The `FBBasis2D` object used to decompose the images. By
        default, this is set to `FFBBasis2D((src.L, src.L))`.
        :param batch_size: The number of images to process at a time (default
        8192).
    """

    def __init__(self, src, basis=None, batch_size=8192):
        self.src = src
        self.basis = basis
        self.batch_size = batch_size
        self.dtype = self.src.dtype

        self.b_mean = None
        self.b_covar = None
        self.A_mean = None
        self.A_covar = None
        self.M_covar = None

        self._build()

    def _build(self):
        src = self.src

        if self.basis is None:
            from aspire.basis import FFBBasis2D

            self.basis = FFBBasis2D((src.L, src.L), dtype=self.dtype)

        if src.unique_filters is None:
            logger.info("CTF filters are not included in Cov2D denoising")
            # set all CTF filters to an identity filter
            self.ctf_idx = np.zeros(src.n, dtype=int)
            self.ctf_fb = [BlkDiagMatrix.eye_like(RadialCTFFilter().fb_mat(self.basis))]
        else:
            logger.info("Represent CTF filters in FB basis")
            unique_filters = src.unique_filters
            self.ctf_idx = src.filter_indices
            self.ctf_fb = [f.fb_mat(self.basis) for f in unique_filters]

    def _calc_rhs(self):
        src = self.src
        basis = self.basis

        ctf_fb = self.ctf_fb
        ctf_idx = self.ctf_idx

        zero_coeff = np.zeros((basis.count,), dtype=self.dtype)

        b_mean = [np.zeros(basis.count, dtype=self.dtype) for _ in ctf_fb]

        b_covar = BlkDiagMatrix.zeros_like(ctf_fb[0])

        for start in range(0, src.n, self.batch_size):
            batch = np.arange(start, min(start + self.batch_size, src.n))

            im = src.images(batch[0], len(batch))
            coeff = basis.evaluate_t(im.data)

            for k in np.unique(ctf_idx[batch]):
                coeff_k = coeff[ctf_idx[batch] == k]
                weight = np.size(coeff_k, 0) / src.n

                mean_coeff_k = self._get_mean(coeff_k)

                ctf_fb_k = ctf_fb[k]
                ctf_fb_k_t = ctf_fb_k.T

                b_mean_k = weight * ctf_fb_k_t.apply(mean_coeff_k)

                b_mean[k] += b_mean_k

                covar_coeff_k = self._get_covar(coeff_k, zero_coeff)

                b_covar_k = ctf_fb_k_t @ covar_coeff_k
                b_covar_k = b_covar_k @ ctf_fb_k
                b_covar_k *= weight

                b_covar += b_covar_k

        self.b_mean = b_mean
        self.b_covar = b_covar

    def _calc_op(self):
        src = self.src

        ctf_fb = self.ctf_fb
        ctf_idx = self.ctf_idx

        A_mean = BlkDiagMatrix.zeros_like(ctf_fb[0])
        A_covar = [None for _ in ctf_fb]
        M_covar = BlkDiagMatrix.zeros_like(ctf_fb[0])

        for k in np.unique(ctf_idx):
            weight = np.count_nonzero(ctf_idx == k) / src.n

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = ctf_fb_k.T

            ctf_fb_k_sq = ctf_fb_k_t @ ctf_fb_k
            A_mean_k = weight * ctf_fb_k_sq
            A_mean += A_mean_k

            A_covar_k = np.sqrt(weight) * ctf_fb_k_sq
            A_covar[k] = A_covar_k

            M_covar += A_covar_k

        self.A_mean = A_mean
        self.A_covar = A_covar
        self.M_covar = M_covar

    def _mean_correct_covar_rhs(self, b_covar, b_mean, mean_coeff):
        src = self.src

        ctf_fb = self.ctf_fb
        ctf_idx = self.ctf_idx

        partition = ctf_fb[0].partition

        # Note: If we don't do this, we'll be modifying the stored `b_covar`
        # since the operations below are in-place.
        b_covar = b_covar.copy()

        for k in np.unique(ctf_idx):
            weight = np.count_nonzero(ctf_idx == k) / src.n

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = ctf_fb_k.T

            mean_coeff_k = ctf_fb_k.apply(mean_coeff)
            mean_coeff_k = ctf_fb_k_t.apply(mean_coeff_k)

            mean_coeff_k = mean_coeff_k[: partition[0][0]]
            b_mean_k = b_mean[k][: partition[0][0]]

            correction = (
                np.outer(mean_coeff_k, b_mean_k)
                + np.outer(b_mean_k, mean_coeff_k)
                - weight * np.outer(mean_coeff_k, mean_coeff_k)
            )

            b_covar[0] -= correction

        return b_covar

    def _noise_correct_covar_rhs(self, b_covar, b_noise, noise_var, shrinker):
        if shrinker == "None":
            b_noise = -noise_var * b_noise
            b_covar += b_noise
        else:
            b_covar = self.shrink_covar_backward(
                b_covar, b_noise, self.src.n, noise_var, shrinker
            )

        return b_covar

    def _solve_covar(self, A_covar, b_covar, M, covar_est_opt):
        ctf_fb = self.ctf_fb

        def precond_fun(S, x):
            p = np.size(S, 0)
            ensure(np.size(x) == p * p, "The sizes of S and x are not consistent.")
            x = m_reshape(x, (p, p))
            y = S @ x @ S
            y = m_reshape(y, (p ** 2,))
            return y

        def apply(A, x):
            p = np.size(A[0], 0)
            x = m_reshape(x, (p, p))
            y = np.zeros_like(x)
            for k in range(0, len(A)):
                y = y + A[k] @ x @ A[k].T
            y = m_reshape(y, (p ** 2,))
            return y

        cg_opt = covar_est_opt
        covar_coeff = BlkDiagMatrix.zeros_like(ctf_fb[0])

        for ell in range(0, len(b_covar)):
            A_ell = []
            for k in range(0, len(A_covar)):
                A_ell.append(A_covar[k][ell])
            p = np.size(A_ell[0], 0)
            b_ell = m_reshape(b_covar[ell], (p ** 2,))
            S = inv(M[ell])
            cg_opt["preconditioner"] = lambda x: precond_fun(S, x)
            covar_coeff_ell, _, _ = conj_grad(lambda x: apply(A_ell, x), b_ell, cg_opt)
            covar_coeff[ell] = m_reshape(covar_coeff_ell, (p, p))

        return covar_coeff

    def get_mean(self):
        """
        Calculate the rotationally invariant mean image in the basis
        coefficients.

        :return: The mean coefficient vector in `self.basis`.
        """

        if not self.b_mean:
            self._calc_rhs()

        if not self.A_mean:
            self._calc_op()

        b_mean_all = np.stack(self.b_mean).sum(axis=0)
        mean_coeff = self.A_mean.solve(b_mean_all)

        return mean_coeff

    def get_covar(self, noise_var=1, mean_coeff=None, covar_est_opt=None):
        """
        Calculate the block diagonal covariance matrix in the basis
        coefficients.

        :param noise_var: The variance of the noise in the images (default 1)
        :param mean_coeff: If specified, overrides the mean coefficient vector
        used to calculate the covariance (default `self.get_mean()`).
        :param :covar_est_opt: The estimation parameters for obtaining the covariance
        matrix in the form of a dictionary. Keys include:
            - 'shrinker': The type of shrinkage we apply to the right-hand side
              in the normal equations. Can be `'None'`, in which case no
              shrinkage is performed. For a list of shrinkers, see the
              documentation of `shrink_covar`.
            - 'verbose': Verbosity (integer) of the conjugate gradient algorithm
              (see documentation for `conj_grad`, default zero).
            - 'max_iter': Maximum number of conjugate gradient iterations (see
              documentation for `conj_grad`, default 250).
            - 'iter_callback': Callback performed at the end of an iteration
              (see documentation for `conj_grad`, default `[]`).
            - 'store_iterates': Determines whether to store intermediate
              iterates (see documentation for `conj_grad`, default `False`).
            - 'rel_tolerance': Relative stopping tolerance of the conjugate
              gradient algorithm (see documentation for `conj_grad`, default
              `1e-12`).
            - 'precision': Precision of conjugate gradient algorithm (see
              documentation for `conj_grad`, default `'float64'`)
        :return: The block diagonal matrix containing the basis coefficients (in
        `self.basis`) for the estimated covariance matrix. These are
        implemented using `BlkDiagMatrix`.
        """

        def identity(x):
            return x

        default_est_opt = {
            "shrinker": "None",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": "float64",
            "preconditioner": identity,
        }

        covar_est_opt = fill_struct(covar_est_opt, default_est_opt)

        if not self.b_covar:
            self._calc_rhs()

        if not self.A_covar or self.M_covar:
            self._calc_op()

        if mean_coeff is None:
            mean_coeff = self.get_mean()

        b_covar = self.b_covar

        b_covar = self._mean_correct_covar_rhs(b_covar, self.b_mean, mean_coeff)
        b_covar = self._noise_correct_covar_rhs(
            b_covar, self.A_mean, noise_var, covar_est_opt["shrinker"]
        )

        covar_coeff = self._solve_covar(
            self.A_covar, b_covar, self.M_covar, covar_est_opt
        )

        return covar_coeff

    def get_cwf_coeffs(
        self, coeffs, ctf_fb, ctf_idx, mean_coeff, covar_coeff, noise_var=1
    ):
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
            mean_coeff = self.get_mean()

        if covar_coeff is None:
            covar_coeff = self.get_covar(noise_var=noise_var, mean_coeff=mean_coeff)

        if (ctf_fb is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coeffs.shape[1], dtype=int)
            ctf_fb = [BlkDiagMatrix.eye_like(covar_coeff)]

        noise_covar_coeff = noise_var * BlkDiagMatrix.eye_like(covar_coeff)

        coeffs_est = np.zeros_like(coeffs)

        for k in np.unique(ctf_idx[:]):
            coeff_k = coeffs[ctf_idx == k]
            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = ctf_fb_k.T
            sig_covar_coeff = ctf_fb_k @ covar_coeff @ ctf_fb_k_t
            sig_noise_covar_coeff = sig_covar_coeff + noise_covar_coeff

            mean_coeff_k = ctf_fb_k.apply(mean_coeff)

            coeff_est_k = coeff_k - mean_coeff_k
            coeff_est_k = sig_noise_covar_coeff.solve(coeff_est_k.T).T
            coeff_est_k = (covar_coeff @ ctf_fb_k_t).apply(coeff_est_k.T).T
            coeff_est_k = coeff_est_k + mean_coeff
            coeffs_est[ctf_idx == k] = coeff_est_k

        return coeffs_est
