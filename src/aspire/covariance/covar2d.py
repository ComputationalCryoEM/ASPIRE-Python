import logging

import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import solve, sqrtm

from aspire.basis import Coef, FFBBasis2D
from aspire.operators import BlkDiagMatrix, DiagMatrix
from aspire.optimization import conj_grad, fill_struct
from aspire.utils import make_symmat
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


def shrink_covar(covar, noise_var, gamma, shrinker="frobenius_norm"):
    """
    Shrink the covariance matrix

    :param covar_in: An input covariance matrix
    :param noise_var: The estimated variance of noise
    :param gamma: An input parameter to specify the maximum values of eigen values to be neglected.
    :param shrinker: An input parameter to select different shrinking methods.
    :return: The shrinked covariance matrix
    """

    assert shrinker in (
        "frobenius_norm",
        "operator_norm",
        "soft_threshold",
    ), "Unsupported shrink method"

    lambs, eig_vec = eig(make_symmat(covar))

    lambda_max = noise_var * (1 + np.sqrt(gamma)) ** 2

    lambs[lambs < lambda_max] = 0

    if shrinker == "operator_norm":
        lambdas = lambs[lambs > lambda_max]
        lambdas = (
            1
            / 2
            * (
                lambdas
                - noise_var * (gamma - 1)
                + np.sqrt(
                    (lambdas - noise_var * (gamma - 1)) ** 2 - 4 * noise_var * lambdas
                )
            )
            - noise_var
        )

        lambs[lambs > lambda_max] = lambdas
    elif shrinker == "frobenius_norm":
        lambdas = lambs[lambs > lambda_max]
        lambdas = (
            1
            / 2
            * (
                lambdas
                - noise_var * (gamma - 1)
                + np.sqrt(
                    (lambdas - noise_var * (gamma - 1)) ** 2 - 4 * noise_var * lambdas
                )
            )
            - noise_var
        )
        c = np.divide(
            (1 - np.divide(noise_var**2 * gamma, lambdas**2)),
            (1 + np.divide(noise_var * gamma, lambdas)),
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
        assert basis.ndim == 2, "Only two-dimensional basis functions are needed."

    def _ctf_identity_mat(self):
        """
        Returns CTF identity corresponding to the `matrix_type` of `self.basis`.

        :return: Identity BlkDiagMatrix or DiagMatrix
        """
        if self.basis.matrix_type == DiagMatrix:
            return DiagMatrix.eye(self.basis.count, dtype=self.dtype)
        else:
            return BlkDiagMatrix.eye(self.basis.blk_diag_cov_shape, dtype=self.dtype)

    def _get_mean(self, coefs):
        """
        Calculate the mean vector from the expansion coefficients of 2D images without CTF information.

        :param coefs: A coefficient vector (or an array of coefficient vectors) to be averaged.
        :return: The mean value vector for all images.
        """

        if coefs.size == 0:
            raise RuntimeError("The coefficients need to be calculated first!")

        mean_coef = np.zeros(self.basis.count, dtype=coefs.dtype)
        mean_coef[self.basis._zero_angular_inds] = np.mean(
            coefs[..., self.basis._zero_angular_inds], axis=0
        )

        return mean_coef

    def _get_covar(self, coefs, mean_coef=None, do_refl=True):
        """
        Calculate the covariance matrix from the expansion coefficients without CTF information.

        :param coefs: A coefficient vector (an array of coefficient vectors) calculated from 2D images.
        :param mean_coef: The mean vector calculated from the `coefs`.
        :param do_refl: If true, enforce invariance to reflection (default false).
        :return: The covariance matrix of coefficients for all images.
        """
        if coefs.size == 0:
            raise RuntimeError("The coefficients need to be calculated first!")
        if mean_coef is None:
            mean_coef = self._get_mean(coefs)

        # Initialize a totally empty BlkDiagMatrix, build incrementally.
        covar_coef = BlkDiagMatrix.empty(0, dtype=coefs.dtype)
        ell = 0

        mask = self.basis.angular_indices == ell

        coef_ell = coefs[..., mask] - mean_coef[mask]
        covar_ell = np.array(coef_ell.T @ coef_ell / coefs.shape[0])
        covar_coef.append(covar_ell)

        for ell in range(1, self.basis.ell_max + 1):
            mask_ell = self.basis.angular_indices == ell
            mask_pos = mask_ell & (self.basis.signs_indices == +1)
            mask_neg = mask_ell & (self.basis.signs_indices == -1)

            covar_ell_diag = np.array(
                coefs[:, mask_pos].T @ coefs[:, mask_pos]
                + coefs[:, mask_neg].T @ coefs[:, mask_neg]
            ) / (2 * coefs.shape[0])

            if do_refl:
                covar_coef.append(covar_ell_diag)
                covar_coef.append(covar_ell_diag)
            else:
                covar_ell_off = np.array(
                    (
                        coefs[:, mask_pos] @ coefs[:, mask_neg].T / coefs.shape[0]
                        - coefs[:, mask_pos].T @ coefs[:, mask_neg]
                    )
                    / (2 * coefs.shape[0])
                )

                hsize = covar_ell_diag.shape[0]
                covar_coef_blk = np.zeros((2, hsize, 2, hsize))

                covar_coef_blk[0:2, :, 0:2, :] = covar_ell_diag[:hsize, :hsize]
                covar_coef_blk[0, :, 1, :] = covar_ell_off[:hsize, :hsize]
                covar_coef_blk[1, :, 0, :] = covar_ell_off.T[:hsize, :hsize]

                covar_coef.append(covar_coef_blk.reshape(2 * hsize, 2 * hsize))

        return covar_coef

    def get_mean(self, coefs, ctf_basis=None, ctf_idx=None):
        """
        Calculate the mean vector from the expansion coefficients with CTF information.

        :param coefs: A coefficient vector (or an array of coefficient vectors) to be averaged.
        :param ctf_basis: The CTF functions in the Basis expansion.
        :param ctf_idx: An array of the CTF function indices for all 2D images.
            If ctf_basis or ctf_idx is None, the identity filter will be applied.
        :return: The mean value vector for all images.
        """

        if not isinstance(coefs, Coef):
            raise TypeError(
                f"`coefs` should be instance of `Coef`, received {type(Coef)}."
            )

        coefs = coefs.asnumpy()

        # TODO: Redundant, remove?
        if coefs.size == 0:
            raise RuntimeError("The coefficients need to be calculated!")

        # should assert we require none or both...
        if (ctf_basis is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coefs.shape[0], dtype=int)
            ctf_basis = [self._ctf_identity_mat()]

        b = np.zeros(self.basis.count, dtype=coefs.dtype)

        A = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape)
        for k in np.unique(ctf_idx[:]).T:
            coef_k = coefs[ctf_idx == k]
            weight = coef_k.shape[0] / coefs.shape[0]
            mean_coef_k = self._get_mean(coef_k)

            ctf_basis_k = ctf_basis[k]
            ctf_basis_k_t = ctf_basis_k.T
            b += weight * ctf_basis_k_t.apply(mean_coef_k)
            A += weight * (ctf_basis_k_t @ ctf_basis_k)

        mean_coef = A.solve(b)
        return Coef(self.basis, mean_coef)

    def get_covar(
        self,
        coefs,
        ctf_basis=None,
        ctf_idx=None,
        mean_coef=None,
        do_refl=True,
        noise_var=0,
        covar_est_opt=None,
        make_psd=True,
    ):
        """
        Calculate the covariance matrix from the expansion coefficients and CTF information.

        :param coefs: A coefficient vector (or an array of coefficient vectors) to be calculated.
        :param ctf_basis: The CTF functions in the Basis expansion.
        :param ctf_idx: An array of the CTF function indices for all 2D images.
            If ctf_basis or ctf_idx is None, the identity filter will be applied.
        :param mean_coef: The mean value vector from all images.
        :param noise_var: The estimated variance of noise. The value should be zero for `coefs`
            from clean images of simulation data.
        :param covar_est_opt: The optimization parameter list for obtaining the Cov2D matrix.
        :param make_psd: If True, make the covariance matrix positive semidefinite
        :return: The basis coefficients of the covariance matrix in
            the form of cell array representing a block diagonal matrix. These
            block diagonal matrices are implemented as BlkDiagMatrix instances.
            The covariance is calculated from the images represented by the coefs array,
            along with all possible rotations and reflections. As a result, the computed covariance
            matrix is invariant to both reflection and rotation. The effect of the filters in ctf_basis
            are accounted for and inverted to yield a covariance estimate of the unfiltered images.
        """

        if not isinstance(coefs, Coef):
            raise TypeError(
                f"`coefs` should be instance of `Coef`, received {type(Coef)}."
            )
        coefs = coefs.asnumpy()

        if coefs.size == 0:
            raise RuntimeError("The coefficients need to be calculated!")

        if (ctf_basis is None) or (ctf_idx is None):
            ctf_idx = np.zeros(coefs.shape[0], dtype=int)
            ctf_basis = [self._ctf_identity_mat()]

        def identity(x):
            return x

        default_est_opt = {
            "shrinker": None,
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
            "preconditioner": identity,
        }

        covar_est_opt = fill_struct(covar_est_opt, default_est_opt)

        if mean_coef is None:
            mean_coef = self.get_mean(Coef(self.basis, coefs), ctf_basis, ctf_idx)

        b_coef = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape)
        b_noise = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape)
        A = []
        for _ in range(len(ctf_basis)):
            A.append(BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape))

        M = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape)

        for k in np.unique(ctf_idx[:]):
            coef_k = coefs[ctf_idx == k].astype(self.dtype)
            weight = coef_k.shape[0] / coefs.shape[0]

            ctf_basis_k = ctf_basis[k]
            ctf_basis_k_t = ctf_basis_k.T
            mean_coef_k = ctf_basis_k.apply(mean_coef.asnumpy()[0])
            covar_coef_k = self._get_covar(coef_k, mean_coef_k)

            b_coef += weight * (ctf_basis_k_t @ covar_coef_k @ ctf_basis_k)

            ctf_basis_k_sq = ctf_basis_k_t @ ctf_basis_k
            b_noise += weight * ctf_basis_k_sq

            A_k = np.sqrt(weight) * ctf_basis_k_sq
            if not isinstance(A_k, BlkDiagMatrix):
                A_k = DiagMatrix(A_k).as_blk_diag(self.basis.blk_diag_cov_shape)

            A[k] = A_k
            M += A[k]

        if not b_coef.check_psd():
            logger.warning("Left side b in Cov2D is not positive semidefinite.")

        if covar_est_opt["shrinker"] is None:
            b = b_coef - noise_var * b_noise
        else:
            b = self.shrink_covar_backward(
                b_coef,
                b_noise,
                np.size(coefs, 0),
                noise_var,
                covar_est_opt["shrinker"],
            )
        if not b.check_psd():
            logger.warning(
                "Left side b after removing noise in Cov2D"
                " is not positive semidefinite."
            )

        # RCOPT okay, this looks like a big batch, come back later

        cg_opt = covar_est_opt

        covar_coef = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape)

        def precond_fun(S, x):
            p = np.size(S, 0)
            assert np.size(x) == p * p, "The sizes of S and x are not consistent."
            x = m_reshape(x, (p, p))
            y = S @ x @ S
            y = m_reshape(y, (p**2,))
            return y

        def apply(A, x):
            p = np.size(A[0], 0)
            x = m_reshape(x, (p, p))
            y = np.zeros_like(x)
            for k in range(0, len(A)):
                y = y + A[k] @ x @ A[k].T
            y = m_reshape(y, (p**2,))
            return y

        for ell in range(0, len(b)):
            A_ell = []
            for k in range(0, len(A)):
                A_ell.append(A[k][ell])
            p = np.size(A_ell[0], 0)
            b_ell = m_reshape(b[ell], (p**2,))
            S = inv(M[ell])
            cg_opt["preconditioner"] = lambda x, S=S: precond_fun(S, x)
            covar_coef_ell, _, _ = conj_grad(
                lambda x, A_ell=A_ell: apply(A_ell, x), b_ell, cg_opt
            )
            covar_coef[ell] = m_reshape(covar_coef_ell, (p, p))

        if not covar_coef.check_psd():
            logger.warning("Covariance matrix in Cov2D is not positive semidefinite.")
            if make_psd:
                logger.info("Convert matrices to positive semidefinite.")
                covar_coef = covar_coef.make_psd()

        return covar_coef

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
            # scipy >= 1.6.0 will upcast the sqrtm result to doubles
            #  https://github.com/scipy/scipy/issues/14853
            S = sqrtm(b_noise[ell]).astype(self.dtype)
            # from Matlab b_ell = S \ b_ell /S
            b_ell = solve(S, b_ell) @ inv(S)
            b_ell = shrink_covar(b_ell, noise_var, p / n, shrinker)
            b_ell = S @ b_ell @ S
            b_out[ell] = b_ell
        return b_out

    def get_cwf_coefs(
        self,
        coefs,
        ctf_basis=None,
        ctf_idx=None,
        mean_coef=None,
        covar_coef=None,
        noise_var=0,
    ):
        """
        Estimate the expansion coefficients using the Covariance Wiener Filtering (CWF) method.

        :param coefs: A coefficient vector (or an array of coefficient vectors) to be calculated.
        :param ctf_basis: The CTF functions in the Basis expansion.
        :param ctf_idx: An array of the CTF function indices for all 2D images.
            If ctf_basis or ctf_idx is None, the identity filter will be applied.
        :param mean_coef: The mean value vector from all images.
        :param covar_coef: The block diagonal covariance matrix of the clean coefficients represented by a cell array.
        :param noise_var: The estimated variance of noise. The value should be zero for `coefs`
            from clean images of simulation data.
        :return: The estimated coefficients of the unfiltered images in certain math basis.
            These are obtained using a Wiener filter with the specified covariance for the clean images
            and white noise of variance `noise_var` for the noise.
        """

        if not isinstance(coefs, Coef):
            raise TypeError(
                f"`coefs` should be instance of `Coef`, received {type(Coef)}."
            )

        if mean_coef is None:
            mean_coef = self.get_mean(coefs, ctf_basis, ctf_idx)

        if covar_coef is None:
            covar_coef = self.get_covar(
                coefs, ctf_basis, ctf_idx, mean_coef, noise_var=noise_var
            )

        coefs = coefs.asnumpy()

        # Handle CTF arguments.
        if (ctf_basis is None) ^ (ctf_idx is None):
            raise RuntimeError(
                "Both `ctf_basis` and `ctf_idx` should be provided,"
                " or both should be `None`."
                f' Given {"ctf_basis" if ctf_idx is None else "ctf_idx"}'
            )
        elif ctf_basis is None:
            # Setup defaults for CTF
            ctf_idx = np.zeros(coefs.shape[0], dtype=int)
            ctf_basis = [BlkDiagMatrix.eye_like(covar_coef)]

        noise_covar_coef = noise_var * BlkDiagMatrix.eye_like(covar_coef)

        coefs_est = np.zeros_like(coefs)

        for k in np.unique(ctf_idx[:]):
            coef_k = coefs[ctf_idx == k]
            ctf_basis_k = ctf_basis[k]
            ctf_basis_k_t = ctf_basis_k.T

            mean_coef_k = ctf_basis_k.apply(mean_coef.asnumpy()[0])
            coef_est_k = coef_k - mean_coef_k

            if noise_var == 0:
                coef_est_k = ctf_basis_k.solve(coef_est_k.T).T
            else:
                sig_covar_coef = ctf_basis_k @ covar_coef @ ctf_basis_k_t
                sig_noise_covar_coef = sig_covar_coef + noise_covar_coef

                coef_est_k = sig_noise_covar_coef.solve(coef_est_k.T).T
                coef_est_k = (covar_coef @ ctf_basis_k_t).apply(coef_est_k.T).T

            coef_est_k = coef_est_k + mean_coef
            coefs_est[ctf_idx == k] = coef_est_k

        return Coef(self.basis, coefs_est)


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
    :param batch_size: The number of images to process at a time (default 8192).
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
            self.basis = FFBBasis2D((src.L, src.L), dtype=self.dtype)

        if not src.unique_filters:
            logger.info("CTF filters are not included in Cov2D denoising")
            # set all CTF filters to an identity filter
            self.ctf_idx = np.zeros(src.n, dtype=int)
            self.ctf_basis = [self._ctf_identity_mat()]

        else:
            logger.info("Represent CTF filters in basis")
            unique_filters = src.unique_filters
            self.ctf_idx = src.filter_indices
            self.ctf_basis = [self.basis.filter_to_basis_mat(f) for f in unique_filters]

    def _calc_rhs(self):
        src = self.src
        basis = self.basis

        ctf_basis = self.ctf_basis
        ctf_idx = self.ctf_idx

        zero_coef = np.zeros((basis.count,), dtype=self.dtype)

        b_mean = [np.zeros(basis.count, dtype=self.dtype) for _ in ctf_basis]

        b_covar = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape, dtype=self.dtype)

        for start in range(0, src.n, self.batch_size):
            batch = np.arange(start, min(start + self.batch_size, src.n))

            im = src.images[batch[0] : batch[0] + len(batch)]
            coef = basis.evaluate_t(im).asnumpy()

            for k in np.unique(ctf_idx[batch]):
                coef_k = coef[ctf_idx[batch] == k]
                weight = np.size(coef_k, 0) / src.n

                mean_coef_k = self._get_mean(coef_k)

                ctf_basis_k = ctf_basis[k]
                ctf_basis_k_t = ctf_basis_k.T

                b_mean_k = weight * ctf_basis_k_t.apply(mean_coef_k)

                if isinstance(b_mean_k, DiagMatrix):
                    # Convert to a column vector
                    b_mean_k = b_mean_k.asnumpy().T

                b_mean[k] += b_mean_k

                covar_coef_k = self._get_covar(coef_k, zero_coef)

                b_covar_k = ctf_basis_k_t @ covar_coef_k

                b_covar_k = b_covar_k @ ctf_basis_k
                b_covar_k *= weight

                b_covar += b_covar_k

        self.b_mean = b_mean
        self.b_covar = b_covar

    def _calc_op(self):
        src = self.src

        ctf_basis = self.ctf_basis
        ctf_idx = self.ctf_idx

        A_mean = BlkDiagMatrix.zeros(self.basis.blk_diag_cov_shape, self.dtype)
        A_covar = [None for _ in ctf_basis]
        M_covar = BlkDiagMatrix.zeros_like(A_mean)

        for k in np.unique(ctf_idx):
            weight = np.count_nonzero(ctf_idx == k) / src.n

            ctf_basis_k = ctf_basis[k]
            ctf_basis_k_t = ctf_basis_k.T

            ctf_basis_k_sq = ctf_basis_k_t @ ctf_basis_k
            A_mean_k = weight * ctf_basis_k_sq
            A_mean += A_mean_k
            A_covar_k = np.sqrt(weight) * ctf_basis_k_sq
            A_covar[k] = A_covar_k

            M_covar += A_covar_k

        self.A_mean = A_mean
        self.A_covar = A_covar
        self.M_covar = M_covar

    def _mean_correct_covar_rhs(self, b_covar, b_mean, mean_coef):
        src = self.src

        ctf_basis = self.ctf_basis
        ctf_idx = self.ctf_idx

        partition = self.basis.blk_diag_cov_shape

        # Note: If we don't do this, we'll be modifying the stored `b_covar`
        # since the operations below are in-place.
        b_covar = b_covar.copy()

        for k in np.unique(ctf_idx):
            weight = np.count_nonzero(ctf_idx == k) / src.n

            ctf_basis_k = ctf_basis[k]
            ctf_basis_k_t = ctf_basis_k.T

            mean_coef_k = ctf_basis_k.apply(mean_coef.asnumpy()[0])
            mean_coef_k = ctf_basis_k_t.apply(mean_coef_k)

            mean_coef_k = mean_coef_k[: partition[0][0]]
            b_mean_k = b_mean[k][: partition[0][0]]

            correction = (
                np.outer(mean_coef_k, b_mean_k)
                + np.outer(b_mean_k, mean_coef_k)
                - weight * np.outer(mean_coef_k, mean_coef_k)
            )

            b_covar[0] -= correction

        return b_covar

    def _noise_correct_covar_rhs(self, b_covar, b_noise, noise_var, shrinker):
        if shrinker is None:
            b_noise = -noise_var * b_noise
            b_covar += b_noise
        else:
            b_covar = self.shrink_covar_backward(
                b_covar, b_noise, self.src.n, noise_var, shrinker
            )

        return b_covar

    def _solve_covar(self, A_covar, b_covar, M, covar_est_opt):
        method = self._solve_covar_cg
        if self.basis.matrix_type == DiagMatrix:
            method = self._solve_covar_direct

        return method(A_covar, b_covar, M, covar_est_opt)

    def _solve_covar_direct(self, A_covar, b_covar, M, covar_est_opt):
        # A_covar is a list of DiagMatrix, representing each ctf in self.basis.
        # b_covar is a BlkDiagMatrix
        # M is sum of weighted A squared, only used for cg, ignore here.
        A_covar = DiagMatrix(np.concatenate([x.asnumpy() for x in A_covar]))
        A2i = A_covar * A_covar

        res = BlkDiagMatrix.empty(b_covar.nblocks, self.dtype)
        for b in range(b_covar.nblocks):
            res.data[b] = b_covar[b] / A2i[b]

        return res

    def _solve_covar_cg(self, A_covar, b_covar, M, covar_est_opt):
        def precond_fun(S, x):
            p = np.size(S, 0)
            assert np.size(x) == p * p, "The sizes of S and x are not consistent."
            x = m_reshape(x, (p, p))
            y = S @ x @ S
            y = m_reshape(y, (p**2,))
            return y

        def apply(A, x):
            p = np.size(A[0], 0)
            x = m_reshape(x, (p, p))
            y = np.zeros_like(x)
            for k in range(0, len(A)):
                y = y + A[k] @ x @ A[k].T
            y = m_reshape(y, (p**2,))
            return y

        cg_opt = covar_est_opt
        covar_coef = BlkDiagMatrix.zeros(
            self.basis.blk_diag_cov_shape, dtype=self.dtype
        )

        for ell in range(0, len(b_covar)):
            A_ell = []
            for k in range(0, len(A_covar)):
                A_ell.append(A_covar[k][ell])
            p = np.size(A_ell[0], 0)
            b_ell = m_reshape(b_covar[ell], (p**2,))
            S = inv(M[ell])
            cg_opt["preconditioner"] = lambda x, S=S: precond_fun(S, x)
            covar_coef_ell, _, _ = conj_grad(
                lambda x, A_ell=A_ell: apply(A_ell, x), b_ell, cg_opt
            )
            covar_coef[ell] = m_reshape(covar_coef_ell, (p, p))

        return covar_coef

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
        mean_coef = self.A_mean.solve(b_mean_all)

        return Coef(self.basis, mean_coef)

    def get_covar(self, noise_var=0, mean_coef=None, covar_est_opt=None, make_psd=True):
        """
        Calculate the block diagonal covariance matrix in the basis
        coefficients.

        :param noise_var: The variance of the noise in the images (default 1)
        :param mean_coef: If specified, overrides the mean coefficient vector
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
        :param make_psd: If True, make the covariance matrix positive semidefinite
        :return: The block diagonal matrix containing the basis coefficients (in
            `self.basis`) for the estimated covariance matrix. These are
            implemented using `BlkDiagMatrix`.
        """

        def identity(x):
            return x

        default_est_opt = {
            "shrinker": None,
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
            "preconditioner": identity,
        }

        covar_est_opt = fill_struct(covar_est_opt, default_est_opt)

        if not self.b_covar:
            self._calc_rhs()

        if not self.A_covar or self.M_covar:
            self._calc_op()

        if mean_coef is None:
            mean_coef = self.get_mean()

        b_covar = self.b_covar

        b_covar = self._mean_correct_covar_rhs(b_covar, self.b_mean, mean_coef)
        if not b_covar.check_psd():
            logger.warning("Left side b in Batched Cov2D is not positive semidefinite.")

        b_covar = self._noise_correct_covar_rhs(
            b_covar, self.A_mean, noise_var, covar_est_opt["shrinker"]
        )
        if not b_covar.check_psd():
            logger.warning(
                "Left side b after removing noise "
                "in Batched Cov2D is not positive semidefinite."
            )

        covar_coef = self._solve_covar(
            self.A_covar, b_covar, self.M_covar, covar_est_opt
        )
        if not covar_coef.check_psd():
            logger.warning(
                "Covariance matrix in Batched Cov2D is not positive semidefinite."
            )
            if make_psd:
                logger.info("Convert matrices to positive semidefinite.")
                covar_coef = covar_coef.make_psd()

        return covar_coef

    def get_cwf_coefs(
        self, coefs, ctf_basis, ctf_idx, mean_coef, covar_coef, noise_var=0
    ):
        """
        Estimate the expansion coefficients using the Covariance Wiener Filtering (CWF) method.

        :param coefs: A coefficient vector (or an array of coefficient vectors) to be calculated.
        :param ctf_basis: The CTF functions in the Basis expansion.
        :param ctf_idx: An array of the CTF function indices for all 2D images.
            If ctf_basis or ctf_idx is None, the identity filter will be applied.
        :param mean_coef: The mean value vector from all images.
        :param covar_coef: The block diagonal covariance matrix of the clean coefficients represented by a cell array.
        :param noise_var: The estimated variance of noise. The value should be zero for `coefs`
            from clean images of simulation data.
        :return: The estimated coefficients of the unfiltered images in certain math basis.
            These are obtained using a Wiener filter with the specified covariance for the clean images
            and white noise of variance `noise_var` for the noise.
        """

        if not isinstance(coefs, Coef):
            raise TypeError(
                f"`coefs` should be instance of `Coef`, received {type(Coef)}."
            )
        coefs = coefs.asnumpy()

        if mean_coef is None:
            mean_coef = self.get_mean()

        if covar_coef is None:
            covar_coef = self.get_covar(noise_var=noise_var, mean_coef=mean_coef)

        # Handle CTF arguments.
        if (ctf_basis is None) ^ (ctf_idx is None):
            raise RuntimeError(
                "Both `ctf_basis` and `ctf_idx` should be provided,"
                " or both should be `None`."
                f' Given {"ctf_basis" if ctf_idx is None else "ctf_idx"}'
            )
        elif ctf_basis is None:
            # Setup defaults for CTF
            ctf_idx = np.zeros(coefs.shape[0], dtype=int)
            ctf_basis = [BlkDiagMatrix.eye_like(covar_coef)]

        noise_covar_coef = noise_var * BlkDiagMatrix.eye_like(covar_coef)

        coefs_est = np.zeros_like(coefs)

        for k in np.unique(ctf_idx[:]):
            coef_k = coefs[ctf_idx == k]
            ctf_basis_k = ctf_basis[k]
            ctf_basis_k_t = ctf_basis_k.T

            mean_coef_k = ctf_basis_k.apply(mean_coef.asnumpy()[0])
            coef_est_k = coef_k - mean_coef_k

            if noise_var == 0:
                coef_est_k = ctf_basis_k.solve(coef_est_k.T).T
            else:
                sig_covar_coef = ctf_basis_k @ covar_coef @ ctf_basis_k_t
                sig_noise_covar_coef = sig_covar_coef + noise_covar_coef

                coef_est_k = sig_noise_covar_coef.solve(coef_est_k.T).T
                coef_est_k = (covar_coef @ ctf_basis_k_t).apply(coef_est_k.T).T

            coef_est_k = coef_est_k + mean_coef
            coefs_est[ctf_idx == k] = coef_est_k

        return Coef(self.basis, coefs_est)
