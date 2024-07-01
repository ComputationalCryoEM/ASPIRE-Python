import logging

import numpy as np
from numpy import pi
from scipy.special import jv

from aspire.basis import FBBasis2D
from aspire.basis.basis_utils import lgwt
from aspire.nufft import anufft, nufft
from aspire.numeric import fft, xp
from aspire.operators import BlkDiagMatrix
from aspire.utils import complex_type
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


class FFBBasis2D(FBBasis2D):
    """
    Define a derived class for Fast Fourier Bessel expansion for 2D images

    The expansion coefficients of 2D images on this basis are obtained by
    a fast method instead of the least squares method.
    The algorithm is described in the publication:
    Z. Zhao, Y. Shkolnisky, A. Singer, Fast Steerable Principal Component Analysis,
    IEEE Transactions on Computational Imaging, 2 (1), pp. 1-12 (2016).​

    """

    def _build(self):
        """
        Build the internal data structure to 2D Fourier-Bessel basis
        """
        logger.info(
            "Expanding 2D image in a frequency-domain Fourier–Bessel"
            " basis using the fast method."
        )

        # set cutoff values
        self.rcut = self.nres / 2
        self.kcut = 0.5
        self.n_r = int(np.ceil(4 * self.rcut * self.kcut))
        n_theta = np.ceil(16 * self.kcut * self.rcut)
        self.n_theta = int((n_theta + np.mod(n_theta, 2)) / 2)

        # get upper bound of zeros, ells, and ks  of Bessel functions
        self._calc_k_max()

        # calculate total number of basis functions
        self.count = self.k_max[0] + sum(2 * self.k_max[1:])

        # generate 1D indices for basis functions
        self._compute_indices()

        # get normalized factors
        self.radial_norms, self.angular_norms = self.norms()

        # precompute the basis functions in 2D grids
        self._precomp = self._precomp()

        # include the normalization factor of angular part into radial part
        self.radial_norm = xp.asarray(self._precomp["radial"]) / xp.asarray(
            np.expand_dims(self.angular_norms, 1)
        )

        # precompute weighted nodes
        self.gl_weighted_nodes = xp.asarray(self._precomp["gl_weights"]) * xp.asarray(
            self._precomp["gl_nodes"]
        )

    def _precomp(self):
        """
        Precomute the basis functions on a polar Fourier grid

        Gaussian quadrature points and weights are also generated.
        The sampling criterion requires n_r=4*c*R and n_theta= 16*c*R.

        """
        n_r = self.n_r
        n_theta = self.n_theta
        r, w = lgwt(n_r, 0.0, self.kcut, dtype=self.dtype)

        radial = np.zeros(shape=(np.sum(self.k_max), n_r), dtype=self.dtype)
        ind_radial = 0
        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                radial[ind_radial] = jv(ell, self.r0[ell][k - 1] * r / self.kcut)
                # NOTE: We need to remove the factor due to the discretization here
                # since it is already included in our quadrature weights
                # Only normalized by the radial part of basis function
                nrm = 1 / (np.sqrt(np.prod(self.sz))) * self.radial_norms[ind_radial]
                radial[ind_radial] /= nrm
                ind_radial += 1

        # Only calculate "positive" frequencies in one half-plane.
        freqs_x = m_reshape(r, (n_r, 1)) @ m_reshape(
            np.cos(np.arange(n_theta, dtype=self.dtype) * 2 * pi / (2 * n_theta)),
            (1, n_theta),
        )
        freqs_y = m_reshape(r, (n_r, 1)) @ m_reshape(
            np.sin(np.arange(n_theta, dtype=self.dtype) * 2 * pi / (2 * n_theta)),
            (1, n_theta),
        )
        freqs = np.vstack((freqs_y[np.newaxis, ...], freqs_x[np.newaxis, ...]))

        return {"gl_nodes": r, "gl_weights": w, "radial": radial, "freqs": freqs}

    def _evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in FB basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            in FB basis to be evaluated. The last dimension must equal `self.count`.
        :return x: The evaluation of the coefficient vector(s) `x` in standard 2D
            coordinate basis. This is Image instance with resolution of `self.sz`
            and the first dimension correspond to remaining dimension of `v`.
        """
        v = xp.asarray(v)
        sz_roll = v.shape[:-1]
        v = v.reshape(-1, self.count)

        # number of 2D image samples
        n_data = v.shape[0]

        # get information on polar grids from precomputed data
        n_theta = self._precomp["freqs"].shape[2]
        n_r = self._precomp["freqs"].shape[1]

        # go through  each basis function and find corresponding coefficient
        pf = xp.zeros((n_data, 2 * n_theta, n_r), dtype=complex_type(self.dtype))

        ind = 0

        idx = ind + np.arange(self.k_max[0], dtype=int)

        pf[:, 0, :] = v[:, self._zero_angular_inds] @ self.radial_norm[idx]
        ind = ind + idx.size

        ind_pos = ind

        for ell in range(1, self.ell_max + 1):
            idx = ind + xp.arange(self.k_max[ell], dtype=int)
            idx_pos = ind_pos + np.arange(self.k_max[ell], dtype=int)
            idx_neg = idx_pos + self.k_max[ell]

            v_ell = (v[:, idx_pos] - 1j * v[:, idx_neg]) / 2.0

            if np.mod(ell, 2) == 1:
                v_ell = 1j * v_ell

            pf_ell = v_ell @ self.radial_norm[idx]
            pf[:, ell, :] = pf_ell

            if np.mod(ell, 2) == 0:
                pf[:, 2 * n_theta - ell, :] = pf_ell.conjugate()
            else:
                pf[:, 2 * n_theta - ell, :] = -pf_ell.conjugate()

            ind = ind + idx.size
            ind_pos = ind_pos + 2 * self.k_max[ell]

        # 1D inverse FFT in the degree of polar angle
        pf = 2 * xp.pi * fft.ifft(pf, axis=1)

        # Only need "positive" frequencies.
        hsize = int(pf.shape[1] / 2)
        pf = pf[:, 0:hsize, :]
        pf *= self.gl_weighted_nodes[None, None, :]
        pf = pf.reshape(n_data, n_r * n_theta)

        # perform inverse non-uniformly FFT transform back to 2D coordinate basis
        freqs = m_reshape(self._precomp["freqs"], (2, n_r * n_theta))

        x = 2 * anufft(pf, 2 * pi * freqs, self.sz, real=True)

        # Return X as Image instance with the last two dimensions as *self.sz
        x = x.reshape((*sz_roll, *self.sz))

        return xp.asnumpy(x)

    def _evaluate_t(self, x):
        """
        Evaluate coefficient in FB basis from those in standard 2D coordinate basis

        :param x: The Image instance representing coefficient array in the
            standard 2D coordinate basis to be evaluated.
        :return: The evaluation of the coefficient array `x` in the FB basis.
            This is an array of vectors whose last dimension equals `self.count`
            and whose first dimension correspond to `x.shape[0]`.
        """
        # get information on polar grids from precomputed data
        n_theta = np.size(self._precomp["freqs"], 2)
        n_r = np.size(self._precomp["freqs"], 1)
        freqs = np.reshape(self._precomp["freqs"], (2, n_r * n_theta))

        # number of 2D image samples
        n_images = x.shape[0]

        # resamping x in a polar Fourier gird using nonuniform discrete Fourier transform
        pf = nufft(xp.asarray(x), 2 * pi * freqs)
        pf = pf.reshape(n_images, n_r, n_theta)

        # Recover "negative" frequencies from "positive" half plane.
        pf = xp.concatenate((pf, pf.conjugate()), axis=2)

        # evaluate radial integral using the Gauss-Legendre quadrature rule
        pf = pf * self.gl_weighted_nodes[None, :, None]

        #  1D FFT on the angular dimension for each concentric circle
        pf = 2 * xp.pi / (2 * n_theta) * fft.fft(pf)

        # This only makes it easier to slice the array later.
        v = xp.zeros((n_images, self.count), dtype=x.dtype)

        # go through each basis function and find the corresponding coefficient
        ind = 0
        idx = ind + xp.arange(self.k_max[0])

        v[:, self._zero_angular_inds] = pf[:, :, 0].real @ self.radial_norm[idx].T
        ind = ind + idx.size

        ind_pos = ind
        for ell in range(1, self.ell_max + 1):
            idx = ind + xp.arange(self.k_max[ell])
            idx_pos = ind_pos + xp.arange(self.k_max[ell])
            idx_neg = idx_pos + self.k_max[ell]

            v_ell = pf[:, :, ell] @ self.radial_norm[idx].T

            if np.mod(ell, 2) == 0:
                v_pos = v_ell.real
                v_neg = -v_ell.imag
            else:
                v_pos = v_ell.imag
                v_neg = v_ell.real

            v[:, idx_pos] = v_pos
            v[:, idx_neg] = v_neg

            ind = ind + idx.size

            ind_pos = ind_pos + 2 * self.k_max[ell]

        return xp.asnumpy(v)

    def filter_to_basis_mat(self, f, **kwargs):
        """
        See `SteerableBasis2D.filter_to_basis_mat`.
        """
        # Note 'method' and 'truncate' not relevant for this optimized FFB code.
        if kwargs.get("method", None) is not None:
            raise NotImplementedError(
                "`FFBBasis2D.filter_to_basis_mat` method {method} not supported."
                "  Use `method=None`."
            )

        # These form a circular dependence, import locally until time to clean up.
        from aspire.basis.basis_utils import lgwt

        # Get the filter's evaluate function.
        h_fun = f.evaluate

        # Set same dimensions as basis object
        n_k = self.n_r
        n_theta = self.n_theta
        radial = self._precomp["radial"]

        # get 2D grid in polar coordinate
        k_vals, wts = lgwt(n_k, 0, 0.5, dtype=self.dtype)
        k, theta = np.meshgrid(
            k_vals, np.arange(n_theta) * 2 * np.pi / (2 * n_theta), indexing="ij"
        )

        # Get function values in polar 2D grid and average out angle contribution
        omegax = k * np.cos(theta)
        omegay = k * np.sin(theta)
        omega = 2 * np.pi * np.vstack((omegax.flatten("C"), omegay.flatten("C")))

        h_vals2d = h_fun(omega).reshape(n_k, n_theta).astype(self.dtype)
        h_vals = np.sum(h_vals2d, axis=1) / n_theta

        # Represent 1D function values in basis
        h_basis = BlkDiagMatrix.empty(2 * self.ell_max + 1, dtype=self.dtype)
        ind_ell = 0
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            rmat = 2 * k_vals.reshape(n_k, 1) * self.r0[ell][0:k_max].T
            basis_vals = np.zeros_like(rmat)
            ind_radial = np.sum(self.k_max[0:ell])
            basis_vals[:, 0:k_max] = radial[ind_radial : ind_radial + k_max].T
            h_basis_vals = basis_vals * h_vals.reshape(n_k, 1)
            h_basis_ell = basis_vals.T @ (
                h_basis_vals * k_vals.reshape(n_k, 1) * wts.reshape(n_k, 1)
            )
            h_basis[ind_ell] = h_basis_ell
            ind_ell += 1
            if ell > 0:
                h_basis[ind_ell] = h_basis[ind_ell - 1]
                ind_ell += 1

        return h_basis
