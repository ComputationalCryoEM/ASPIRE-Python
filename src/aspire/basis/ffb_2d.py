import logging

import numpy as np
from numpy import pi
from scipy.special import jv

from aspire.basis import FBBasis2D
from aspire.basis.basis_utils import lgwt
from aspire.image import Image
from aspire.nufft import anufft, nufft
from aspire.numeric import fft, xp
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
        self._getfbzeros()

        # calculate total number of basis functions
        self.count = self.k_max[0] + sum(2 * self.k_max[1:])

        # generate 1D indices for basis functions
        self._indices = self.indices()

        # get normalized factors
        self.radial_norms, self.angular_norms = self.norms()

        # precompute the basis functions in 2D grids
        self._precomp = self._precomp()

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
                radial[ind_radial] = jv(ell, self.r0[k - 1, ell] * r / self.kcut)
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

    def get_radial(self):
        """
        Return precomputed radial part
        """
        return self._precomp["radial"]

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in FB basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            in FB basis to be evaluated. The last dimension must equal `self.count`.
        :return x: The evaluation of the coefficient vector(s) `x` in standard 2D
            coordinate basis. This is Image instance with resolution of `self.sz`
            and the first dimension correspond to remaining dimension of `v`.
        """

        if v.dtype != self.dtype:
            logger.debug(
                f"{self.__class__.__name__}::evaluate"
                f" Inconsistent dtypes v: {v.dtype} self: {self.dtype}"
            )

        sz_roll = v.shape[:-1]
        v = v.reshape(-1, self.count)

        # number of 2D image samples
        n_data = v.shape[0]

        # get information on polar grids from precomputed data
        n_theta = np.size(self._precomp["freqs"], 2)
        n_r = np.size(self._precomp["freqs"], 1)

        # go through  each basis function and find corresponding coefficient
        pf = np.zeros((n_data, 2 * n_theta, n_r), dtype=complex_type(self.dtype))
        mask = self._indices["ells"] == 0

        ind = 0

        idx = ind + np.arange(self.k_max[0], dtype=np.int)

        # include the normalization factor of angular part into radial part
        radial_norm = self._precomp["radial"] / np.expand_dims(self.angular_norms, 1)
        pf[:, 0, :] = v[:, mask] @ radial_norm[idx]
        ind = ind + np.size(idx)

        ind_pos = ind

        for ell in range(1, self.ell_max + 1):
            idx = ind + np.arange(self.k_max[ell], dtype=np.int)
            idx_pos = ind_pos + np.arange(self.k_max[ell], dtype=np.int)
            idx_neg = idx_pos + self.k_max[ell]

            v_ell = (v[:, idx_pos] - 1j * v[:, idx_neg]) / 2.0

            if np.mod(ell, 2) == 1:
                v_ell = 1j * v_ell

            pf_ell = v_ell @ radial_norm[idx]
            pf[:, ell, :] = pf_ell

            if np.mod(ell, 2) == 0:
                pf[:, 2 * n_theta - ell, :] = pf_ell.conjugate()
            else:
                pf[:, 2 * n_theta - ell, :] = -pf_ell.conjugate()

            ind = ind + np.size(idx)
            ind_pos = ind_pos + 2 * self.k_max[ell]

        # 1D inverse FFT in the degree of polar angle
        pf = 2 * pi * xp.asnumpy(fft.ifft(xp.asarray(pf), axis=1))

        # Only need "positive" frequencies.
        hsize = int(np.size(pf, 1) / 2)
        pf = pf[:, 0:hsize, :]

        for i_r in range(0, n_r):
            pf[..., i_r] = pf[..., i_r] * (
                self._precomp["gl_weights"][i_r] * self._precomp["gl_nodes"][i_r]
            )

        pf = np.reshape(pf, (n_data, n_r * n_theta))

        # perform inverse non-uniformly FFT transform back to 2D coordinate basis
        freqs = m_reshape(self._precomp["freqs"], (2, n_r * n_theta))

        x = 2 * anufft(pf, 2 * pi * freqs, self.sz, real=True)

        # Return X as Image instance with the last two dimensions as *self.sz
        x = x.reshape((*sz_roll, *self.sz))

        return Image(x)

    def evaluate_t(self, x):
        """
        Evaluate coefficient in FB basis from those in standard 2D coordinate basis

        :param x: The Image instance representing coefficient array in the
        standard 2D coordinate basis to be evaluated.
        :return v: The evaluation of the coefficient array `v` in the FB basis.
            This is an array of vectors whose last dimension equals `self.count`
            and whose first dimension correspond to `x.n_images`.
        """

        if x.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}::evaluate_t"
                f" Inconsistent dtypes v: {x.dtype} self: {self.dtype}"
            )

        if not isinstance(x, Image):
            logger.warning(
                f"{self.__class__.__name__}::evaluate_t"
                " passed numpy array instead of Image."
            )
            x = Image(x)

        # get information on polar grids from precomputed data
        n_theta = np.size(self._precomp["freqs"], 2)
        n_r = np.size(self._precomp["freqs"], 1)
        freqs = np.reshape(self._precomp["freqs"], (2, n_r * n_theta))

        # number of 2D image samples
        n_images = x.n_images
        x_data = x.data

        # resamping x in a polar Fourier gird using nonuniform discrete Fourier transform
        pf = nufft(x_data, 2 * pi * freqs)
        pf = np.reshape(pf, (n_images, n_r, n_theta))

        # Recover "negative" frequencies from "positive" half plane.
        pf = np.concatenate((pf, pf.conjugate()), axis=2)

        # evaluate radial integral using the Gauss-Legendre quadrature rule
        for i_r in range(0, n_r):
            pf[:, i_r, :] = pf[:, i_r, :] * (
                self._precomp["gl_weights"][i_r] * self._precomp["gl_nodes"][i_r]
            )

        #  1D FFT on the angular dimension for each concentric circle
        pf = 2 * pi / (2 * n_theta) * xp.asnumpy(fft.fft(xp.asarray(pf)))

        # This only makes it easier to slice the array later.
        v = np.zeros((n_images, self.count), dtype=x.dtype)

        # go through each basis function and find the corresponding coefficient
        ind = 0
        idx = ind + np.arange(self.k_max[0])
        mask = self._indices["ells"] == 0

        # include the normalization factor of angular part into radial part
        radial_norm = self._precomp["radial"] / np.expand_dims(self.angular_norms, 1)
        v[:, mask] = pf[:, :, 0].real @ radial_norm[idx].T
        ind = ind + np.size(idx)

        ind_pos = ind
        for ell in range(1, self.ell_max + 1):
            idx = ind + np.arange(self.k_max[ell])
            idx_pos = ind_pos + np.arange(self.k_max[ell])
            idx_neg = idx_pos + self.k_max[ell]

            v_ell = pf[:, :, ell] @ radial_norm[idx].T

            if np.mod(ell, 2) == 0:
                v_pos = np.real(v_ell)
                v_neg = -np.imag(v_ell)
            else:
                v_pos = np.imag(v_ell)
                v_neg = np.real(v_ell)

            v[:, idx_pos] = v_pos
            v[:, idx_neg] = v_neg

            ind = ind + np.size(idx)

            ind_pos = ind_pos + 2 * self.k_max[ell]

        return v
