import logging
import numpy as np
from numpy import pi
from scipy.special import jv

from aspire.nfft import anufft3, nufft3
from aspire.utils.matrix import roll_dim, unroll_dim
from aspire.utils.matlab_compat import m_reshape
from aspire.basis.basis_utils import lgwt
from aspire.basis.fb_2d import FBBasis2D

from aspire.utils.numeric import xp

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
        logger.info('Expanding 2D image in a frequency-domain Fourier-Bessel'
                    ' basis using the fast method.')

        # set cutoff values
        self.rcut = self.nres / 2
        self.kcut = 0.5

        # get upper bound of zeros, ells, and ks  of Bessel functions
        self._getfbzeros()

        # calculate total number of basis functions
        self.count = self.k_max[0] + sum(2 * self.k_max[1:])

        # generate 1D indices for basis functions
        self._indices = self.indices()

        # precompute the basis functions in 2D grids
        self._precomp = self._precomp()

        # get normalized factors
        self._norms = self.norms()

    def _precomp(self):
        """
        Precomute the basis functions on a polar Fourier grid

        Gaussian quadrature points and weights are also generated.
        The sampling criterion requires n_r=4*c*R and n_theta= 16*c*R.
        
        """
        n_r = int(np.ceil(4 * self.rcut * self.kcut))
        r, w = lgwt(n_r, 0.0, self.kcut)

        radial = np.zeros(shape=(n_r, np.sum(self.k_max)))
        ind_radial = 0
        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                radial[:, ind_radial] = jv(ell, self.r0[k - 1, ell] * r / self.kcut)
                # NOTE: We need to remove the factor due to the discretization here
                # since it is already included in our quadrature weights
                nrm = 1 / (np.sqrt(np.prod(self.sz))) * self.basis_norm_2d(ell, k)
                radial[:, ind_radial] /= nrm
                ind_radial += 1

        n_theta = np.ceil(16 * self.kcut * self.rcut)
        n_theta = int((n_theta + np.mod(n_theta, 2)) / 2)

        # Only calculate "positive" frequencies in one half-plane.
        freqs_x = m_reshape(r, (n_r, 1)) @ m_reshape(
            np.cos(np.arange(n_theta) * 2 * pi / (2 * n_theta)), (1, n_theta))
        freqs_y = m_reshape(r, (n_r, 1)) @ m_reshape(
            np.sin(np.arange(n_theta) * 2 * pi / (2 * n_theta)), (1, n_theta))
        freqs = np.vstack((freqs_x[np.newaxis, ...], freqs_y[np.newaxis, ...]))

        return {
            'gl_nodes': r,
            'gl_weights': w,
            'radial': radial,
            'freqs': freqs
        }

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in FB basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            in FB basis to be evaluated. The first dimension must equal `self.count`.
        :return x: The evaluation of the coefficient vector(s) `x` in standard 2D
            coordinate basis. This is an array whose first two dimensions equal `self.sz`
            and the remaining dimensions correspond to dimensions two and higher of `v`.
        """
        # make should the first dimension of v is self.count
        v, sz_roll = unroll_dim(v, 2)
        v = xp.asarray(v)

        # get information on polar grids from precomputed data
        freqs = self._precomp["freqs"]
        n_theta = np.size(freqs, 2)
        n_r = np.size(freqs, 1)

        freqs = m_reshape(freqs, new_shape=(2, n_r * n_theta))

        # number of 2D image samples
        n_data = xp.size(v, 1)

        gl_weights = xp.asarray(self._precomp["gl_weights"])
        gl_nodes = xp.asarray(self._precomp["gl_nodes"])
        radial = xp.asarray(self._precomp["radial"])

        # go through  each basis function and find corresponding coefficient
        pf = xp.zeros((n_r, 2 * n_theta, n_data), dtype=xp.complex)
        mask = xp.asarray(self._indices["ells"] == 0)

        ind = 0

        idx = ind + xp.arange(self.k_max[0])

        pf[:, 0, :] = xp.matmul(radial[:, idx], v[mask, ...])

        ind = ind + xp.size(idx)

        ind_pos = ind

        for ell in range(1, self.ell_max + 1):
            idx = ind + xp.arange(self.k_max[ell])
            idx_pos = ind_pos + xp.arange(self.k_max[ell])
            idx_neg = idx_pos + self.k_max[ell]

            v_ell = (v[idx_pos, :] - 1j * v[idx_neg, :]) / 2.0

            if xp.mod(ell, 2) == 1:
                v_ell = 1j * v_ell

            pf_ell = xp.matmul(radial[:, idx], v_ell)
            pf[:, ell, :] = pf_ell

            if xp.mod(ell, 2) == 0:
                pf[:, 2 * n_theta - ell, :] = xp.conj(pf_ell)
            else:
                pf[:, 2 * n_theta - ell, :] = -xp.conj(pf_ell)

            ind = ind + xp.size(idx)
            ind_pos = ind_pos + 2 * self.k_max[ell]

        # 1D inverse FFT in the degree of polar angle
        pf = 2 * pi * xp.fft.ifft(pf, 2*n_theta, axis=1)

        # Only need "positive" frequencies.
        hsize = int(xp.size(pf, 1) / 2)
        pf = pf[:, 0:hsize, :]

        for i_r in range(0, n_r):
            pf[i_r, ...] = pf[i_r, ...] * (
                    gl_weights[i_r] * gl_nodes[i_r])

        pf = m_reshape(xp.asnumpy(pf), (n_r * n_theta, n_data))

        # perform inverse non-uniformly FFT transform back to 2D coordinate basis
        # C major outer dimension, with F indexing in C order
        # This will at least give an idea on the single vs many plan cost.
        # TODO: try to avoid the axis swapping etc when we do clean up
        pfc = np.empty((n_data, n_r * n_theta), pf.dtype)
        for transf in range(n_data):
            pfc[transf] = pf[..., transf]
        pfc = pfc.reshape(pf.shape)
        x = 2 * anufft3(pfc, 2 * pi * freqs, self.sz, real=True, many=n_data)

        # return the x with the first two dimensions of self.sz
        x = roll_dim(x, sz_roll)
        return x

    def evaluate_t(self, x):
        """
        Evaluate coefficient in FB basis from those in standard 2D coordinate basis

        :param x: The coefficient array in the standard 2D coordinate basis to be
            evaluated. The first two dimensions must equal `self.sz`.
        :return v: The evaluation of the coefficient array `v` in the FB basis.
            This is an array of vectors whose first dimension equals `self.count`
            and whose remaining dimensions correspond to higher dimensions of `x`.
        """
        # ensure the first two dimensions with size of self.sz
        x, sz_roll = unroll_dim(x, self.ndim + 1)

        # get information on polar grids from precomputed data
        freqs = self._precomp["freqs"]
        n_theta = np.size(freqs, 2)
        n_r = np.size(freqs, 1)
        freqs = m_reshape(freqs, new_shape=(2, n_r * n_theta))
        # number of 2D image samples
        n_data = np.size(x, 2)

        pfc = np.zeros((n_data, n_r*n_theta), dtype=np.complex128)     # lets try c order
        pfc[:, :] = nufft3(x, 2 * pi * freqs, self.sz, many=n_data)     # works with finufft and cufinufft 2d2many
        pf = pfc.T
        pf = m_reshape(pf, new_shape=(n_r, n_theta, n_data))

        # Recover "negative" frequencies from "positive" half plane.
        pf = np.concatenate((pf, pf.conjugate()), axis=1)
        pf = xp.asarray(pf)
        # evaluate radial integral using the Gauss-Legendre quadrature rule
        gl_weights = xp.asarray(self._precomp["gl_weights"])
        gl_nodes = xp.asarray(self._precomp["gl_nodes"])
        radial = xp.asarray(self._precomp["radial"])
        for i_r in range(0, n_r):
            pf[i_r, ...] = pf[i_r, ...] * (
                    gl_weights[i_r] * gl_nodes[i_r])

        #  1D FFT on the angular dimension for each concentric circle
        pf = 2 * pi / (2 * n_theta) * xp.fft.fft(pf, 2*n_theta, axis=1)

        # This only makes it easier to slice the array later.
        v = xp.zeros((self.count, n_data), dtype=x.dtype)

        # go through each basis function and find the corresponding coefficient
        ind = 0
        idx = ind + xp.arange(self.k_max[0])
        mask = xp.asarray(self._indices["ells"] == 0)
        v[mask, :] = xp.matmul(radial[:, idx].T, np.real(pf[:, 0, :]))

        v = xp.reshape(v, (self.count, -1))
        ind = ind + xp.size(idx)

        ind_pos = ind
        for ell in range(1, self.ell_max + 1):
            idx = ind + xp.arange(self.k_max[ell])
            idx_pos = ind_pos + xp.arange(self.k_max[ell])
            idx_neg = idx_pos + self.k_max[ell]

            v_ell = xp.matmul(radial[:, idx].T, pf[:, ell, :])

            if xp.mod(ell, 2) == 0:
                v_pos = xp.real(v_ell)
                v_neg = -xp.imag(v_ell)
            else:
                v_pos = xp.imag(v_ell)
                v_neg = xp.real(v_ell)

            v[idx_pos, :] = v_pos
            v[idx_neg, :] = v_neg

            ind = ind + xp.size(idx)

            ind_pos = ind_pos + 2 * self.k_max[ell]

        # return v coefficients with the first dimension of self.count
        v = xp.asnumpy(v)
        v = roll_dim(v, sz_roll)
        return v
