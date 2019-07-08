import logging
import numpy as np
from scipy.special import jv
from scipy.sparse.linalg import LinearOperator, cg

from aspyre.utils import ensure
from aspyre.utils.matrix import roll_dim, unroll_dim, im_to_vec, vec_to_im
from aspyre.utils.matlab_compat import m_flatten, m_reshape
from aspyre.basis.basis_func import unique_coords_nd
from aspyre.basis import Basis


logger = logging.getLogger(__name__)


class FBBasis2D(Basis):
    """
    Define a derived class using the Fourier-Bessel basis for mapping 2D images.

    The expansion coefficients of 2D images on this basis are obtained by
    the least squares method. The algorithm is described in the publication:
    Z. Zhao, A. Singer, Fourier-Bessel Rotational Invariant Eigenimages,
    The Journal of the Optical Society of America A, 30 (5), pp. 871-877 (2013).

    """

    # TODO: Methods that return dictionaries should return useful objects instead

    def __init__(self, size, ell_max=None):
        d = len(size)
        ensure(d == 2, 'Only two-dimensional basis functions are supported.')
        ensure(len(set(size)) == 1, 'Only square domains are supported.')
        super().__init__(size, ell_max)

    def _build(self):

        logger.info('Expanding 2D images in spatial domain.')

        # get upper bound of zeros, ells, and ks  of Bessel functions
        self._getfbzeros()

        # calculate total number of basis functions
        self.basis_count = self.k_max[0] + sum(2 * self.k_max[1:])

        # obtain a 2D grid to represent basis functions
        self.basis_coords = unique_coords_nd(self.N, self.d)

        # generate 1D indices for basis functions
        self._indices = self.indices()

        # precompute the basis functions in 2D grids
        self._precomp = self.precomp()

        # get normalized factors
        self._norms = self.norms()

    def indices(self):
        indices_ells = np.zeros(self.basis_count)
        indices_ks = np.zeros(self.basis_count)
        indices_sgns = np.zeros(self.basis_count)

        i = 0
        for ell in range(self.ell_max + 1):
            sgns = (1,) if ell == 0 else (1, -1)
            ks = range(0, self.k_max[ell])

            for sgn in sgns:
                rng = range(i, i + len(ks))
                indices_ells[rng] = ell
                indices_ks[rng] = ks
                indices_sgns[rng] = sgn

                i += len(rng)

        return {
            'ells': indices_ells,
            'ks': indices_ks,
            'sgns': indices_sgns
        }

    def precomp(self):

        r_unique = self.basis_coords['r_unique']
        ang_unique = self.basis_coords['ang_unique']

        ind_radial = 0
        ind_ang = 0

        radial = np.zeros(shape=(len(r_unique), np.sum(self.k_max)))
        ang = np.zeros(shape=(ang_unique.shape[-1], 2 * self.ell_max + 1))

        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                radial[:, ind_radial] = jv(ell, self.r0[k - 1, ell] * r_unique)
                ind_radial += 1

            sgns = (1,) if ell == 0 else (1, -1)
            for sgn in sgns:
                fn = np.cos if sgn == 1 else np.sin
                ang[:, ind_ang] = fn(ell * ang_unique)
                ind_ang += 1

        return {
            'radial': radial,
            'ang': ang
        }

    def norms(self):
        norms = np.zeros(np.sum(self.k_max))
        norm_fn = self.basis_norm_2d

        i = 0
        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                norms[i] = norm_fn(ell, k)
                i += 1

        return norms

    def basis_norm_2d(self, ell, k):
        result = np.abs(jv(ell + 1, self.r0[k - 1, ell])) * np.sqrt(np.pi / 2.) * self.N / 2.
        if ell == 0:
            result *= np.sqrt(2)

        return result

    def evaluate(self, v):
        """
        Evaluate coefficient vector in basis
        :param v: A coefficient vector (or an array of coefficient vectors) to be evaluated.
            The first dimension must equal `self.basis_count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose first dimensions equal `self.z` and the remaining dimensions correspond to
            dimensions two and higher of `v`.
        """
        v, sz_roll = unroll_dim(v, 2)

        r_idx = self.basis_coords['r_idx']
        ang_idx = self.basis_coords['ang_idx']
        mask = m_flatten(self.basis_coords['mask'])

        ind = 0
        ind_radial = 0
        ind_ang = 0

        x = np.zeros(shape=tuple([np.prod(self.sz)] + list(v.shape[1:])))
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            idx_radial = ind_radial + np.arange(0, k_max)
            nrms = self._norms[idx_radial]
            radial = self._precomp['radial'][:, idx_radial]
            radial = radial / nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp['ang'][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max)
                x[mask] += ang_radial @ v[idx]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        x = m_reshape(x, self.sz + x.shape[1:])
        x = roll_dim(x, sz_roll)

        return x

    def evaluate_t(self, v):
        """
        Evaluate coefficient in dual basis
        :param v: The coefficient array to be evaluated. The first dimensions must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual basis of `basis`.
            This is an array of vectors whose first dimension equals `self.basis_count` and whose remaining dimensions
            correspond to higher dimensions of `v`.
        """
        x, sz_roll = unroll_dim(v, self.d + 1)
        x = m_reshape(x, new_shape=tuple([np.prod(self.sz)] + list(x.shape[self.d:])))

        r_idx = self.basis_coords['r_idx']
        ang_idx = self.basis_coords['ang_idx']
        mask = m_flatten(self.basis_coords['mask'])

        ind = 0
        ind_radial = 0
        ind_ang = 0

        v = np.zeros(shape=tuple([self.basis_count] + list(x.shape[1:])))
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            idx_radial = ind_radial + np.arange(0, k_max)
            nrms = self._norms[idx_radial]
            radial = self._precomp['radial'][:, idx_radial]
            radial = radial / nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp['ang'][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max)
                v[idx] = ang_radial.T @ x[mask]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        v = roll_dim(v, sz_roll)
        return v

    def expand_t(self, v):
        ensure(v.shape[0] == self.basis_count, f'First dimension of v must be {self.basis_count}')

        v, sz_roll = unroll_dim(v, 2)
        b = im_to_vec(self.evaluate(v))

        operator = LinearOperator(
            shape=(self.N**2, self.N**2),
            matvec=lambda x: im_to_vec(self.evaluate(self.evaluate_t(vec_to_im(x))))
        )

        # TODO: (from MATLAB implementation) - Check that this tolerance make sense for multiple columns in v
        tol = 10 * np.finfo(v.dtype).eps
        logger.info('Expanding array in dual basis')
        v, info = cg(operator, b, tol=tol)

        if info != 0:
            raise RuntimeError('Unable to converge!')

        v = roll_dim(v, sz_roll)
        x = vec_to_im(v)

        return x

