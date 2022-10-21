import logging

import numpy as np
import scipy.sparse as sparse
from scipy.fft import dct, idct
from scipy.special import jv

from aspire.basis import FBBasisMixin, SteerableBasis2D
from aspire.basis.basis_utils import besselj_zeros
from aspire.nufft import nufft, anufft
from aspire.numeric import fft

logger = logging.getLogger(__name__)


class FLEBasis2D(SteerableBasis2D, FBBasisMixin):
    """
    FLE Basis.

    https://arxiv.org/pdf/2207.13674.pdf
    """

    def __init__(self, size, bandlimit=None, epsilon=1e-10, dtype=np.float32):
        """
        :param size: The size of the vectors for which to define the FLE basis.
                 Currently only square images are supported.
        :param bandlimit: Maximum frequency band for computing basis functions. Note that the
            `ell_max` of other Basis objects is computed *from* the bandlimit for the FLE basis.
             Defaults to the resolution of the basis.
        :param epsilon: Relative precision between FLE fast method and dense matrix multiplication.
        """
        if isinstance(size, int):
            size = (size, size)
        ndim = len(size)
        assert ndim == 2, "Only two-dimensional basis functions are supported."
        assert len(set(size)) == 1, "Only square domains are supported"

        self.bandlimit = bandlimit
        self.epsilon = epsilon
        self.dtype = dtype
        super().__init__(size, ell_max=None, dtype=self.dtype)

    def _build(self):
        # get upper bound of zeros, ells, and ks of Bessel functions
        # self._calc_k_max()
        # self.count = self.k_max[0] + sum(2*self.k_max[1:])

        if not self.bandlimit:
            self.bandlimit = self.nres

        # Heuristic for max iterations
        maxitr = 1 + int(3 * np.log2(self.nres))
        numsparse = 32
        if self.epsilon >= 1e-10:
            numsparse = 22
            maxitr = 1 + int(2 * np.log2(self.nres))
        if self.epsilon >= 1e-7:
            numsparse = 16
            maxitr = 1 + int(np.log2(self.nres))
        if self.epsilon >= 1e-4:
            numsparse = 8
            maxitr = 1 + int(np.log2(self.nres)) // 2
        self.maxitr = maxitr
        self.numsparse = numsparse

        # Compute grid points
        self.R = self.nres // 2
        self.h = 1 / self.R
        x = np.arange(-self.R, self.R + self.nres % 2)
        y = np.arange(-self.R, self.R + self.nres % 2)
        xs, ys = np.meshgrid(x, y)
        self.xs, self.ys, = (
            xs / self.R,
            ys / self.R,
        )
        self.rs = np.sqrt(self.xs**2 + self.ys**2)
        # radial mask to remove energy outside disk
        self.radial_mask = self.rs > 1 + 1e-13

        #
        self._precomp()

    def _precomp(self):

        # Regular Fourier-Bessel bandlimit (equivalent to pi*R**2)
        self.max_basis_functions = int(self.nres**2 * np.pi / 4)

        # Compute basis functions
        self._lap_eig_disk()

        # Some important constants
        self.smallest_lambda = np.min(self.bessel_zeros)
        self.greatest_lambda = np.max(self.bessel_zeros)
        self.nmax = np.max(np.abs(self.ells))
        # TODO: explain
        self.ndx = 2 * np.abs(self.ells) - (self.ells < 0)
        self.ndmax = np.max(self.ndx)
        idx_list = [[] for i in range(self.ndmax + 1)]
        for i in range(self.count):
            nd = self.ndx[i]
            idx_list[nd].append(i)
        self.idx_list = idx_list

        self.nus = np.zeros(1 + 2 * self.nmax, dtype=int)
        self.nus[0] = 0
        for i in range(1, self.nmax + 1):
            self.nus[2 * i - 1] = -i
            self.nus[2 * i] = i
        self.c2r_nus = self.precomp_transform_complex_to_real(self.nus)
        self.r2c_nus = sparse.csr_matrix(self.c2r_nus.transpose().conj())

        # radial and angular nodes for NUFFT
        self._compute_nufft_points()
        self.num_interp = self.num_radial_nodes
        if self.numsparse > 0:
            self.num_interp = 2 * self.num_radial_nodes

        self._build_interpolation_matrix()

    def _compute_nufft_points(self):
        """
        Compute the number of radial and angular nodes for the non-uniform FFT.
        """

        # Number of radial nodes
        # (Lemma 4.1)
        # compute max {2.4 * self.nres , Log2 ( 1 / epsilon) }
        Q = int(np.ceil(2.4 * self.nres))
        num_radial_nodes = Q
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * self.nres / 4)
            if tmp <= self.epsilon:
                num_radial_nodes = int(max(q, np.log2(1 / self.epsilon)))
                break
        self.num_radial_nodes = max(
            num_radial_nodes, int(np.ceil(np.log2(1 / self.epsilon)))
        )

        # Number of angular nodes
        # (Lemma 4.2)
        # compute max {7.08 * self.nres, Log2(1/epsilon) + Log2(self.nres**2) }

        S = int(max(7.08 * self.nres, -np.log2(self.epsilon) + 2 * np.log2(self.nres)))
        num_angular_nodes = S
        for s in range(int(self.greatest_lambda + self.ndmax) + 1, S + 1):
            tmp = self.nres**2 * ((self.greatest_lambda + self.ndmax) / s) ** s
            if tmp <= self.epsilon:
                num_angular_nodes = int(max(int(s), np.log2(1 / self.epsilon)))
                break

        # must be even
        if num_angular_nodes % 2 == 1:
            num_angular_nodes += 1

        self.num_angular_nodes = num_angular_nodes

        # create gridpoints
        nodes = 1 - (2 * np.arange(self.num_radial_nodes) + 1) / (
            2 * self.num_radial_nodes
        )
        nodes = (np.cos(np.pi * nodes) + 1) / 2
        nodes = (
            self.greatest_lambda - self.smallest_lambda
        ) * nodes + self.smallest_lambda
        nodes = nodes.reshape(-1, 1)

        radius = self.nres // 2
        h = 1 / radius

        phi = (
            2 * np.pi * np.arange(self.num_angular_nodes // 2) / self.num_angular_nodes
        )
        x = np.cos(phi).reshape(1, -1)
        y = np.sin(phi).reshape(1, -1)
        x = x * nodes * h
        y = y * nodes * h
        self.grid_x = x.flatten()
        self.grid_y = y.flatten()

    def _build_interpolation_matrix(self):
        A3 = [None] * (self.ndmax + 1)
        A3_T = [None] * (self.ndmax + 1)
        chebyshev_pts = np.cos(
            np.pi * (1 - (2 * np.arange(self.num_interp) + 1) / (2 * self.num_interp))
        )
        weights = self.get_weights(chebyshev_pts)
        for i in range(self.ndmax + 1):
            ys = np.zeros(self.num_interp)

            # target points
            x = (
                2
                * (self.bessel_zeros[self.idx_list[i]] - self.smallest_lambda)
                / (self.greatest_lambda - self.smallest_lambda)
                - 1
            )
            vals, x_ind, xs_ind = np.intersect1d(x, chebyshev_pts, return_indices=True)
            x[x_ind] = x[x_ind] + 2e-16

            n = len(x)
            mm = len(chebyshev_pts)
            if self.numsparse > 0:
                A3[i], A3_T[i] = self.barycentric_interp_sparse(
                    x, chebyshev_pts, ys, self.numsparse
                )
            else:
                A3[i] = np.zeros((n, mm))
                denom = np.zeros(n)
                for j in range(mm):
                    xdiff = x - chebyshev_pts[j]
                    temp = weights[j] / xdiff
                    A3[i][:, j] = temp.flatten()
                    denom = denom + temp
                denom = denom.reshape(-1, 1)
                A3[i] = A3[i] / denom
                A3_T[i] = A3[i].T
        self.A3 = A3
        self.A3_T = A3_T

    def _lap_eig_disk(self):
        """
        Compute the eigenvalues of the Laplacian operator on a disk with Dirichlet boundary conditions.
        """
        # max number of Bessel function orders being considered
        max_ell = int(3 * np.sqrt(self.max_basis_functions))
        # max number of zeros per Bessel function (number of frequencies per bessel)
        max_k = int(2 * np.sqrt(self.max_basis_functions))

        # preallocate containers for roots
        # 0 frequency plus pos and negative frequencies for each bessel function
        # num functions per frequency
        num_ells = 1 + 2 * max_ell
        self.ells = np.zeros((num_ells, max_k), dtype=int, order="F")
        self.ks = np.zeros((num_ells, max_k), dtype=int, order="F")
        self.bessel_zeros = np.ones((num_ells, max_k), dtype=np.float64) * np.Inf

        # keep track of which order Bessel function we're on
        self.ells[0, :] = 0
        # bessel_roots[0, m] is the m'th zero of J_0
        self.bessel_zeros[0, :] = besselj_zeros(0, max_k)
        # table of values of which zero of J_0 we are finding
        self.ks[0, :] = np.arange(max_k) + 1

        # add roots of J_ell for ell>0 twice with +k and -k (frequencies)
        # iterate over Bessel function order
        for ell in range(1, max_ell + 1):
            self.ells[2 * ell - 1, :] = -ell
            self.ks[2 * ell - 1, :] = np.arange(max_k) + 1

            self.bessel_zeros[2 * ell - 1, :max_k] = besselj_zeros(ell, max_k)

            self.ells[2 * ell, :] = ell
            self.ks[2 * ell, :] = self.ks[2 * ell - 1, :]
            self.bessel_zeros[2 * ell, :] = self.bessel_zeros[2 * ell - 1, :]

        # Reshape the arrays and order by the size of the Bessel function zeros
        self._flatten_and_sort_bessel_zeros()

        # Apply threshold criterion to throw out some basis functions
        # Grab final number of basis functions for this Basis
        self.count = self._threshold_basis_functions()

        self._create_basis_functions()

    def _flatten_and_sort_bessel_zeros(self):
        """
        Reshapes arrays self.ells, self.ks, and self.bessel_zeros
        """
        # flatten list of zeros, ells and ks:
        self.ells = self.ells.flatten()
        self.ks = self.ks.flatten()
        self.bessel_zeros = self.bessel_zeros.flatten()

        # TODO: Better way of doing the next two sections
        # (Specifically ordering the neg and pos integers in the correct way)
        # sort by size of zeros
        idx = np.argsort(self.bessel_zeros)
        self.ells = self.ells[idx]
        self.ks = self.ks[idx]
        self.bessel_zeros = self.bessel_zeros[idx]

        # sort complex conjugate pairs: -n first, +n second
        idx = np.arange(self.max_basis_functions + 1)
        for i in range(self.max_basis_functions + 1):
            if self.ells[i] >= 0:
                continue
            if np.abs(self.bessel_zeros[i] - self.bessel_zeros[i + 1]) < 1e-14:
                continue
            idx[i - 1] = i
            idx[i] = i - 1

        self.ells = self.ells[idx]
        self.ks = self.ks[idx]
        self.bessel_zeros = self.bessel_zeros[idx]

    def _threshold_basis_functions(self):
        """
        Implements the bandlimit threshold which caps the number of basis functions
        that are actually required.
        :return: The final overall number of basis functions to be used.
        """
        # Maximum bandlimit
        # (Section 4.1)
        # Can remove frequencies above this threshold based on the fact that
        # there should not be more basis functions than pixels contained in the
        # unit disk inscribed on the image
        _final_num_basis_functions = self.max_basis_functions
        if self.bandlimit:
            for _ in range(len(self.bessel_zeros)):
                if (
                    self.bessel_zeros[_final_num_basis_functions] / (np.pi)
                    >= (self.bandlimit - 1) // 2
                ):
                    _final_num_basis_functions -= 1

        # potentially subtract one to keep complex conjugate pairs
        if self.ells[_final_num_basis_functions - 1] < 0:
            _final_num_basis_functions -= 1

        # discard zeros above the threshold
        self.ells = self.ells[:_final_num_basis_functions]
        self.ks = self.ks[:_final_num_basis_functions]
        self.bessel_zeros = self.bessel_zeros[:_final_num_basis_functions]

        return _final_num_basis_functions

    def _create_basis_functions(self):
        """
        Generate the actual basis functions as Python lambda operators
        """
        norm_constants = np.zeros(self.count)
        basis_functions = [None] * self.count
        for i in range(self.count):
            # parameters defining the basis function: bessel order and which bessel root
            ell = self.ells[i]
            bessel_zero = self.bessel_zeros[i]

            # compute normalization constant
            # see Eq. 6
            c = 1 / np.sqrt(np.pi * jv(ell + 1, bessel_zero) ** 2)
            # create function
            # See Eq. 1
            if ell == 0:
                basis_functions[i] = (
                    lambda r, t, c=c, ell=ell, bessel_zero=bessel_zero: c
                    * jv(ell, bessel_zero * r)
                    * (r <= 1)
                )
            else:
                basis_functions[i] = (
                    lambda r, t, c=c, ell=ell, bessel_zero=bessel_zero: c
                    * jv(ell, bessel_zero * r)
                    * np.exp(1j * ell * t)
                    * (-1) ** np.abs(ell)
                    * (r <= 1)
                )

            norm_constants[i] = c

        self.norm_constants = norm_constants
        self.basis_functions = basis_functions

    def _evaluate(self, coeffs):
        """
        Placeholder.

        Evaluate FLE coefficients and return in standard 2D Cartesian coordinates.

        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The last dimension must be equal to `self.count`
        """
        return np.zeros((coeffs.shape[0], self.nres, self.nres))

    def _evaluate_t(self, imgs):
        """
        Placeholder.

        Evaluate 2D Cartesian image(s) and return the corresponding FLE coefficients.

        :param imgs: The array to be evaluated. The last dimensions
            must equal `self.sz`
        """
        imgs = imgs.asnumpy()
        imgs[:, self.radial_mask] = 0
        z = self._step1_t(imgs)
        b = self._step2_t(z)
        coeffs = self._step3_t(b)

        return coeffs

    def _step1_t(self, im):
        """
        Step 1 of the adjoint transformation (images to coefficients).
        Calculates the NUFFT of the image on gridpoints self.grid_x and self.grid_y.
        """
        im = im.reshape(-1, self.nres, self.nres).astype(np.complex128)
        num_img = im.shape[0]
        z = np.zeros(
            (num_img, self.num_radial_nodes, self.num_angular_nodes),
            dtype=np.complex128,
        )
        _z = (
            nufft(im, np.stack((self.grid_x, self.grid_y)), epsilon=self.epsilon)
            * self.h**2
        )
        _z = _z.reshape(-1, self.num_radial_nodes, self.num_angular_nodes // 2)
        z[:, :, : self.num_angular_nodes // 2] = _z
        z[:, :, self.num_angular_nodes // 2 :] = np.conj(_z)
        z = z.reshape(-1, self.num_radial_nodes * self.num_angular_nodes)
        return z

    def _step2_t(self, z):
        """
        Compute values of the analytic functions Beta_n at the Chebyshev nodes.
        See Lemma 2.2.
        """
        z = z.reshape(-1, self.num_radial_nodes, self.num_angular_nodes)
        num_img = z.shape[0]
        # Compute FFT along angular nodes
        betas = fft.fft(z, axis=2) / self.num_angular_nodes
        betas = betas[:, :, self.nus]
        betas = np.conj(betas)
        betas = np.swapaxes(betas, 0, 2)
        betas = betas.reshape(-1, self.num_radial_nodes * num_img)
        betas = self.c2r_nus @ betas
        betas = betas.reshape(-1, self.num_radial_nodes, num_img)
        betas = np.real(np.swapaxes(betas, 0, 2))
        return betas

    def _step3_t(self, betas):
        """
        Use barycenteric interpolation to compute the values of the Betas
        at the Bessel roots to arrive at the Fourier-Bessel coefficients.
        """
        num_img = betas.shape[0]
        if self.num_interp > self.num_radial_nodes:
            betas = dct(betas, axis=1, type=2) / (2 * self.num_radial_nodes)
            zeros = np.zeros(betas.shape)
            betas = np.concatenate((betas, zeros), axis=1)
            betas = idct(betas, axis=1, type=2) * 2 * betas.shape[1]
        betas = np.moveaxis(betas, 0, -1)

        coeffs = np.zeros((self.count, num_img), dtype=np.float64)
        for i in range(self.ndmax + 1):
            coeffs[self.idx_list[i]] = self.A3[i] @ betas[:, i, :]
        coeffs = coeffs.T

        return coeffs * self.norm_constants / self.h

    def _step3(self, coeffs):

        coeffs = coeffs.reshape(-1, self.count)
        num_img = coeffs.shape[0]
        coeffs *= self.h * self.norm_constants
        coeffs = coeffs.T

        out = np.zeros(
            (self.num_interp, 2 * self.nmax + 1, num_img), dtype=np.float64, order="F"
        )
        for i in range(self.ndmax + 1):
            out[:, i, :] = self.A3_T[i] @ coeffs[self.idx_list[i]]
        out = np.moveaxis(out, -1, 0)
        if self.num_interp > self.num_radial_nodes:
            out = dct(out, axis=1, type=2)
            out = out[:, : self.num_radial_nodes, :]
            out = idct(out, axis=1, type=2)

        return out

    def _step2(self, betas):
        num_img = betas.shape[0]
        tmp = np.zeros(
            (num_img, self.num_radial_nodes, self.num_angular_nodes),
            dtype=np.complex128,
        )

        betas = np.swapaxes(betas, 0, 2)
        betas = betas.reshape(-1, self.num_radial_nodes * num_img)
        betas = self.r2c_nus @ betas
        betas = betas.reshape(-1, self.num_radial_nodes, num_img)
        betas = np.swapaxes(betas, 0, 2)

        tmp[:, :, self.nus] = np.conj(betas)
        z = fft.ifft(tmp, axis=2)

        return z

    def _step1(self, z):
        num_img = z.shape[0]
        z = z[:, :, : self.num_angular_nodes // 2].reshape(num_img, -1)
        im = anufft(
            z,
            np.stack((self.grid_x, self.grid_y)),
            (self.nres, self.nres),
            epsilon=self.epsilon,
        )
        im = im + np.conj(im)
        im = np.real(im)
        im = im.reshape(num_img, self.nres, self.nres)
        im[:, self.radial_mask] = 0

        return im

    def create_dense_matrix(self):
        ts = np.arctan2(self.ys, self.xs)

        B = np.zeros((self.nres, self.nres, self.count), dtype=np.complex128, order="F")
        for i in range(self.count):
            B[:, :, i] = self.basis_functions[i](self.rs, ts) * self.h
        B = B.reshape(self.nres**2, self.count)
        B = self._transform_complex_to_real(np.conj(B), self.ells)
        return B.reshape(self.nres**2, self.count)

    def _transform_complex_to_real(self, Z, ns):
        """
        Transforms coefficients of the matrix B (see Eq. 3) from complex
        to real. B is the linear transformation that takes FB coefficients
        to images.
        """
        ne = Z.shape[1]
        X = np.zeros(Z.shape, dtype=np.float64)

        for i in range(ne):
            n = ns[i]
            if n == 0:
                X[:, i] = np.real(Z[:, i])
            if n < 0:
                s = (-1) ** np.abs(n)
                x0 = (Z[:, i] + s * Z[:, i + 1]) / np.sqrt(2)
                x1 = (-Z[:, i] + s * Z[:, i + 1]) / (1j * np.sqrt(2))
                X[:, i] = np.real(x0)
                X[:, i + 1] = np.real(x1)

        return X

    def precomp_transform_complex_to_real(self, ns):

        ne = len(ns)
        nnz = np.sum(ns == 0) + 2 * np.sum(ns != 0)
        idx = np.zeros(nnz, dtype=int)
        jdx = np.zeros(nnz, dtype=int)
        vals = np.zeros(nnz, dtype=np.complex128)

        k = 0
        for i in range(ne):
            n = ns[i]
            if n == 0:
                vals[k] = 1
                idx[k] = i
                jdx[k] = i
                k = k + 1
            if n < 0:
                s = (-1) ** np.abs(n)

                vals[k] = 1 / np.sqrt(2)
                idx[k] = i
                jdx[k] = i
                k = k + 1

                vals[k] = s / np.sqrt(2)
                idx[k] = i
                jdx[k] = i + 1
                k = k + 1

                vals[k] = -1 / (1j * np.sqrt(2))
                idx[k] = i + 1
                jdx[k] = i
                k = k + 1

                vals[k] = s / (1j * np.sqrt(2))
                idx[k] = i + 1
                jdx[k] = i + 1
                k = k + 1

        A = sparse.csr_matrix((vals, (idx, jdx)), shape=(ne, ne), dtype=np.complex128)

        return A

    def barycentric_interp_sparse(self, x, xs, ys, s):
        # https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

        n = len(x)
        m = len(xs)

        # Modify points by 2e-16 to avoid division by zero
        vals, x_ind, xs_ind = np.intersect1d(
            x, xs, return_indices=True, assume_unique=True
        )
        x[x_ind] = x[x_ind] + 2e-16

        idx = np.zeros((n, s))
        jdx = np.zeros((n, s))
        vals = np.zeros((n, s))
        xss = np.zeros((n, s))
        denom = np.zeros((n, 1))
        temp = np.zeros((n, 1))
        ws = np.zeros((n, s))
        xdiff = np.zeros(n)
        for i in range(n):

            # get a kind of blanced interval around our point
            k = np.searchsorted(x[i] < xs, True)

            idp = np.arange(k - s // 2, k + (s + 1) // 2)
            if idp[0] < 0:
                idp = np.arange(s)
            if idp[-1] >= m:
                idp = np.arange(m - s, m)
            xss[i, :] = xs[idp]
            jdx[i, :] = idp
            idx[i, :] = i

        x = x.reshape(-1, 1)
        Iw = np.ones(s, dtype=bool)
        ew = np.zeros((n, 1))
        xtw = np.zeros((n, s - 1))

        Iw[0] = False
        const = np.zeros((n, 1))
        for _ in range(s):
            ew = np.sum(-np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1)
            constw = np.exp(ew / s)
            constw = constw.reshape(-1, 1)
            const += constw
        const = const / s

        for j in range(s):
            Iw[j] = False
            xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
            ws[:, j] = 1 / np.prod(xtw, axis=1)
            Iw[j] = True

        xdiff = xdiff.flatten()
        x = x.flatten()
        temp = temp.flatten()
        denom = denom.flatten()
        for j in range(s):
            xdiff = x - xss[:, j]
            temp = ws[:, j] / xdiff
            vals[:, j] = vals[:, j] + temp
            denom = denom + temp
        vals = vals / denom.reshape(-1, 1)

        vals = vals.flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()
        A = sparse.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=np.float64)
        A_T = sparse.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)

        return A, A_T

    def get_weights(self, xs):

        m = len(xs)
        ident = np.ones(m, dtype=bool)
        ident[0] = False
        e = np.sum(-np.log(np.abs(xs[0] - xs[ident])))
        const = np.exp(e / m)
        ws = np.zeros(m)
        ident = np.ones(m, dtype=bool)
        for j in range(m):
            ident[j] = False
            xt = const * (xs[j] - xs[ident])
            ws[j] = 1 / np.prod(xt)
            ident[j] = True

        return ws
