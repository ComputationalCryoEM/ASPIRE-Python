import logging

import numpy as np
from numpy import pi

from aspire.basis import FBBasis3D
from aspire.basis.basis_utils import lgwt, norm_assoc_legendre, sph_bessel
from aspire.nufft import anufft, nufft
from aspire.utils.matlab_compat import m_flatten, m_reshape

logger = logging.getLogger(__name__)


class FFBBasis3D(FBBasis3D):
    """
    Define a derived class for fast spherical Harmonics Bessel basis expanding 3D volumes

    # TODO: Methods that return dictionaries should return useful objects instead

    """

    def _build(self):
        """
        Build the internal data structure for 3D Fourier-Bessel basis
        """
        logger.info(
            "Expanding 3D map in a frequency-domain Fourier–Bessel"
            " basis using the fast method."
        )

        # set cutoff values
        self.rcut = self.nres / 2
        self.kcut = 0.5

        # get upper bound of zeros, ells, and ks  of Bessel functions
        self._getfbzeros()

        # calculate total number of basis functions
        self.count = sum(self.k_max * (2 * np.arange(0, self.ell_max + 1) + 1))

        # generate 1D indices for basis functions
        self._indices = self.indices()

        # precompute the basis functions in 3D grids
        self._precomp = self._precomp()

        # get normalized factors
        self._norms = self.norms()

    def _precomp(self):
        """
        Precomute the basis functions on a polar Fourier 3D grid

        Gaussian quadrature points and weights are also generated
        in radical and phi dimensions.
        """
        n_r = int(self.ell_max + 1)
        n_theta = int(2 * self.sz[0])
        n_phi = int(self.ell_max + 1)

        r, wt_r = lgwt(n_r, 0.0, self.kcut, dtype=self.dtype)
        z, wt_z = lgwt(n_phi, -1, 1, dtype=self.dtype)
        r = m_reshape(r, (n_r, 1))
        wt_r = m_reshape(wt_r, (n_r, 1))
        z = m_reshape(z, (n_phi, 1))
        wt_z = m_reshape(wt_z, (n_phi, 1))
        phi = np.arccos(z)
        wt_phi = wt_z
        theta = 2 * pi * np.arange(n_theta, dtype=self.dtype).T / (2 * n_theta)
        theta = m_reshape(theta, (n_theta, 1))

        # evaluate basis function in the radial dimension
        radial_wtd = np.zeros(
            shape=(n_r, np.max(self.k_max), self.ell_max + 1), dtype=self.dtype
        )
        for ell in range(0, self.ell_max + 1):
            k_max_ell = self.k_max[ell]
            rmat = r * self.r0[0:k_max_ell, ell].T / self.kcut
            radial_ell = np.zeros_like(rmat)
            for ik in range(0, k_max_ell):
                radial_ell[:, ik] = sph_bessel(ell, rmat[:, ik])
            nrm = np.abs(sph_bessel(ell + 1, self.r0[0:k_max_ell, ell].T) / 4)
            radial_ell = radial_ell / nrm
            radial_ell_wtd = r ** 2 * wt_r * radial_ell
            radial_wtd[:, 0:k_max_ell, ell] = radial_ell_wtd

        # evaluate basis function in the phi dimension
        ang_phi_wtd_even = []
        ang_phi_wtd_odd = []
        for m in range(0, self.ell_max + 1):
            n_even_ell = int(
                np.floor((self.ell_max - m) / 2)
                + 1
                - np.mod(self.ell_max, 2) * np.mod(m, 2)
            )
            n_odd_ell = int(self.ell_max - m + 1 - n_even_ell)
            phi_wtd_m_even = np.zeros((n_phi, n_even_ell), dtype=phi.dtype)
            phi_wtd_m_odd = np.zeros((n_phi, n_odd_ell), dtype=phi.dtype)

            ind_even = 0
            ind_odd = 0
            for ell in range(m, self.ell_max + 1):
                phi_m_ell = norm_assoc_legendre(ell, m, z)
                nrm_inv = np.sqrt(0.5 / pi)
                phi_m_ell = nrm_inv * phi_m_ell
                phi_wtd_m_ell = wt_phi * phi_m_ell
                if np.mod(ell, 2) == 0:
                    phi_wtd_m_even[:, ind_even] = phi_wtd_m_ell[:, 0]
                    ind_even = ind_even + 1
                else:
                    phi_wtd_m_odd[:, ind_odd] = phi_wtd_m_ell[:, 0]
                    ind_odd = ind_odd + 1

            ang_phi_wtd_even.append(phi_wtd_m_even)
            ang_phi_wtd_odd.append(phi_wtd_m_odd)

        # evaluate basis function in the theta dimension
        ang_theta = np.zeros((n_theta, 2 * self.ell_max + 1), dtype=theta.dtype)

        ang_theta[:, 0 : self.ell_max] = np.sqrt(2) * np.sin(
            theta @ m_reshape(np.arange(self.ell_max, 0, -1), (1, self.ell_max))
        )
        ang_theta[:, self.ell_max] = np.ones(n_theta, dtype=theta.dtype)
        ang_theta[:, self.ell_max + 1 : 2 * self.ell_max + 1] = np.sqrt(2) * np.cos(
            theta @ m_reshape(np.arange(1, self.ell_max + 1), (1, self.ell_max))
        )

        ang_theta_wtd = (2 * pi / n_theta) * ang_theta

        theta_grid, phi_grid, r_grid = np.meshgrid(
            theta, phi, r, sparse=False, indexing="ij"
        )
        fourier_x = m_flatten(r_grid * np.cos(theta_grid) * np.sin(phi_grid))
        fourier_y = m_flatten(r_grid * np.sin(theta_grid) * np.sin(phi_grid))
        fourier_z = m_flatten(r_grid * np.cos(phi_grid))
        fourier_pts = (
            2
            * pi
            * np.vstack(
                (
                    fourier_z[np.newaxis, ...],
                    fourier_y[np.newaxis, ...],
                    fourier_x[np.newaxis, ...],
                )
            )
        )

        return {
            "radial_wtd": radial_wtd,
            "ang_phi_wtd_even": ang_phi_wtd_even,
            "ang_phi_wtd_odd": ang_phi_wtd_odd,
            "ang_theta_wtd": ang_theta_wtd,
            "fourier_pts": fourier_pts,
        }

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 3D coordinate basis from those in 3D FB basis

        :param v: A coefficient vector (or an array of coefficient vectors) in FB basis
            to be evaluated. The last dimension must equal `self.count`.
        :return x: The evaluation of the coefficient vector(s) `x` in standard 3D
            coordinate basis. This is an array whose last three dimensions equal
            `self.sz` and the remaining dimensions correspond to `v`.
        """
        # roll dimensions of v
        sz_roll = v.shape[:-1]
        v = v.reshape((-1, self.count))

        # get information on polar grids from precomputed data
        n_theta = np.size(self._precomp["ang_theta_wtd"], 0)
        n_phi = np.size(self._precomp["ang_phi_wtd_even"][0], 0)
        n_r = np.size(self._precomp["radial_wtd"], 0)

        # number of 3D image samples
        n_data = v.shape[0]

        u_even = np.zeros(
            (
                n_r,
                int(2 * self.ell_max + 1),
                n_data,
                int(np.floor(self.ell_max / 2) + 1),
            ),
            dtype=v.dtype,
        )
        u_odd = np.zeros(
            (n_r, int(2 * self.ell_max + 1), n_data, int(np.ceil(self.ell_max / 2))),
            dtype=v.dtype,
        )

        # go through each basis function and find corresponding coefficient
        # evaluate the radial parts
        for ell in range(0, self.ell_max + 1):
            k_max_ell = self.k_max[ell]
            radial_wtd = self._precomp["radial_wtd"][:, 0:k_max_ell, ell]

            ind = self._indices["ells"] == ell

            v_ell = m_reshape(v[:, ind].T, (k_max_ell, (2 * ell + 1) * n_data))
            v_ell = radial_wtd @ v_ell
            v_ell = m_reshape(v_ell, (n_r, 2 * ell + 1, n_data))

            if np.mod(ell, 2) == 0:
                u_even[
                    :,
                    int(self.ell_max - ell) : int(self.ell_max + ell + 1),
                    :,
                    int(ell / 2),
                ] = v_ell
            else:
                u_odd[
                    :,
                    int(self.ell_max - ell) : int(self.ell_max + ell + 1),
                    :,
                    int((ell - 1) / 2),
                ] = v_ell

        u_even = np.transpose(u_even, (3, 0, 1, 2))
        u_odd = np.transpose(u_odd, (3, 0, 1, 2))
        w_even = np.zeros((n_phi, n_r, n_data, 2 * self.ell_max + 1), dtype=v.dtype)
        w_odd = np.zeros((n_phi, n_r, n_data, 2 * self.ell_max + 1), dtype=v.dtype)

        # evaluate the phi parts
        for m in range(0, self.ell_max + 1):
            ang_phi_wtd_m_even = self._precomp["ang_phi_wtd_even"][m]
            ang_phi_wtd_m_odd = self._precomp["ang_phi_wtd_odd"][m]

            n_even_ell = np.size(ang_phi_wtd_m_even, 1)
            n_odd_ell = np.size(ang_phi_wtd_m_odd, 1)

            if m == 0:
                sgns = (1,)
            else:
                sgns = (1, -1)

            for sgn in sgns:

                end = np.size(u_even, 0)
                u_m_even = u_even[end - n_even_ell : end, :, self.ell_max + sgn * m, :]
                end = np.size(u_odd, 0)
                u_m_odd = u_odd[end - n_odd_ell : end, :, self.ell_max + sgn * m, :]

                u_m_even = m_reshape(u_m_even, (n_even_ell, n_r * n_data))
                u_m_odd = m_reshape(u_m_odd, (n_odd_ell, n_r * n_data))

                w_m_even = ang_phi_wtd_m_even @ u_m_even
                w_m_odd = ang_phi_wtd_m_odd @ u_m_odd

                w_m_even = m_reshape(w_m_even, (n_phi, n_r, n_data))
                w_m_odd = m_reshape(w_m_odd, (n_phi, n_r, n_data))

                w_even[:, :, :, self.ell_max + sgn * m] = w_m_even
                w_odd[:, :, :, self.ell_max + sgn * m] = w_m_odd

        w_even = np.transpose(w_even, (3, 0, 1, 2))
        w_odd = np.transpose(w_odd, (3, 0, 1, 2))
        u_even = w_even
        u_odd = w_odd

        u_even = m_reshape(u_even, (2 * self.ell_max + 1, n_phi * n_r * n_data))
        u_odd = m_reshape(u_odd, (2 * self.ell_max + 1, n_phi * n_r * n_data))

        # evaluate the theta parts
        w_even = self._precomp["ang_theta_wtd"] @ u_even
        w_odd = self._precomp["ang_theta_wtd"] @ u_odd

        pf = w_even + 1j * w_odd
        pf = m_reshape(pf, (n_theta * n_phi * n_r, n_data))
        pf = np.moveaxis(pf, 0, -1)

        # perform inverse non-uniformly FFT transformation back to 3D rectangular coordinates
        freqs = m_reshape(self._precomp["fourier_pts"], (3, n_r * n_theta * n_phi))
        x = anufft(pf, freqs, self.sz, real=True)

        # Roll, return the x with the last three dimensions as self.sz
        # Higher dimensions should be like v.
        x = x.reshape((*sz_roll, *self.sz))
        return x

    def evaluate_t(self, x):
        """
        Evaluate coefficient in FB basis from those in standard 3D coordinate basis

        :param x: The coefficient array in the standard 3D coordinate basis
            to be evaluated. The last three dimensions must equal `self.sz`.
        :return v: The evaluation of the coefficient array `v` in the FB basis.
            This is an array of vectors whose last dimension equals
            `self.count` and whose remaining dimensions correspond to higher
            dimensions of `x`.
        """
        # roll dimensions
        sz_roll = x.shape[:-3]
        x = x.reshape((-1, *self.sz))

        n_data = x.shape[0]
        n_r = np.size(self._precomp["radial_wtd"], 0)
        n_phi = np.size(self._precomp["ang_phi_wtd_even"][0], 0)
        n_theta = np.size(self._precomp["ang_theta_wtd"], 0)

        # resamping x in a polar Fourier gird using nonuniform discrete Fourier transform
        pf = nufft(x, self._precomp["fourier_pts"])

        pf = m_reshape(pf.T, (n_theta, n_phi * n_r * n_data))

        # evaluate the theta parts
        u_even = self._precomp["ang_theta_wtd"].T @ np.real(pf)
        u_odd = self._precomp["ang_theta_wtd"].T @ np.imag(pf)

        u_even = m_reshape(u_even, (2 * self.ell_max + 1, n_phi, n_r, n_data))
        u_odd = m_reshape(u_odd, (2 * self.ell_max + 1, n_phi, n_r, n_data))

        u_even = np.transpose(u_even, (1, 2, 3, 0))
        u_odd = np.transpose(u_odd, (1, 2, 3, 0))

        w_even = np.zeros(
            (int(np.floor(self.ell_max / 2) + 1), n_r, 2 * self.ell_max + 1, n_data),
            dtype=x.dtype,
        )
        w_odd = np.zeros(
            (int(np.ceil(self.ell_max / 2)), n_r, 2 * self.ell_max + 1, n_data),
            dtype=x.dtype,
        )

        # evaluate the phi parts
        for m in range(0, self.ell_max + 1):
            ang_phi_wtd_m_even = self._precomp["ang_phi_wtd_even"][m]
            ang_phi_wtd_m_odd = self._precomp["ang_phi_wtd_odd"][m]

            n_even_ell = np.size(ang_phi_wtd_m_even, 1)
            n_odd_ell = np.size(ang_phi_wtd_m_odd, 1)

            if m == 0:
                sgns = (1,)
            else:
                sgns = (1, -1)

            for sgn in sgns:
                u_m_even = u_even[:, :, :, self.ell_max + sgn * m]
                u_m_odd = u_odd[:, :, :, self.ell_max + sgn * m]

                u_m_even = m_reshape(u_m_even, (n_phi, n_r * n_data))
                u_m_odd = m_reshape(u_m_odd, (n_phi, n_r * n_data))

                w_m_even = ang_phi_wtd_m_even.T @ u_m_even
                w_m_odd = ang_phi_wtd_m_odd.T @ u_m_odd

                w_m_even = m_reshape(w_m_even, (n_even_ell, n_r, n_data))
                w_m_odd = m_reshape(w_m_odd, (n_odd_ell, n_r, n_data))
                end = np.size(w_even, 0)
                w_even[end - n_even_ell : end, :, self.ell_max + sgn * m, :] = w_m_even
                end = np.size(w_odd, 0)
                w_odd[end - n_odd_ell : end, :, self.ell_max + sgn * m, :] = w_m_odd

        w_even = np.transpose(w_even, (1, 2, 3, 0))
        w_odd = np.transpose(w_odd, (1, 2, 3, 0))

        # evaluate the radial parts
        v = np.zeros((n_data, self.count), dtype=x.dtype)
        for ell in range(0, self.ell_max + 1):
            k_max_ell = self.k_max[ell]
            radial_wtd = self._precomp["radial_wtd"][:, 0:k_max_ell, ell]

            if np.mod(ell, 2) == 0:
                v_ell = w_even[
                    :,
                    int(self.ell_max - ell) : int(self.ell_max + 1 + ell),
                    :,
                    int(ell / 2),
                ]
            else:
                v_ell = w_odd[
                    :,
                    int(self.ell_max - ell) : int(self.ell_max + 1 + ell),
                    :,
                    int((ell - 1) / 2),
                ]

            v_ell = m_reshape(v_ell, (n_r, (2 * ell + 1) * n_data))

            v_ell = radial_wtd.T @ v_ell

            v_ell = m_reshape(v_ell, (k_max_ell * (2 * ell + 1), n_data))

            # TODO: Fix this to avoid lookup each time.
            ind = self._indices["ells"] == ell
            v[:, ind] = v_ell.T

        # Roll dimensions, last dimension should be self.count,
        # Higher dimensions like x.
        v = v.reshape((*sz_roll, self.count))
        return v
