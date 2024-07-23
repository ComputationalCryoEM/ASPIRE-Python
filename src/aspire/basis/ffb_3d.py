import logging

import numpy as np

from aspire.basis import FBBasis3D
from aspire.basis.basis_utils import lgwt, norm_assoc_legendre, sph_bessel
from aspire.nufft import anufft, nufft
from aspire.numeric import xp
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
        self._calc_k_max()

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
        n_r = int(np.ceil(4 * self.rcut * self.kcut))
        n_theta = int(2 * self.nres)
        n_phi = int(2 * self.nres + 1)

        r, wt_r = lgwt(n_r, 0.0, self.kcut, dtype=self.dtype)
        z, wt_z = lgwt(n_phi, -1, 1, dtype=self.dtype)
        r = m_reshape(xp.asarray(r), (n_r, 1))
        rh = xp.asnumpy(r)
        wt_r = m_reshape(xp.asarray(wt_r), (n_r, 1))
        z = m_reshape(xp.asarray(z), (n_phi, 1))
        wt_z = m_reshape(xp.asarray(wt_z), (n_phi, 1))
        phi = xp.arccos(z)
        wt_phi = wt_z
        theta = 2 * xp.pi * xp.arange(n_theta, dtype=self.dtype).T / (2 * n_theta)
        theta = m_reshape(theta, (n_theta, 1))

        # evaluate basis function in the radial dimension
        radial_wtd = xp.zeros(
            shape=(n_r, np.max(self.k_max), self.ell_max + 1), dtype=self.dtype
        )
        for ell in range(0, self.ell_max + 1):
            k_max_ell = self.k_max[ell]
            rmat = rh * self.r0[ell][0:k_max_ell].T / self.kcut  # host
            radial_ell = xp.zeros_like(rmat)
            for ik in range(0, k_max_ell):
                radial_ell[:, ik] = xp.asarray(sph_bessel(ell, rmat[:, ik]))
            nrm = xp.abs(
                xp.asarray(sph_bessel(ell + 1, self.r0[ell][0:k_max_ell].T)) / 4
            )
            radial_ell = radial_ell / nrm
            radial_ell_wtd = r**2 * wt_r * radial_ell
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
            phi_wtd_m_even = xp.zeros((n_phi, n_even_ell), dtype=phi.dtype)
            phi_wtd_m_odd = xp.zeros((n_phi, n_odd_ell), dtype=phi.dtype)

            ind_even = 0
            ind_odd = 0
            for ell in range(m, self.ell_max + 1):
                phi_m_ell = xp.asarray(norm_assoc_legendre(ell, m, z))
                nrm_inv = np.sqrt(0.5 / np.pi)
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
        ang_theta = xp.zeros((n_theta, 2 * self.ell_max + 1), dtype=theta.dtype)

        ang_theta[:, 0 : self.ell_max] = np.sqrt(2) * xp.sin(
            theta @ m_reshape(xp.arange(self.ell_max, 0, -1), (1, self.ell_max))
        )
        ang_theta[:, self.ell_max] = xp.ones(n_theta, dtype=theta.dtype)
        ang_theta[:, self.ell_max + 1 : 2 * self.ell_max + 1] = np.sqrt(2) * xp.cos(
            theta @ m_reshape(xp.arange(1, self.ell_max + 1), (1, self.ell_max))
        )

        ang_theta_wtd = (2 * np.pi / n_theta) * ang_theta

        theta_grid, phi_grid, r_grid = xp.meshgrid(
            theta.flatten(), phi.flatten(), r.flatten(), sparse=False, indexing="ij"
        )
        fourier_x = m_flatten(r_grid * xp.cos(theta_grid) * xp.sin(phi_grid))
        fourier_y = m_flatten(r_grid * xp.sin(theta_grid) * xp.sin(phi_grid))
        fourier_z = m_flatten(r_grid * xp.cos(phi_grid))
        fourier_pts = (
            2
            * xp.pi
            * xp.vstack(
                (
                    fourier_z[None, ...],
                    fourier_y[None, ...],
                    fourier_x[None, ...],
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

    def _evaluate(self, v):
        """
        Evaluate coefficients in standard 3D coordinate basis from those in 3D FB basis

        :param v: A coefficient vector (or an array of coefficient vectors) in FB basis
            to be evaluated. The last dimension must equal `self.count`.
        :return x: The evaluation of the coefficient vector(s) `x` in standard 3D
            coordinate basis. This is an array whose last three dimensions equal
            `self.sz` and the remaining dimensions correspond to `v`.
        """
        v = xp.asarray(v)
        # roll dimensions of v
        sz_roll = v.shape[:-1]
        v = v.reshape((-1, self.count))

        # get information on polar grids from precomputed data
        n_theta = np.size(self._precomp["ang_theta_wtd"], 0)
        n_phi = np.size(self._precomp["ang_phi_wtd_even"][0], 0)
        n_r = np.size(self._precomp["radial_wtd"], 0)

        # number of 3D image samples
        n_data = v.shape[0]

        u_even = xp.zeros(
            (
                n_r,
                int(2 * self.ell_max + 1),
                n_data,
                int(np.floor(self.ell_max / 2) + 1),
            ),
            dtype=v.dtype,
        )
        u_odd = xp.zeros(
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

        u_even = u_even.transpose((3, 0, 1, 2))
        u_odd = u_odd.transpose((3, 0, 1, 2))
        w_even = xp.zeros((n_phi, n_r, n_data, 2 * self.ell_max + 1), dtype=v.dtype)
        w_odd = xp.zeros((n_phi, n_r, n_data, 2 * self.ell_max + 1), dtype=v.dtype)

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

        w_even = w_even.transpose((3, 0, 1, 2))
        w_odd = w_odd.transpose((3, 0, 1, 2))
        u_even = w_even
        u_odd = w_odd

        u_even = m_reshape(u_even, (2 * self.ell_max + 1, n_phi * n_r * n_data))
        u_odd = m_reshape(u_odd, (2 * self.ell_max + 1, n_phi * n_r * n_data))

        # evaluate the theta parts
        w_even = self._precomp["ang_theta_wtd"] @ u_even
        w_odd = self._precomp["ang_theta_wtd"] @ u_odd

        pf = w_even + 1j * w_odd
        pf = m_reshape(pf, (n_theta * n_phi * n_r, n_data))
        pf = xp.moveaxis(pf, 0, -1)

        # perform inverse non-uniformly FFT transformation back to 3D rectangular coordinates
        freqs = m_reshape(self._precomp["fourier_pts"], (3, n_r * n_theta * n_phi))
        x = anufft(pf, freqs, self.sz, real=True)

        # Roll, return the x with the last three dimensions as self.sz
        # Higher dimensions should be like v.
        x = x.reshape((*sz_roll, *self.sz))
        return xp.asnumpy(x)

    def _evaluate_t(self, x):
        """
        Evaluate coefficient in FB basis from those in standard 3D coordinate basis

        :param x: The coefficient array in the standard 3D coordinate basis
            to be evaluated. The last three dimensions must equal `self.sz`.
        :return: The evaluation of the coefficient array `x` in the FB basis.
            This is an array of vectors whose last dimension equals
            `self.count` and whose remaining dimensions correspond to higher
            dimensions of `x`.
        """
        x = xp.asarray(x)
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
        ang_theta_wtd_trans = self._precomp["ang_theta_wtd"].T
        u_even = ang_theta_wtd_trans @ pf.real
        u_odd = ang_theta_wtd_trans @ pf.imag

        u_even = m_reshape(u_even, (2 * self.ell_max + 1, n_phi, n_r, n_data))
        u_odd = m_reshape(u_odd, (2 * self.ell_max + 1, n_phi, n_r, n_data))

        u_even = u_even.transpose((1, 2, 3, 0))
        u_odd = u_odd.transpose((1, 2, 3, 0))

        w_even = xp.zeros(
            (int(np.floor(self.ell_max / 2) + 1), n_r, 2 * self.ell_max + 1, n_data),
            dtype=x.dtype,
        )
        w_odd = xp.zeros(
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

        w_even = w_even.transpose((1, 2, 3, 0))
        w_odd = w_odd.transpose((1, 2, 3, 0))

        # evaluate the radial parts
        v = xp.zeros((n_data, self.count), dtype=x.dtype)
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
        return xp.asnumpy(v)
