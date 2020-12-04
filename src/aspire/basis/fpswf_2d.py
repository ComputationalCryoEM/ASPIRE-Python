import logging

import numpy as np
from numpy import pi
from numpy.linalg import lstsq
from scipy.optimize import least_squares
from scipy.special import jn

from aspire.basis.basis_utils import lgwt, t_x_mat, t_x_mat_dot
from aspire.basis.pswf_2d import PSWFBasis2D
from aspire.nufft import nufft
from aspire.numeric import fft, xp
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class FPSWFBasis2D(PSWFBasis2D):
    """
    Define a derived class for fast Prolate Spheroidal Wave Function (PSWF) expanding 2D images

    The numerical evaluation for 2D PSWFs at arbitrary points in the unit disk is based on the fast method
    described in the papers as below:
        1) Boris Landa and Yoel Shkolnisky, "Steerable principal components for space-frequency localized images",
        SIAM J. Imag. Sci. 10, 508-534 (2017).
        2) Boris Landa and Yoel Shkolnisky, "Approximation scheme for essentially bandlimited and space-concentrated
        functions on a disk", Appl. Comput. Harmon. Anal. 43, 381-403 (2017).
        3) Yoel Shkolnisky, "Prolate spheroidal wave functions on a disc-Integration and approximation of
        two-dimensional bandlimited functions", Appl. Comput. Harmon. Anal. 22, 235-256 (2007).
    """

    def __init__(self, size, gamma_truncation=1.0, beta=1.0, dtype=np.float32):
        """
        Initialize an object for 2D prolate spheroidal wave function (PSWF) basis expansion using fast method.

        :param size: The size of the vectors for which to define the basis
            and the image resultion. Currently only square images are supported.
        :param gamma_trunc: Truncation parameter of PSWFs, between 0 and 1e6,
            which controls the length of the expansion and the approximation error.
            Smaller values (close to zero) guarantee smaller errors, yet longer
            expansions, and vice-versa. Note: Due to numerical considerations,
            do not exceed 1e6.
        :param beta: Bandlimit ratio relative to the Nyquist rate, between 0 and 1.
            In general, the bandlimit is c = beta*pi*(size[0]//2), therefore for
            the default value beta = 1 there is no oversampling assumed. This
            parameter controls the bandlimit of the PSWFs.
        :param dtype: Internal ndarray datatype.
        """
        super().__init__(size, gamma_truncation, beta, dtype=dtype)

    def _build(self):
        """
        Build internal data structures for the fast 2D PSWF method
        """
        logger.info("Expanding 2D images using fast PSWF method.")

        # initial the whole set of PSWF basis functions based on the bandlimit and eps error.
        self.bandlimit = self.beta * np.pi * self.rcut
        self.d_vec_all, self.alpha_all, self.lengths = self._init_pswf_func2d(
            self.bandlimit, eps=np.spacing(1)
        )

        # generate_the 2D grid and corresponding indices inside the disc.
        self._generate_grid()

        # precompute the basis functions in 2D grids
        self._precomp()

    def _precomp(self):
        """
        Precomute the PSWF functions on a polar Fourier 2D grid for the fast method
        """
        self._generate_samples()

        eps = np.spacing(1)
        a, b, c, d, e, f = self._generate_pswf_quad(
            4 * self.rcut, 2 * self.bandlimit, eps, eps, eps
        )
        self.pswf_radial_quad = self._evaluate_pswf2d_all(
            d, np.zeros(len(d)), self.max_ns
        )
        self.quad_rule_pts_x = a
        self.quad_rule_pts_y = b
        self.quad_rule_wts = c
        self.radial_quad_pts = d
        self.quad_rule_radial_wts = e
        self.num_angular_pts = f

        # pre computing variables for forward
        us_fft_pts = np.column_stack((self.quad_rule_pts_y, self.quad_rule_pts_x))
        us_fft_pts = self.bandlimit / (self.rcut * np.pi * 2) * us_fft_pts  # for pynfft
        (
            blk_r,
            num_angular_pts,
            r_quad_indices,
            numel_for_n,
            indices_for_n,
            n_max,
        ) = self._pswf_integration_sub_routine()

        self.us_fft_pts = us_fft_pts
        self.blk_r = blk_r
        self.num_angular_pts = num_angular_pts
        self.r_quad_indices = r_quad_indices
        self.numel_for_n = numel_for_n
        self.indices_for_n = indices_for_n
        self.n_max = n_max
        self.size_x = len(self._disk_mask)

    def evaluate_t(self, images):
        """
        Evaluate coefficient vectors in PSWF basis using the fast method

        :param images: coefficient array in the standard 2D coordinate basis
            to be evaluated.
        :return : The evaluation of the coefficient array in the PSWF basis.
        """

        images = np.moveaxis(images, 0, -1)  # RCOPT

        # start and finish are for the threads option in the future
        images_shape = images.shape
        start = 0

        if len(images_shape) == 3:
            # if we got several images
            finish = images_shape[2]
        else:
            # else we got only one image
            images_shape = images_shape + (1,)
            images = images[..., np.newaxis]
            finish = 1
        images_disk = np.zeros(images.shape, dtype=images.dtype, order="F")
        images_disk[self._disk_mask, :] = images[self._disk_mask, :]
        nfft_res = self._compute_nfft_potts(images_disk, start, finish)
        coefficients = self._pswf_integration(nfft_res)

        return coefficients.T  # RCOPT

    def evaluate(self, coefficients):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in PSWF basis

        :param coefficients: A coefficient vector (or an array of coefficient vectors)
            in PSWF basis to be evaluated.
        :return : The evaluation of the coefficient vector(s) in standard 2D
            coordinate basis.
        """

        coefficients = coefficients.T  # RCOPT

        # if we got only one vector
        if len(coefficients.shape) == 1:
            coefficients = coefficients.reshape((len(coefficients), 1))

        angular_is_zero = np.absolute(self.ang_freqs) == 0
        flatten_images = self.samples[:, angular_is_zero].dot(
            coefficients[angular_is_zero]
        ) + (
            2.0
            * np.real(
                self.samples[:, ~angular_is_zero].dot(coefficients[~angular_is_zero])
            )
        )

        n_images = int(flatten_images.shape[1])
        images = np.zeros((self._image_height, self._image_height, n_images)).astype(
            complex_type(self.dtype)
        )
        images[self._disk_mask, :] = flatten_images
        # TODO: no need to switch x and y any more, need to make consistent with direct method
        return np.real(images).T  # RCOPT

    def _generate_pswf_quad(
        self, n, bandlimit, phi_approximate_error, lambda_max, epsilon
    ):
        """
        Generate Gaussian quadrature points and weights for 2D PSWF functions
        """
        radial_quad_points, radial_quad_weights = self._generate_pswf_radial_quad(
            n, bandlimit, phi_approximate_error, lambda_max
        )

        num_angular_points = (
            np.ceil(np.e * radial_quad_points * bandlimit / 2 - np.log(epsilon)).astype(
                "int"
            )
            + 1
        )

        for i in range(len(radial_quad_points)):
            ang_error_vec = np.absolute(
                jn(
                    range(1, 2 * num_angular_points[i] + 1),
                    bandlimit * radial_quad_points[i],
                )
            )

            num_angular_points[i] = self._sum_minus_cumsum_smaller_eps(
                ang_error_vec, epsilon
            )
            if num_angular_points[i] % 2 == 1:
                num_angular_points[i] += 1

        temp = 2 * np.pi / num_angular_points

        t = 2

        quad_rule_radial_weights = temp * radial_quad_points * radial_quad_weights
        quad_rule_weights = np.repeat(
            quad_rule_radial_weights, repeats=num_angular_points
        )
        quad_rule_pts_r = np.repeat(
            radial_quad_points, repeats=(num_angular_points / t).astype("int")
        )
        quad_rule_pts_theta = np.concatenate(
            [
                temp[i] * np.arange(num_angular_points[i] / t)
                for i in range(len(radial_quad_points))
            ]
        )

        pts_x = quad_rule_pts_r * np.cos(quad_rule_pts_theta)
        pts_y = quad_rule_pts_r * np.sin(quad_rule_pts_theta)

        return (
            pts_x,
            pts_y,
            quad_rule_weights,
            radial_quad_points,
            quad_rule_radial_weights,
            num_angular_points,
        )

    def _generate_pswf_radial_quad(
        self, n, bandlimit, phi_approximate_error, lambda_max
    ):
        """
        Generate Gaussian quadrature points and weights for the radical parts of 2D PSWFs
        """
        x, w = lgwt(20 * n, 0, 1, dtype=self.dtype)

        big_n = 0

        x_as_mat = x.reshape((len(x), 1))

        alpha_n, d_vec, approx_length = self.pswf_func2d(
            big_n, n, bandlimit, phi_approximate_error, x, w
        )

        cut_indices = np.where(
            bandlimit / 2 / np.pi * np.absolute(alpha_n) < lambda_max
        )[0]

        if len(cut_indices) == 0:
            k = len(alpha_n)
        else:
            k = cut_indices[0]

        if k % 2 == 0:
            k = k + 1

        range_array = np.arange(approx_length).reshape((1, approx_length))

        idx_for_quad_nodes = int((k + 1) / 2)
        num_quad_pts = idx_for_quad_nodes - 1

        phi_zeros = self._find_initial_nodes(
            x, n, bandlimit / 2, phi_approximate_error, idx_for_quad_nodes
        )

        def phi_for_quad_weights(t):
            return np.dot(
                t_x_mat_dot(t, big_n, range_array, approx_length), d_vec[:, : k - 1]
            )

        b = np.dot(w * np.sqrt(x), phi_for_quad_weights(x_as_mat))

        a = phi_for_quad_weights(
            phi_zeros.reshape((len(phi_zeros), 1))
        ).transpose() * np.sqrt(phi_zeros)
        init_quad_weights = lstsq(a, b, rcond=None)
        init_quad_weights = init_quad_weights[0]
        tolerance = np.spacing(1)

        def obj_func(quad_rule):
            q = quad_rule.reshape((len(quad_rule), 1))
            temp = np.dot(
                (
                    phi_for_quad_weights(q[:num_quad_pts]) * np.sqrt(q[:num_quad_pts])
                ).transpose(),
                q[num_quad_pts:],
            )
            temp = temp.reshape(temp.shape[0])
            return temp - b

        arr_to_send = np.concatenate((phi_zeros, init_quad_weights))
        quad_rule_final = least_squares(
            obj_func, arr_to_send, xtol=tolerance, ftol=tolerance, max_nfev=1000
        )
        quad_rule_final = quad_rule_final.x
        quad_rule_pts = quad_rule_final[:num_quad_pts]
        quad_rule_weights = quad_rule_final[num_quad_pts:]
        return quad_rule_pts, quad_rule_weights

    def _find_initial_nodes(
        self, x, n, bandlimit, phi_approximate_error, idx_for_quad_nodes
    ):
        """
        Find initial quadrature nodes
        """
        big_n = 0

        d_vec, approx_length, range_array = self._pswf_2d_minor_computations(
            big_n, n, bandlimit, phi_approximate_error
        )

        def phi_for_quad_nodes(t):
            return np.dot(
                t_x_mat(t, big_n, range_array, approx_length),
                d_vec[:, idx_for_quad_nodes - 1],
            )

        fun_vec = phi_for_quad_nodes(x)
        sign_flipping_vec = np.where(np.sign(fun_vec[:-1]) != np.sign(fun_vec[1:]))[0]
        phi_zeros = np.zeros(idx_for_quad_nodes - 1)

        tmp = phi_for_quad_nodes(x)
        for i, j in enumerate(sign_flipping_vec[: idx_for_quad_nodes - 1]):
            new_zero = x[j] - tmp[j] * (x[j + 1] - x[j]) / (tmp[j + 1] - tmp[j])
            phi_zeros[i] = new_zero

        phi_zeros = np.array(phi_zeros)
        return phi_zeros

    def _sum_minus_cumsum_smaller_eps(self, x, eps):
        y = np.cumsum(np.flipud(x))
        return len(y) - np.where(y > eps)[0][0] + 1

    def _pswf_integration_sub_routine(self):
        t = 2
        num_angular_pts = (self.num_angular_pts / t).astype("int")

        r_quad_indices = [0]
        r_quad_indices.extend(num_angular_pts)
        r_quad_indices = np.cumsum(r_quad_indices, dtype="int")

        n_max = int(max(self.ang_freqs) + 1)

        numel_for_n = np.zeros(n_max, dtype="int")
        for i in range(n_max):
            numel_for_n[i] = np.count_nonzero(self.ang_freqs == i)

        indices_for_n = [0]
        indices_for_n.extend(numel_for_n)
        indices_for_n = np.cumsum(indices_for_n, dtype="int")

        blk_r = [0] * n_max
        temp_const = self.bandlimit / (2 * np.pi * self.rcut)
        for i in range(n_max):
            blk_r[i] = (
                temp_const
                * self.pswf_radial_quad[
                    :, indices_for_n[i] + np.arange(numel_for_n[i])
                ].T
            )

        return blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max

    def _compute_nfft_potts(self, images, start, finish):
        """
        Perform NuFFT transform for images in rectangular coordinates
        """
        x = self.us_fft_pts
        num_images = finish - start

        m = x.shape[0]

        images_nufft = np.zeros((m, num_images), dtype=complex_type(self.dtype))
        for i in range(start, finish):
            images_nufft[:, i - start] = nufft(images[..., i], 2 * pi * x.T)

        return images_nufft

    def _pswf_integration(self, images_nufft):
        """
        Perform integration part for rotational invariant property.
        """
        num_images = images_nufft.shape[1]
        n_max_float = float(self.n_max) / 2
        r_n_eval_mat = np.zeros(
            (len(self.radial_quad_pts), self.n_max, num_images),
            dtype=complex_type(self.dtype),
        )

        for i in range(len(self.radial_quad_pts)):
            curr_r_mat = images_nufft[
                self.r_quad_indices[i] : self.r_quad_indices[i]
                + self.num_angular_pts[i],
                :,
            ]
            curr_r_mat = np.concatenate((curr_r_mat, np.conj(curr_r_mat)))
            fft_plan = xp.asnumpy(fft.fft(xp.asarray(curr_r_mat), axis=0))
            angular_eval = fft_plan * self.quad_rule_radial_wts[i]

            r_n_eval_mat[i, :, :] = np.tile(
                angular_eval,
                (int(max(1, np.ceil(n_max_float / self.num_angular_pts[i]))), 1),
            )[: self.n_max, :]

        r_n_eval_mat = r_n_eval_mat.reshape(
            (len(self.radial_quad_pts) * self.n_max, num_images), order="F"
        )
        coeff_vec_quad = np.zeros(
            (len(self.ang_freqs), num_images), dtype=complex_type(self.dtype)
        )
        m = len(self.pswf_radial_quad)
        for i in range(self.n_max):
            coeff_vec_quad[
                self.indices_for_n[i] + np.arange(self.numel_for_n[i]), :
            ] = np.dot(self.blk_r[i], r_n_eval_mat[i * m : (i + 1) * m, :])

        return coeff_vec_quad
