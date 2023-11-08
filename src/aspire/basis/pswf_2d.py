import logging

import numpy as np

from aspire.basis import Coef, ComplexCoef, SteerableBasis2D
from aspire.basis.basis_utils import (
    d_decay_approx_fun,
    k_operator,
    lgwt,
    t_radial_part_mat,
    t_x_derivative_mat,
    t_x_mat,
)
from aspire.basis.pswf_utils import BNMatrix
from aspire.operators import BlkDiagMatrix
from aspire.utils import complex_type, grid_2d

logger = logging.getLogger(__name__)


class PSWFBasis2D(SteerableBasis2D):
    """
    Define a derived class for direct Prolate Spheroidal Wave Function (PSWF) expanding 2D images

    The numerical evaluation for 2D PSWFs at arbitrary points in the unit disk is based on the
    direct method described in the papers as below:
        1) Boris Landa and Yoel Shkolnisky, "Steerable principal components
        for space-frequency localized images", SIAM J. Imag. Sci. 10, 508-534 (2017).
        2) Boris Landa and Yoel Shkolnisky, "Approximation scheme for essentially
        bandlimited and space-concentrated functions on a disk", Appl. Comput.
        Harmon. Anal. 43, 381-403 (2017).
        3) Yoel Shkolnisky, "Prolate spheroidal wave functions on a disc-Integration
        and approximation of two-dimensional bandlimited functions", Appl.
        Comput. Harmon. Anal. 22, 235-256 (2007).
    """

    matrix_type = BlkDiagMatrix

    def __init__(self, size, gamma_trunc=1.0, beta=1.0, dtype=np.float32):
        """
        Initialize an object for 2D PSWF basis expansion using direct method

        :param size: The size of the vectors for which to define the basis
            and the image resolution. May be a 2-tuple or an integer, in which case
            a square basis is assumed. Currently only square images are supported.
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
        if isinstance(size, int):
            size = (size, size)
        self.rcut = size[0] // 2
        self.gmcut = gamma_trunc
        self.beta = beta
        super().__init__(size, dtype=dtype)

    def _build(self):
        """
        Build internal data structures for the direct 2D PSWF method
        """
        logger.info("Expanding 2D images using direct PSWF method.")

        # initial the whole set of PSWF basis functions based on the bandlimit and eps error.
        self.bandlimit = self.beta * np.pi * self.rcut
        self.d_vec_all, self.alpha_all, self.lengths = self._init_pswf_func2d(
            self.bandlimit, eps=np.spacing(1)
        )

        # generate_the 2D grid and corresponding indices inside the disc.
        self._generate_grid()

        # precompute the basis functions in 2D grids
        self._precomp()

    def _generate_grid(self):
        """
        Generate the 2D sampling grid
        """
        grid = grid_2d(self.nres, normalized=False, indexing="yx")
        self._disk_mask = grid["r"] <= self.rcut
        self._r_disk = grid["r"][self._disk_mask] / self.rcut
        self._theta_disk = grid["phi"][self._disk_mask]

    def _precomp(self):
        """
        Precompute PSWF functions on a polar Fourier 2D grid
        """
        self._generate_samples()

    def _generate_samples(self):
        """
        Generate sample points for PSWF functions
        """
        max_ns = []
        a = np.square(float(self.beta * self.rcut) / 2)
        m = 0
        alpha_all = []

        while True:
            alpha = self.alpha_all[m]

            lambda_var = a * np.square(np.absolute(alpha))
            gamma = np.sqrt(np.absolute(lambda_var / (1 - lambda_var)))

            n_end = np.where(gamma <= self.gmcut)[0]

            if len(n_end) != 0:
                n_end = n_end[0]
                if n_end == 0:
                    break
                max_ns.extend([n_end])
                alpha_all.extend(alpha[:n_end])
                m += 1

        self.alpha_nn = np.array(alpha_all, dtype=complex_type(self.dtype)).reshape(
            -1, 1
        )
        self.max_ns = max_ns

        self.samples = self._evaluate_pswf2d_all(self._r_disk, self._theta_disk, max_ns)
        self.complex_angular_indices = np.repeat(
            np.arange(len(max_ns), dtype=int), max_ns
        )
        self.complex_radial_indices = np.concatenate(
            [np.arange(1, i + 1, dtype=int) for i in max_ns]
        )

        # Added to support subclassing SteerableBasis
        self.complex_signs_indices = np.sign(self.complex_angular_indices)

        self.samples = (self.beta / 2.0) * self.samples * self.alpha_nn
        self.samples_conj_transpose = self.samples.conj().transpose()
        # the column dimension of samples_conj_transpose is the number of basis coefficients
        self.complex_count = self.samples_conj_transpose.shape[1]

        # Add required real indices attributes and maps
        # TODO, this block of code can probably be consolidated with
        # FB basis.  For now, just get everything working together.
        nz = np.sum(self.complex_signs_indices == 0)
        nnz = self.complex_count - nz

        self.real_count = nz + 2 * nnz
        self.count = self.real_count

        self.radial_indices = np.empty(self.real_count, dtype=int)
        self.angular_indices = np.empty(self.real_count, dtype=int)
        self.signs_indices = np.empty(self.real_count, dtype=int)

        self._pos = np.zeros(self.complex_count, dtype=int)
        self._neg = np.zeros(self.complex_count, dtype=int)

        i = 0
        ci = 0
        self.k_max = []
        self.ell_max = np.max(self.complex_angular_indices)
        for ell in range(self.ell_max + 1):
            sgns = (1,) if ell == 0 else (1, -1)
            k_max = np.sum(self.complex_angular_indices == ell)
            self.k_max.append(k_max)
            ks = np.arange(0, k_max)

            for sgn in sgns:
                rng = np.arange(i, i + len(ks))
                self.angular_indices[rng] = ell
                self.radial_indices[rng] = ks
                self.signs_indices[rng] = sgn

                if sgn == 1:
                    self._pos[ci + ks] = rng
                elif sgn == -1:
                    self._neg[ci + ks] = rng

                i += len(ks)

            ci += len(ks)

    def _evaluate_t(self, images):
        """
        Evaluate coefficient vectors in PSWF basis using the direct method

        :param images: coefficient array in the standard 2D coordinate basis
            to be evaluated.
        :return: The evaluation of the coefficient array in the PSWF basis.
        """
        flattened_images = images[:, self._disk_mask]
        complex_coef = ComplexCoef(self, flattened_images @ self.samples_conj_transpose)
        return complex_coef.to_real().asnumpy()

    def _evaluate(self, coefficients):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in PSWF basis

        :param coefficients: A coefficient vector (or an array of coefficient
            vectors) in PSWF basis to be evaluated. (n_image, count)
        :return : Image in standard 2D coordinate basis.

        """

        # Convert real coefficient to complex.
        coefficients = Coef(self, coefficients).to_complex()

        # Handle a single coefficient vector or stack of vectors.
        coefficients = np.atleast_2d(coefficients)
        n_images = coefficients.shape[0]

        angular_is_zero = np.absolute(self.complex_angular_indices) == 0

        flatten_images = coefficients[:, angular_is_zero] @ self.samples[
            angular_is_zero
        ] + 2.0 * np.real(
            coefficients[:, ~angular_is_zero] @ self.samples[~angular_is_zero]
        )

        images = np.zeros((n_images, self.nres, self.nres), dtype=self.dtype)
        images[:, self._disk_mask] = np.real(flatten_images)

        return images

    def _init_pswf_func2d(self, c, eps):
        """
        Initialize the whole set of PSWF functions with the input bandlimit and error

        :param c: bandlimit (>0) can be estimated by beta * pi * rcut
        :param eps: error tolerance
        :return:
            alpha_all (list of arrays):
            alpha = alpha_all[i] contains all the eigenvalues for N=i such that
                lambda > eps, where lambda is the normalized  alpha values (i.e.
                lambda is between 0 and 1), given by lambda=sqrt(c*np.absolute(alpha)/(2*pi)).
            d_vec_all (list of 2D lists): the corresponding eigenvectors for alpha_all.
            n_order_length_vec (list of ints): n_order_length_vec[i] = len(alpha_all[i])
        """
        d_vec_all = []
        alpha_all = []
        n_order_length_vec = []

        m = 0
        n = int(np.ceil(2 * c / np.pi))
        r, w = lgwt(n, 0, 1, dtype=self.dtype)

        cons = c / 2 / np.pi
        while True:
            alpha, d_vec, a = self.pswf_func2d(m, n, c, eps, r, w)

            lambda_var = np.sqrt(cons * np.absolute(alpha))

            n_end = np.where(lambda_var <= eps)[0]

            if len(n_end) != 0:
                n_end = n_end[0]
                if n_end == 0:
                    break
                n_order_length_vec.append(n_end)
                alpha_all.append(alpha[:n_end])
                d_vec_all.append(d_vec[:, :n_end])
                m += 1
                n = n_end + 1
            else:
                n *= 2
                r, w = lgwt(n, 0, 1)

        return d_vec_all, alpha_all, n_order_length_vec

    def _evaluate_pswf2d_all(self, r, theta, max_ns):
        """
        Evaluate the numerical values of PSWF functions for all N's, up to given n for each N

        :param r: Radial part to evaluate
        :param theta: Phase part to evaluate
        :param max_ns: List of ints max_ns[i] is max n to to use for N=i, not included.
            If max_ns[i]<1 N=i won't be used
        :return: (sum(max_ns), len(r)) ndarray
            Indices are corresponding to the list (N, n)
            (0, 0),..., (max_ns[0], 0), (0, 1),..., (max_ns[1], 1),... , (0, len(max_ns)-1),
            (max_ns[-1], len(max_ns)-1)
        """
        max_ns_ints = [int(max_n) for max_n in max_ns]
        out_mat = []
        for i, max_n in enumerate(max_ns_ints):
            if max_n < 1:
                continue

            d_vec = self.d_vec_all[i]

            phase_part = np.exp(1j * i * theta) / np.sqrt(2 * np.pi)
            range_array = np.arange(len(d_vec), dtype=self.dtype)
            r_radial_part_mat = t_radial_part_mat(r, i, range_array, len(d_vec)).dot(
                d_vec[:, :max_n]
            )

            pswf_n_n_mat = phase_part * r_radial_part_mat.T

            out_mat.extend(pswf_n_n_mat)
        out_mat = np.array(out_mat, dtype=complex_type(self.dtype))
        return out_mat

    def pswf_func2d(self, big_n, n, bandlimit, phi_approximate_error, r, w):
        """
        Calculate the eigenvalues and eigenvectors of PSWF basis functions for all N's and n's

        :param big_n: The integer N in PSWF basis.
        :param n: The integer n in PSWF basis.
        :param bandlimit: The band limit estimated by beta * pi * rcut.
        :param phi_approximate_error: The input approximate error for phi.
        :param r: The Legendre–Gauss quadrature nodes.
        :param w: The Legendre–Gauss quadrature weights.
        :return:
            alpha_n (ndarray): the eigen-values for N.
            d_vec (ndarray): the corresponding eigen-vectors for alpha_n.
            approx_length (int): the number of eigenvalues,len(alpha_n).
        """

        d_vec, approx_length, range_array = self._pswf_2d_minor_computations(
            big_n, n, bandlimit, phi_approximate_error
        )

        t1 = 1 - 2 * np.square(r)
        t2 = np.sqrt(2 * (2 * range_array + big_n + 1))

        phi = t_x_mat(r, big_n, range_array, approx_length).dot(d_vec[:, : (n + 1)])
        phi_derivatives = t_x_derivative_mat(
            t1, t2, r, big_n, range_array, approx_length
        ).dot(d_vec[:, : (n + 1)])

        max_phi_idx = np.argmax(np.absolute(phi[:, 0]))
        max_phi_val = phi[max_phi_idx, 0]
        x_for_calc = r[max_phi_idx]

        right_hand_side_integral = np.einsum(
            "j, j, j ->", w, k_operator(big_n, bandlimit * x_for_calc * r), phi[:, 0]
        )
        lambda_n_1 = right_hand_side_integral / max_phi_val

        temp_calc = r * w
        upper_integral_values = np.einsum(
            "j, ji, ji -> i", temp_calc, phi_derivatives[:, :-1], phi[:, 1:]
        )
        lower_integral_values = np.einsum(
            "j, ji, ji -> i", temp_calc, phi[:, :-1], phi_derivatives[:, 1:]
        )

        lambda_n = np.append(
            np.reshape(lambda_n_1, (1, 1)),
            (
                lambda_n_1
                * np.cumprod(upper_integral_values / lower_integral_values).reshape(
                    (n, 1)
                )
            ),
        )
        alpha_n = lambda_n * 2 * np.pi * (np.power(1j, big_n) / np.sqrt(bandlimit))

        return alpha_n, d_vec, approx_length

    def _pswf_2d_minor_computations(self, big_n, n, bandlimit, phi_approximate_error):
        """
        Approximate the number of n's for fixed N (big_n) and compute the d_vec

        :param big_n: int
        :param n: int
                used for the computation of approx_length
        :param bandlimit: float > 0
        :param phi_approximate_error: float > 0
        :return: d_vec: (approx_length, approx_length) ndarray
                d_vec[:, i] = d^{N, i} defined from eq (18) of paper 3).
            approx_length: int
            range_array: (approx_length,) ndarray
                range_array[i] = i
        """

        first_idx_for_decrease = np.ceil(
            (
                np.sqrt(np.square(2 * n + big_n + 1) + np.square(bandlimit) / 2)
                - (2 * n + big_n + 1)
            )
            / 2
        )

        d_approx = d_decay_approx_fun(big_n, n, bandlimit, first_idx_for_decrease)
        d_decay_index_counter = first_idx_for_decrease

        while d_approx > phi_approximate_error:
            d_decay_index_counter = d_decay_index_counter + 1
            d_approx = d_approx * d_decay_approx_fun(
                big_n, n, bandlimit, d_decay_index_counter
            )
        approx_length = int(n + 1 + d_decay_index_counter)

        d_vec, _ = BNMatrix(big_n, bandlimit, approx_length).get_eig_vectors()

        range_array = np.arange(approx_length, dtype=self.dtype)
        return d_vec, approx_length, range_array

    def filter_to_basis_mat(self, *args, **kwargs):
        """
        See `SteerableBasis2D.filter_to_basis_mat`.
        """
        return super().filter_to_basis_mat(*args, **kwargs)
