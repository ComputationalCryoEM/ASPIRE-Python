import logging

import numpy as np

from aspire.basis import Basis
from aspire.basis.basis_utils import (
    d_decay_approx_fun,
    k_operator,
    lgwt,
    t_radial_part_mat,
    t_x_derivative_mat,
    t_x_mat,
)
from aspire.basis.pswf_utils import BNMatrix
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class PSWFBasis2D(Basis):
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

    def __init__(self, size, gamma_trunc=1.0, beta=1.0, dtype=np.float32):
        """
        Initialize an object for 2D PSWF basis expansion using direct method

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

        TODO: need to re-implement to use the similar grid function as FB methods.
        """
        if self.nres % 2 == 0:
            x_1d_grid = range(-self.rcut, self.rcut)
        else:
            x_1d_grid = range(-self.rcut, self.rcut + 1)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_in_disk = r_2d_grid <= self.rcut
        x = y_2d_grid[points_in_disk]
        y = x_2d_grid[points_in_disk]
        self._r_disk = np.sqrt(np.square(x) + np.square(y)) / self.rcut
        self._theta_disk = np.angle(x + 1j * y)
        self._image_height = len(x_1d_grid)
        self._disk_mask = points_in_disk
        self._disk_mask_vec = points_in_disk.reshape(self._image_height ** 2)

    def _precomp(self):
        """
        Precompute PSWF functions on a polar Fourier 2D grid
        """
        self._generate_samples()

        self.non_neg_freq_inds = slice(0, len(self.ang_freqs))

        tmp = np.nonzero(self.ang_freqs == 0)[0]
        self.zero_freq_inds = slice(tmp[0], tmp[-1] + 1)

        tmp = np.nonzero(self.ang_freqs > 0)[0]
        self.pos_freq_inds = slice(tmp[0], tmp[-1] + 1)

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

        self.alpha_nn = np.array(alpha_all)
        self.max_ns = max_ns

        self.samples = self._evaluate_pswf2d_all(self._r_disk, self._theta_disk, max_ns)
        self.ang_freqs = np.repeat(np.arange(len(max_ns)), max_ns).astype("float")
        self.rad_freqs = np.concatenate([range(1, i + 1) for i in max_ns]).astype(
            "float"
        )
        self.samples = (self.beta / 2.0) * self.samples * self.alpha_nn
        self.samples_conj_transpose = self.samples.conj().transpose()

    def evaluate_t(self, images):
        """
        Evaluate coefficient vectors in PSWF basis using the direct method

        :param images: coefficient array in the standard 2D coordinate basis
            to be evaluated.
        :return : The evaluation of the coefficient array in the PSWF basis.
        """
        images = images.T  # RCOPT

        images_shape = images.shape

        images_shape = (images_shape + (1,)) if len(images_shape) == 2 else images_shape
        flattened_images = images.reshape(
            (images_shape[0] * images_shape[1], images_shape[2]), order="F"
        )

        flattened_images = flattened_images[self._disk_mask_vec, :]
        coefficients = self.samples_conj_transpose.dot(flattened_images)
        return coefficients.T

    def evaluate(self, coefficients):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in PSWF basis

        :param coeffcients: A coefficient vector (or an array of coefficient
            vectors) in PSWF basis to be evaluated.
        :return : The evaluation of the coefficient vector(s) in standard 2D
            coordinate basis.
        """
        coefficients = coefficients.T  # RCOPT

        # if we got only one vector
        if len(coefficients.shape) == 1:
            coefficients = coefficients[:, np.newaxis]

        angular_is_zero = np.absolute(self.ang_freqs) == 0
        flatten_images = self.samples[:, angular_is_zero].dot(
            coefficients[angular_is_zero]
        ) + 2.0 * np.real(
            self.samples[:, ~angular_is_zero].dot(coefficients[~angular_is_zero])
        )

        n_images = int(flatten_images.shape[1])
        images = np.zeros((self._image_height, self._image_height, n_images)).astype(
            complex_type(self.dtype)
        )
        images[self._disk_mask, :] = flatten_images
        images = np.transpose(images, axes=(1, 0, 2))
        return np.real(images).T  # RCOPT

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
        :return: (len(r), sum(max_ns)) ndarray
            Indices are corresponding to the list (N, n)
            (0, 0),..., (0, max_ns[0]), (1, 0),..., (1, max_ns[1]),... , (len(max_ns)-1, 0),
            (len(max_ns)-1, max_ns[-1])
        """
        max_ns_ints = [int(max_n) for max_n in max_ns]
        out_mat = []
        for i, max_n in enumerate(max_ns_ints):
            if max_n < 1:
                continue

            d_vec = self.d_vec_all[i]

            phase_part = np.exp(1j * i * theta) / np.sqrt(2 * np.pi)
            range_array = np.arange(len(d_vec))
            r_radial_part_mat = t_radial_part_mat(r, i, range_array, len(d_vec)).dot(
                d_vec[:, :max_n]
            )

            pswf_n_n_mat = phase_part * r_radial_part_mat.T

            out_mat.extend(pswf_n_n_mat)
        out_mat = np.array(out_mat, dtype=complex_type(self.dtype)).T
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

        range_array = np.array(range(approx_length))
        return d_vec, approx_length, range_array
