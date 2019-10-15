import logging
import numpy as np

from aspire.basis import Basis
from aspire.basis.bn_matrix import BNMatrix
from aspire.basis.basis_func import d_decay_approx_fun, t_radial_part_mat, t_x_mat, t_x_derivative_mat
from aspire.basis.basis_func import k_operator, leggauss_0_1

logger = logging.getLogger(__name__)


class PSWFBasis2D(Basis):
    """
    Define a derived class using the Prolate Spheroidal Wave Function (PSWF) basis for mapping 2D images.
    The numerical evaluation for 2D PSWFs at arbitrary points in the unit disk is based on the direct method
    described in the papers as below:
        1) Boris Landa and Yoel Shkolnisky, "Steerable principal components for space-frequency localized images",
        SIAM J. Imag. Sci. 10, 508-534 (2017).
        2) Boris Landa and Yoel Shkolnisky, "Approximation scheme for essentially bandlimited and space-concentrated
        functions on a disk", Appl. Comput. Harmon. Anal. 43, 381-403 (2017).
        3) Yoel Shkolnisky, "Prolate spheroidal wave functions on a disc-Integration and approximation of
        two-dimensional bandlimited functions", Appl. Comput. Harmon. Anal. 22, 235-256 (2007).
    """
    def __init__(self, size, gamma_truncation, beta):
        # find max alpha for each N
        self.rcut = size[0] // 2
        self.gmcut = gamma_truncation
        self.beta = beta
        super().__init__(size)

    def _build(self):

        logger.info('Expanding 2D images using direct PSWF method.')

        # initial the whole set of PSWF basis functions based on the bandlimit and eps error.
        self.bandlimit = self.beta * np.pi * self.rcut
        self.d_vec_all, self.alpha_all, self.lengths = self.init_pswf_func2d(self.bandlimit, eps=np.spacing(1))

        # generate_the 2D grid and corresponding indices inside the disc.
        self.generate_grid()

        # precompute the basis functions in 2D grids
        self.precomp()

        # calculate total number of basis functions
        # self.basis_count = self.k_max[0] + sum(2 * self.k_max[1:])

        # obtain a 2D grid to represent basis functions
        # self.basis_coords = unique_coords_nd(self.N, self.d)

        # generate 1D indices for basis functions
        # self._indices = self.indices()

    def generate_grid(self):
        if self.nres % 2 == 0:
            x_1d_grid = range(-self.rcut, self.rcut)
        else:
            x_1d_grid = range(-self.rcut, self.rcut + 1)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_inside_the_circle = r_2d_grid <= self.rcut
        x = y_2d_grid[points_inside_the_circle]
        y = x_2d_grid[points_inside_the_circle]
        self.r_2d_grid_on_the_circle = np.sqrt(np.square(x) + np.square(y)) / self.rcut
        self.theta_2d_grid_on_the_circle = np.angle(x + 1j * y)
        self.image_height = len(x_1d_grid)
        self.points_inside_the_circle = points_inside_the_circle
        self.points_inside_the_circle_vec = points_inside_the_circle.reshape(self.image_height ** 2)

    def precomp(self):

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

        self.samples = self.evaluate_pswf2d_all(self.r_2d_grid_on_the_circle, self.theta_2d_grid_on_the_circle, max_ns)

        self.angular_frequency = np.repeat(np.arange(len(max_ns)), max_ns).astype('float')
        self.radian_frequency = np.concatenate([range(1, l + 1) for l in max_ns]).astype('float')
        self.alpha_nn = np.array(alpha_all)

        self.samples = (self.beta / 2.0) * self.samples * self.alpha_nn
        self.samples_conj_transpose = self.samples.conj().transpose()

        self.non_neg_freq_inds = slice(0, len(self.angular_frequency))

        tmp = np.nonzero(self.angular_frequency == 0)[0]
        self.zero_freq_inds = slice(tmp[0], tmp[-1] + 1)

        tmp = np.nonzero(self.angular_frequency > 0)[0]
        self.pos_freq_inds = slice(tmp[0], tmp[-1] + 1)

    def evaluate_t(self, images):
        images_shape = images.shape

        # if we got several images
        if len(images_shape) == 3:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], images_shape[2]), order='F')

        # else we got only one image
        else:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], 1), order='F')

        flattened_images = flattened_images[self.points_inside_the_circle_vec, :]
        coefficients = self.samples_conj_transpose.dot(flattened_images)
        return coefficients

    def evaluate(self, coefficients):
        # if we got only one vector
        if len(coefficients.shape) == 1:
            coefficients = coefficients.reshape((len(coefficients), 1))

        angular_is_zero = np.absolute(self.angular_frequency) == 0
        flatten_images = self.samples[:, angular_is_zero].dot(coefficients[angular_is_zero]) + \
                         2.0 * np.real(self.samples[:, ~angular_is_zero].dot(coefficients[~angular_is_zero]))

        n_images = int(flatten_images.shape[1])
        images = np.zeros((self.image_height, self.image_height, n_images)).astype('complex')
        images[self.get_points_inside_the_circle(), :] = flatten_images
        images = np.transpose(images, axes=(1, 0, 2))
        return np.real(images)

    def get_points_inside_the_circle(self):
        return self.points_inside_the_circle

    def mask_points_inside_the_circle(self, images):
        return self.__mask_points_inside_the_circle_cpu(images)

    def __mask_points_inside_the_circle_cpu(self, images):
        return images * self.points_inside_the_circle

    def get_samples_as_images(self):
        print("get_samples_as_images is not supported.")
        return -1

    def get_angular_frequency(self):
        return self.angular_frequency

    def get_num_prolates(self):
        # TODO: fix once Itay gives the samples in the correct way
        print("get_samples_as_images is not supported.")
        return -1

    def get_non_neg_freq_inds(self):
        return self.non_neg_freq_inds

    def get_zero_freq_inds(self):
        return self.zero_freq_inds

    def get_pos_freq_inds(self):
        return self.pos_freq_inds

    def get_neg_freq_inds(self):
        ValueError('no negative frequencies')

    def init_pswf_func2d(self, c, eps):
        """
        initialize the whole set of PSWF functions with the input bandlimit and error.   
        :param c: bandlimit  
        :param eps: error tolerance
        :return:
            alpha_all (list of arrays):
            alpha = alpha_all[i] contains all the eigenvalues for N=i such that lambda > eps,
                    where lambda is the normalized  alpha values (i.e. lambda is between 0 and 1) ,
                    given by lambda=sqrt(c*np.absolute(alpha)/(2*pi)).
            d_vec_all (list of 2D lists): the corresponding eigenvectors for alpha_all
            n_order_length_vec (list of ints): n_order_length_vec[i] = len(alpha_all[i])
        """
        d_vec_all = []
        alpha_all = []
        n_order_length_vec = []

        m = 0
        n = int(np.ceil(2 * c / np.pi))
        #n = self.resolution + 1
        x, w = leggauss_0_1(n)
        #x, w = leggauss_0_1(1000)

        cons = c / 2 / np.pi
        while True:
            alpha, d_vec, a = self.pswf_func2d(m, n, c, eps, x, w)

            # should check this lambda
            lambda_var = np.sqrt(cons * np.absolute(alpha))

            n_end = np.where(lambda_var <= eps)[0]

            if len(n_end) != 0:
                n_end = n_end[0]
                if n_end == 0:
                    break
                n_order_length_vec.extend([n_end])
                alpha_all.append(alpha[:n_end])
                d_vec_all.append(d_vec[:, :n_end])
                m += 1
                # print("generating pswfs for angular index: {}".format(m))
                n = n_end + 1
            else:
                n *= 2
                x, w = leggauss_0_1(n)

        return d_vec_all, alpha_all, n_order_length_vec

    def evaluate_pswf2d_all(self, x, y, max_ns):
        """
        Evaluate the numerical values of PSWF basis functions for all N's, up to certain given n for each N
        :param x: Radial part to evaluate
        :param y: Phase part to evaluate
        :param max_ns: List of ints max_ns[i] is max n to to use for N=i, not included. If max_ns[i]<1  N=i wont be used
        :return: (len(x), sum(max_ns)) ndarray
            Indices are corresponding to the list (N, n)
            (0, 0),..., (0, max_ns[0]), (1, 0),..., (1, max_ns[1]),... , (len(max_ns)-1, 0), (len(max_ns)-1, max_ns[-1])
        """
        max_ns_ints = [int(max_n) for max_n in max_ns]
        out_mat = []
        for i, max_n in enumerate(max_ns_ints):
            if max_n < 1:
                continue

            d_vec = self.d_vec_all[i]

            phase_part = np.exp(1j * i * y) / np.sqrt(2 * np.pi)
            range_array = np.arange(len(d_vec))
            r_radial_part_mat = t_radial_part_mat(x, i, range_array, len(d_vec)).dot(d_vec[:, :max_n])

            # pswf_n_n_mat = r_radial_part_mat * phase_part.reshape((len(phase_part), 1)).dot(np.ones((1, max_n)))
            pswf_n_n_mat = (phase_part * r_radial_part_mat.T)

            out_mat.extend(pswf_n_n_mat)
        out_mat = np.array(out_mat).T
        return out_mat


    def pswf_func2d(self, big_n, n, bandlimit, phi_approximate_error, x, w):

        d_vec, approx_length, range_array = self._pswf_2d_minor_computations(big_n, n, bandlimit, phi_approximate_error)

        t1 = 1 - 2 * np.square(x)
        t2 = np.sqrt(2 * (2 * range_array + big_n + 1))

        phi = t_x_mat(x, big_n, range_array, approx_length).dot(d_vec[:, :(n + 1)])
        phi_derivatives = t_x_derivative_mat(t1, t2, x, big_n, range_array, approx_length).dot(d_vec[:, :(n + 1)])

        max_phi_idx = np.argmax(np.absolute(phi[:, 0]))
        max_phi_val = phi[max_phi_idx, 0]
        x_for_calc = x[max_phi_idx]

        right_hand_side_integral = np.einsum('j, j, j ->', w, k_operator(big_n, bandlimit * x_for_calc * x), phi[:, 0])
        lambda_n_1 = right_hand_side_integral / max_phi_val

        temp_calc = x * w
        upper_integral_values = np.einsum('j, ji, ji -> i', temp_calc, phi_derivatives[:, :-1], phi[:, 1:])
        lower_integral_values = np.einsum('j, ji, ji -> i', temp_calc, phi[:, :-1], phi_derivatives[:, 1:])

        lambda_n = np.append(np.reshape(lambda_n_1, (1, 1)), (
                lambda_n_1 * np.cumprod(upper_integral_values / lower_integral_values).reshape((n, 1))))
        alpha_n = lambda_n * 2 * np.pi * (np.power(1j, big_n) / np.sqrt(bandlimit))

        return alpha_n, d_vec, approx_length

    def _pswf_2d_minor_computations(self, big_n, n, bandlimit, phi_approximate_error):
        """
        approximate the number of n's for fixed  N (big_n) and compute the d_vec defined in eq (18) of paper 3).
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
            (np.sqrt(np.square(2 * n + big_n + 1) + np.square(bandlimit) / 2) - (2 * n + big_n + 1)) / 2)

        d_approx = d_decay_approx_fun(big_n, n, bandlimit, first_idx_for_decrease)
        d_decay_index_counter = first_idx_for_decrease

        while d_approx > phi_approximate_error:
            d_decay_index_counter = d_decay_index_counter + 1
            d_approx = d_approx * d_decay_approx_fun(big_n, n, bandlimit, d_decay_index_counter)
        approx_length = int(n + 1 + d_decay_index_counter)

        d_vec, _ = BNMatrix(big_n, bandlimit, approx_length).get_eig_vectors()

        range_array = np.array(range(approx_length))
        return d_vec, approx_length, range_array
