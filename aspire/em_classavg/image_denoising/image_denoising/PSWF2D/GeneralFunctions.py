import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jn
from aspire.em_classavg.image_denoising.image_denoising.PSWF2D.BN.BN import BN
import time


def pswf_2d(big_n, n, bandlimit, phi_approximate_error, x, w):
    tic0 = time.clock()
    # x, w = leggauss_0_1(20 * n)

    tic1 = time.clock()
    d_vec, approx_length, range_array = pswf_2d_minor_computations(big_n, n, bandlimit, phi_approximate_error)

    tic2 = time.clock()
    t1 = 1 - 2 * np.square(x)
    t2 = np.sqrt(2 * (2 * range_array + big_n + 1))

    # defined in eq (19) in the paper
    t_x_derivative_mat = -2 * (big_n + range_array + 1) * np.outer(np.power(x, big_n + 1.5), t2) *\
        np.column_stack((np.zeros(len(x)), p_n(approx_length - 2, big_n + 1, 1, t1))) +\
        (big_n + 0.5) * np.outer(np.power(x, big_n - 0.5), t2) * p_n(approx_length - 1, big_n, 0, t1)

    tic3 = time.clock()
    phi = t_x_mat(x, big_n, range_array, approx_length).dot(d_vec[:, :(n + 1)])
    phi_derivatives = t_x_derivative_mat.dot(d_vec[:, :(n + 1)])

    tic4 = time.clock()
    max_phi_idx = np.argmax(np.absolute(phi[:, 0]))
    max_phi_val = phi[max_phi_idx, 0]
    x_for_calc = x[max_phi_idx]

    tic5 = time.clock()
    right_hand_side_integral = np.einsum('j, j, j ->', w, k_operator(big_n, bandlimit * x_for_calc * x), phi[:, 0])
    lambda_n_1 = right_hand_side_integral / max_phi_val

    tic6 = time.clock()
    # upper_integral_values = np.diag((temp_calc * phi_derivatives[:, :-1]).transpose().dot(phi[:, 1:]))
    # lower_integral_values = np.diag((temp_calc * phi[:, :-1]).transpose().dot(phi_derivatives[:, 1:]))

    temp_calc = x * w
    upper_integral_values = np.einsum('j, ji, ji -> i', temp_calc, phi_derivatives[:, :-1], phi[:, 1:])
    lower_integral_values = np.einsum('j, ji, ji -> i', temp_calc, phi[:, :-1], phi_derivatives[:, 1:])

    tic7 = time.clock()
    lambda_n = np.append(np.reshape(lambda_n_1, (1, 1)), (
            lambda_n_1 * np.cumprod(upper_integral_values / lower_integral_values).reshape((n, 1))))
    alpha_n = lambda_n * 2 * np.pi * (np.power(1j, big_n) / np.sqrt(bandlimit))

    tic8 = time.clock()

    # full_time = tic8 - tic0
    # print('leggauss time {}'.format((tic1 - tic0)/full_time))
    # print('d_vec time {}'.format((tic2 - tic1)/full_time))
    # print('t_x_derivative_mat time {}'.format((tic3 - tic2)/full_time))
    # print('phi time {}'.format((tic4 - tic3)/full_time))
    # print('x_for_calc time {}'.format((tic5 - tic4)/full_time))
    # print('lambda_n_1 time {}'.format((tic6 - tic5)/full_time))
    # print('integral_values time {}'.format((tic7 - tic6)/full_time))
    # print('finish time {}\n'.format((tic8 - tic7)/full_time))

    # print('leggauss time {}'.format(tic1 - tic0))
    # print('d_vec time {}'.format(tic2 - tic1))
    # print('t_x_derivative_mat time {}'.format(tic3 - tic2))
    # print('phi time {}'.format(tic4 - tic3))
    # print('x_for_calc time {}'.format(tic5 - tic4))
    # print('lambda_n_1 time {}'.format(tic6 - tic5))
    # print('integral_values time {}'.format(tic7 - tic6))
    # print('finish time {}'.format(tic8 - tic7))
    # print('total time {}\n'.format(tic8 - tic0))

    return alpha_n, d_vec, approx_length


def pswf_2d_minor_computations(big_n, n, bandlimit, phi_approximate_error):
    """
    approximate the number of n's, and compute the d_vec defined in eq (18).
    :param big_n: int
    :param n: int
        used for the computation of approx_length
    :param bandlimit: float > 0
    :param phi_approximate_error: float > 0
    :return: d_vec: (approx_length, approx_length) ndarray
        d_vec[:, i] = d^{N, i} defined from eq (18)
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

    d_vec, _ = BN(big_n, bandlimit, approx_length).get_eig_vectors()

    range_array = np.array(range(approx_length))
    return d_vec, approx_length, range_array


def j_polynomial(m, n, alpha, beta, x):
    """
    The Jacobi polynomials defined in the paper, eq (2), page 6
    :param m: int, > 0
        The dimension of x
    :param n: int, > 0
        Number of polynomials to compute
    :param alpha: float, > -1
    :param beta: float, > -1
    :param x: (m,) ndarray
    :return: v: (m, n + 1) ndarray
        v[:, i] = P^{(alpha, beta)}_n(x) as defined in the paper
    """

    if n < 0:
        return np.array([])
    if n == 0:
        return np.ones(m)
    x = x.reshape((m,))
    v = np.zeros((m, n + 1))
    alpha_p_beta = alpha + beta
    alpha_m_beta = alpha - beta
    v[:, 0] = 1
    v[:, 1] = (1 + 0.5 * alpha_p_beta) * x + 0.5 * alpha_m_beta

    for i in range(2, n + 1):
        c1 = 2 * i * (i + alpha_p_beta) * (2 * i + alpha_p_beta - 2)
        c2 = (2 * i + alpha_p_beta) * (2 * i + alpha_p_beta - 1) * (2 * i + alpha_p_beta - 2)
        c3 = (2 * i + alpha_p_beta - 1) * alpha_p_beta * alpha_m_beta
        c4 = -2 * (i - 1 + alpha) * (i - 1 + beta) * (2 * i + alpha_p_beta)
        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v


def d_decay_approx_fun(a, b, c, d): return np.square(c) / (16 * (np.square(d) + d * (2 * b + a + 1)) - np.square(c))


def p_n(n, alpha, beta, x):
    """
    wrapper to j_polynomial, returns the first n jacobi polynomial of x
    """
    return j_polynomial(len(x), n, alpha, beta, x)


def t_x_mat(x, n, j, approx_length):
    a = np.power(x, n + 0.5)
    b = np.sqrt(2 * (2 * j + n + 1))
    c = p_n(approx_length - 1, n, 0, 1 - 2 * np.square(x))
    return np.einsum('i,j,ij->ij', a, b, c)


def k_operator(nu, x): return jn(nu, x) * np.sqrt(x)


def leggauss_0_1(n):
    """
    Wrapper for numpy.polynomial leggauss
    :param n: int > 0, the number of sampled points to integrate
    :return: Legendre-Gauss Quadrature of degree n between 0 and 1
    """
    sample_points, weights = leggauss(n)
    sample_points = (sample_points + 1) / 2
    weights = weights / 2
    return sample_points, weights
