import numpy as np


def generate_bn_mat(n, c, approx_length):

    k = np.arange(1, approx_length, dtype=float)
    diagonal = np.ones(approx_length)

    c_square = np.square(c)

    diagonal[0] = generate_bn_mat_b_n_on_diagonal(n, 0, c)
    diagonal[1:] = -((n + 2 * k + 0.5) * (n + 2 * k + 1.5) + c_square * generate_bn_mat_gamma_0(n, k))

    off_diagonal = generate_bn_mat_b_n_above_diagonal(n, k, c)
    off_diagonal += generate_bn_mat_b_n_below_diagonal(n, k - 1, c)
    off_diagonal /= 2

    return diagonal, off_diagonal


def generate_bn_mat_h(n, k):
    """
    Defined in the paper eq (17) (basic equation) + (19) (the usage equation)
    """
    
    return np.sqrt(2 * (2 * k + n + 1))


def generate_bn_mat_gamma_plus_1(n, k):
    """
    Defined in the paper eq (24)
    """
    
    return -((np.square(k + n + 1) * generate_bn_mat_h(n, k)) / ((2 * k + n + 1) * (2 * k + n + 2) *
                                                                 generate_bn_mat_h(n, k + 1))) * ((k + 1) / (k + n + 1))


def generate_bn_mat_gamma_0(n, k):
    """
    Defined in the paper eq (24)
    """
    
    if n == 0:
        return 0.5
    return (2.0 * k * (k + 1) + n * (2 * k + n + 1)) / ((2 * k + n) * (2 * k + n + 2))


def generate_bn_mat_gamma_minus_1(n, k):
    """
    Defined in the paper eq (24)
    """
    
    return -((np.square(k) * generate_bn_mat_h(n, k)) / ((2 * k + n + 1) * (2 * k + n) * generate_bn_mat_h(n, k - 1)))\
           * ((n + k) / k)


def generate_bn_mat_k(n, k):
    """
    Defined in the paper below eq (20)
    """

    return (n + 2 * k + 0.5) * (n + 2 * k + 1.5)


def generate_bn_mat_b_n_above_diagonal(n, k, c):
    """
    Defined in the paper eq (26)
    """

    return -np.square(c) * generate_bn_mat_gamma_minus_1(n, k)


def generate_bn_mat_b_n_on_diagonal(n, k, c):
    """
    Defined in the paper eq (26)
    """

    return -(generate_bn_mat_k(n, k) + np.square(c) * generate_bn_mat_gamma_0(n, k))


def generate_bn_mat_b_n_below_diagonal(n, k, c):
    """
    Defined in the paper eq (26)
    """

    return -np.square(c) * generate_bn_mat_gamma_plus_1(n, k)
