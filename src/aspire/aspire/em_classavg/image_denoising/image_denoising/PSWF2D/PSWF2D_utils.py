from aspire.em_classavg.image_denoising.image_denoising.PSWF2D.GeneralFunctions import pswf_2d, p_n, leggauss_0_1
import numpy as np


def init_pswf_2d(c, eps):
    d_vec_all = []
    alpha_all = []
    n_order_length_vec = []

    m = 0
    n = int(np.ceil(2 * c / np.pi))
    x, w = leggauss_0_1(n)
    cons = c / 2 / np.pi
    while True:
        alpha, d_vec, a = pswf_2d(m, n, c, eps, x, w)

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


def t_radial_part_mat(x, n, j, m):
    a = np.power(x, n)
    b = np.sqrt(2 * (2 * j + n + 1))
    c = p_n(m - 1, n, 0, 1 - 2 * np.square(x))
    return np.einsum('i,j,ij->ij', a, b, c)
