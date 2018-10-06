from numpy.linalg import lstsq
from scipy.optimize import least_squares
import numpy as np
from scipy.special import jn
from image_denoising.image_denoising.PSWF2D.GeneralFunctions import leggauss_0_1, pswf_2d, j_polynomial, pswf_2d_minor_computations, t_x_mat


def generate_pswf_quad(n, bandlimit, phi_approximate_error, lambda_max, epsilon):
    radial_quad_points, radial_quad_weights = generate_pswf_radial_quad(n, bandlimit, phi_approximate_error,
                                                                        lambda_max)

    num_angular_points = np.ceil(np.e * radial_quad_points * bandlimit / 2 - np.log(epsilon)).astype('int') + 1

    for i in range(len(radial_quad_points)):
        ang_error_vec = np.absolute(jn(range(1, 2 * num_angular_points[i] + 1),
                                       bandlimit * radial_quad_points[i]))

        num_angular_points[i] = sum_minus_cumsum_smaller_eps(ang_error_vec, epsilon)
        if num_angular_points[i] % 2 == 1:
            num_angular_points[i] += 1

    temp = 2 * np.pi / num_angular_points

    t = 2

    quad_rule_radial_weights = temp * radial_quad_points * radial_quad_weights
    quad_rule_weights = np.repeat(quad_rule_radial_weights, repeats=num_angular_points)
    quad_rule_pts_r = np.repeat(radial_quad_points, repeats=(num_angular_points / t).astype('int'))
    quad_rule_pts_theta = np.concatenate([temp[i] * np.arange(num_angular_points[i] / t)
                                          for i in range(len(radial_quad_points))])

    pts_x = quad_rule_pts_r * np.cos(quad_rule_pts_theta)
    pts_y = quad_rule_pts_r * np.sin(quad_rule_pts_theta)

    return pts_x, pts_y, quad_rule_weights, radial_quad_points, quad_rule_radial_weights, num_angular_points


def generate_pswf_radial_quad(n, bandlimit, phi_approximate_error, lambda_max):
    x, w = leggauss_0_1(20 * n)

    big_n = 0

    x_as_mat = x.reshape((len(x), 1))

    alpha_n, d_vec, approx_length = pswf_2d(big_n, n, bandlimit, phi_approximate_error, x, w)

    cut_indices = np.where(bandlimit / 2 / np.pi * np.absolute(alpha_n) < lambda_max)[0]

    if len(cut_indices) == 0:
        k = len(alpha_n)
    else:
        k = cut_indices[0]

    if k % 2 == 0:
        k = k + 1

    range_array = np.arange(approx_length).reshape((1, approx_length))

    idx_for_quad_nodes = int((k + 1) / 2)
    num_quad_pts = idx_for_quad_nodes - 1

    phi_zeros = find_initial_nodes(x, n, bandlimit / 2, phi_approximate_error, idx_for_quad_nodes)

    def phi_for_quad_weights(t):
        return np.dot(t_x_mat2(t, big_n, range_array, approx_length), d_vec[:, :k - 1])

    b = np.dot(w * np.sqrt(x), phi_for_quad_weights(x_as_mat))

    a = phi_for_quad_weights(phi_zeros.reshape((len(phi_zeros), 1))).transpose() * np.sqrt(phi_zeros)
    init_quad_weights = lstsq(a, b)
    init_quad_weights = init_quad_weights[0]
    tolerance = 1e-16

    def obj_func(quad_rule):
        q = quad_rule.reshape((len(quad_rule), 1))
        temp = np.dot((phi_for_quad_weights(q[:num_quad_pts]) * np.sqrt(q[:num_quad_pts])).transpose(),
                      q[num_quad_pts:])
        temp = temp.reshape(temp.shape[0])
        return temp - b

    arr_to_send = np.concatenate((phi_zeros, init_quad_weights))
    quad_rule_final = least_squares(obj_func, arr_to_send, xtol=tolerance, ftol=tolerance, max_nfev=1000)
    quad_rule_final = quad_rule_final.x
    quad_rule_pts = quad_rule_final[:num_quad_pts]
    quad_rule_weights = quad_rule_final[num_quad_pts:]
    return quad_rule_pts, quad_rule_weights


def find_initial_nodes(x, n, bandlimit, phi_approximate_error, idx_for_quad_nodes):
    big_n = 0

    d_vec, approx_length, range_array = pswf_2d_minor_computations(big_n, n, bandlimit, phi_approximate_error)

    def phi_for_quad_nodes(t):
        return np.dot(t_x_mat(t, big_n, range_array, approx_length), d_vec[:, idx_for_quad_nodes - 1])

    fun_vec = phi_for_quad_nodes(x)
    sign_flipping_vec = np.where(np.sign(fun_vec[:-1]) != np.sign(fun_vec[1:]))[0]
    phi_zeros = np.zeros(idx_for_quad_nodes - 1)

    tmp = phi_for_quad_nodes(x)
    for i, j in enumerate(sign_flipping_vec[:idx_for_quad_nodes - 1]):
        new_zero = x[j] - tmp[j] * \
                   (x[j + 1] - x[j]) / (tmp[j + 1] - tmp[j])
        phi_zeros[i] = new_zero

    phi_zeros = np.array(phi_zeros)
    return phi_zeros


def sum_minus_cumsum_smaller_eps(x, eps):
    y = np.cumsum(np.flipud(x))
    return len(y) - np.where(y > eps)[0][0] + 1


def parameters_for_forward(resolution, beta, pswf_quad_int):
    bandlimit = beta * np.pi * resolution

    us_fft_pts = np.column_stack((pswf_quad_int.quad_rule_pts_x, pswf_quad_int.quad_rule_pts_y))

    # us_fft_pts = bandlimit / resolution * us_fft_pts.T  # for gal's lib
    us_fft_pts = bandlimit / (resolution * np.pi * 2) * us_fft_pts  # for pynfft
    # us_fft_pts = (bandlimit / resolution) * us_fft_pts  # for pynufft

    blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max =\
        pswf_integration_sub_routine(bandlimit, resolution, pswf_quad_int)

    return us_fft_pts, blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max


def pswf_integration_sub_routine(bandlimit, resolution, pswf_quad_int):
    angular_frequency = pswf_quad_int.angular_frequency
    pswf_radial_quad = pswf_quad_int.pswf_radial_quad

    t = 2

    num_angular_pts = (pswf_quad_int.num_angular_pts / t).astype('int')

    r_quad_indices = [0]
    r_quad_indices.extend(num_angular_pts)
    r_quad_indices = np.cumsum(r_quad_indices, dtype='int')

    n_max = int(max(pswf_quad_int.angular_frequency) + 1)

    numel_for_n = np.zeros(n_max, dtype='int')
    for i in range(n_max):
        numel_for_n[i] = np.count_nonzero(angular_frequency == i)

    indices_for_n = [0]
    indices_for_n.extend(numel_for_n)
    indices_for_n = np.cumsum(indices_for_n, dtype='int')

    blk_r = [0] * n_max
    temp_const = bandlimit / (2 * np.pi * resolution)
    for i in range(n_max):
        blk_r[i] = temp_const * pswf_radial_quad[:, indices_for_n[i] + np.arange(numel_for_n[i])].T

    return blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max


def p_n(n, alpha, beta, x):
    """
    wrapper to j_polynomial, returns the first n jacobi polynomial of x
    """
    return j_polynomial(len(x), n, alpha, beta, x)


# use np.outer instead of x as mat
def t_x_mat2(x, n, j, approx_length): return np.power(x, n + 0.5).dot(np.sqrt(2 * (2 * j + n + 1))) * \
                                            p_n(approx_length - 1, n, 0, 1 - 2 * np.square(x))
