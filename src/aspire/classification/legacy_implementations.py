import logging

import numpy as np
from scipy.linalg import qr

logger = logging.getLogger(__name__)


def pca_y(x, k, num_iters=2):
    """
    PCA using QR factorization.

    See:

    An algorithm for the principal component analysis of large data sets.
    Halko, Martinsson, Shkolnisky, Tygert , SIAM 2011.

    :param x: Data matrix
    :param k: Number of estimated Principal Components.
    :param num_iters: Number of dot product applications.
    :return: (left Singular Vectors, Singular Values, right Singular Vectors)
    """

    m, n = x.shape

    def operator(mat):
        return x.dot(mat)

    def operator_transpose(mat):
        return np.conj(x.T).dot(mat)

    flag = False
    if m < n:
        flag = True
        operator_transpose, operator = operator, operator_transpose
        m, n = n, m

    ones = np.ones((n, k + 2))
    if x.dtype == np.dtype("complex"):
        h = operator(
            (2 * np.random.random((k + 2, n)).T - ones)
            + 1j * (2 * np.random.random((k + 2, n)).T - ones)
        )
    else:
        h = operator(2 * np.random.random((k + 2, n)).T - ones)

    f = [h]

    for _ in range(num_iters):
        h = operator_transpose(h)
        h = operator(h)
        f.append(h)

    f = np.concatenate(f, axis=1)
    # f has e-16 error, q has e-13
    q, _, _ = qr(f, mode="economic", pivoting=True)
    b = np.conj(operator_transpose(q)).T
    u, s, v = np.linalg.svd(b, full_matrices=False)
    # not sure how to fix the signs but it seems like I dont need to
    # TODO use fix_svd, here and matlab
    # u, v = fix_svd(u, v)

    v = v.conj()
    u = np.dot(q, u)

    u = u[:, :k]
    v = v[:k]
    s = s[:k]

    if flag:
        u, v = v.T, u.T

    return u, s, v


# copied for debugging/poc purposes
def icfft2(x):
    if len(x.shape) == 2:
        return np.fft.fftshift(
            np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x))))
        )
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, (1, 2))
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.ifft2(y)
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.fftshift(y, (1, 2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


# lol
def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


# copied for debugging/poc purposes
# very slow function compared to matlab
def rot_align(m, coeff, pairs):
    n_theta = 360.0
    p = pairs.shape[0]
    c = np.zeros((m + 1, p), dtype="complex128")
    m_list = np.arange(1, m + 1)

    for i in range(m + 1):
        c[i] = np.einsum(
            "ij, ij -> j", np.conj(coeff[i][:, pairs[:, 0]]), coeff[i][:, pairs[:, 1]]
        )

    c2 = np.flipud(np.conj(c[1:]))
    b = (2 * m + 1) * np.real(icfft(np.concatenate((c2, c), axis=0)))
    rot = np.argmax(b, axis=0)
    rot = (rot - m) * n_theta / (2 * m + 1)

    x_old = -np.ones(p)
    x_new = rot
    precision = 0.001
    num_iter = 0

    m_list_ang = m_list * np.pi / 180
    m_list_ang_1j = 1j * m_list_ang
    c_for_f_prime_1 = np.einsum("i, ij -> ji", m_list_ang, c[1:]).copy()
    c_for_f_prime_2 = np.einsum("i, ji -> ji", m_list_ang, c_for_f_prime_1).copy()

    diff = np.absolute(x_new - x_old)
    while np.max(diff) > precision:
        diff = np.absolute(x_new - x_old)
        indices = np.where(diff > precision)[0]
        x_old1 = x_new[indices]
        tmp = np.exp(np.outer(m_list_ang_1j, x_old1))

        delta = np.imag(
            np.einsum("ji, ij -> j", c_for_f_prime_1[indices], tmp)
        ) / np.real(np.einsum("ji, ij -> j", c_for_f_prime_2[indices], tmp))
        delta_bigger10 = np.where(np.abs(delta) > 10)[0]
        tmp_random = np.random.rand(len(delta))
        tmp_random = tmp_random[delta_bigger10]
        delta[delta_bigger10] = np.sign(delta_bigger10) * 10 * tmp_random
        x_new[indices] = x_old1 - delta
        num_iter += 1
        if num_iter > 100:
            break

    rot = x_new
    m_list = np.arange(m + 1)
    m_list_ang = m_list * np.pi / 180
    c = c * np.exp(1j * np.outer(m_list_ang, rot))
    corr = (np.real(c[0]) + 2 * np.sum(np.real(c[1:]), axis=0)) / 2

    return corr, rot
