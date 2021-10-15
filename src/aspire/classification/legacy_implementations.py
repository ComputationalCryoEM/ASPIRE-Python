import logging

import numpy as np
import scipy.sparse as sps
from scipy.linalg import qr

from aspire.numeric import fft

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
    b = (2 * m + 1) * np.real(
        fft.centered_ifft(np.concatenate((c2, c), axis=0), axis=0)
    )
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


def bispec_operator_1(freqs):
    max_freq = np.max(freqs)
    count = 0
    for i in range(2, max_freq):
        for j in range(1, min(i, max_freq - i + 1)):
            k = i + j
            id1 = np.where(freqs == i)[0]
            id2 = np.where(freqs == j)[0]
            id3 = np.where(freqs == k)[0]
            nd1 = len(id1)
            nd2 = len(id2)
            nd3 = len(id3)
            count += nd1 * nd2 * nd3

    full_list = np.zeros((count, 3), dtype="int")
    count = 0
    for i in range(2, max_freq):
        for j in range(1, min(i, max_freq - i + 1)):
            k = i + j
            id1 = np.where(freqs == i)[0]
            id2 = np.where(freqs == j)[0]
            id3 = np.where(freqs == k)[0]
            nd1 = len(id1)
            nd2 = len(id2)
            nd3 = len(id3)
            nd = nd1 * nd2 * nd3
            if nd != 0:
                tmp1 = np.tile(id1, nd2)
                tmp2 = np.repeat(id2, nd1)
                tmp = np.stack((tmp1, tmp2), axis=1)
                tmp1 = np.tile(tmp, (nd3, 1))
                tmp2 = np.repeat(id3, nd1 * nd2)
                full_list[count : count + nd, :2] = tmp1
                full_list[count : count + nd, 2] = tmp2
                count += nd

    val = np.ones(full_list.shape)
    val[:, 2] = -1  # conjugation
    n_col = count
    full_list = full_list.flatten("F")
    val = val.flatten("F")
    col = np.tile(np.arange(n_col), 3)
    o1 = sps.csr_matrix(
        (np.ones(len(full_list)), (col, full_list)), shape=(n_col, len(freqs))
    )
    o2 = sps.csr_matrix((val, (col, full_list)), shape=(n_col, len(freqs)))
    return o1, o2


def bispec_2drot_large(coeff, freqs, eigval, alpha, sample_n):
    """
    alpha 1/3
    sample_n 4000
    """
    freqs_not_zero = freqs != 0

    coeff_norm = np.log(np.power(np.absolute(coeff[freqs_not_zero]), alpha))
    if np.any(coeff_norm == float("-inf")):
        raise ValueError("coeff_norm should not be -inf")

    phase = coeff[freqs_not_zero] / np.absolute(coeff[freqs_not_zero])
    phase = np.arctan2(np.imag(phase), np.real(phase))
    eigval = eigval[freqs_not_zero]
    o1, o2 = bispec_operator_1(freqs[freqs_not_zero])

    # GBW, naively handle vanishing eigvals.
    #  This became a problem with very noisy images...
    p = np.power(eigval, alpha)
    mask = np.where(p, p, -1)  # taking the log in the next step will yield a 0
    m = np.exp(o1 * np.log(p, where=(mask > 0)))
    p_m = m / m.sum()
    x = np.random.rand(len(m))
    m_id = np.where(x < sample_n * p_m)[0]
    o1 = o1[m_id]
    o2 = o2[m_id]
    m = np.exp(o1 * coeff_norm + 1j * o2 * phase)

    # svd of the reduced bispectrum
    u, s, v = pca_y(m, 300)

    coeff_b = np.einsum("i, ij -> ij", s, np.conjugate(v))
    coeff_b_r = np.conjugate(u.T).dot(np.conjugate(m))

    coeff_b = coeff_b / np.linalg.norm(coeff_b, axis=0)
    coeff_b_r = coeff_b_r / np.linalg.norm(coeff_b_r, axis=0)

    return coeff_b, coeff_b_r
