import os
import pyfftw
import mrcfile
import finufftpy

import scipy.special as sp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.linalg as scl
import scipy.optimize as optim
from console_progressbar import ProgressBar

from numpy.polynomial.legendre import leggauss

from aspire.aspire.common.config import ClassAveragesConfig
from aspire.aspire.utils.data_utils import mat_to_npy, mat_to_npy_vec, load_stack_from_file, c_to_fortran
from aspire.aspire.utils.array_utils import estimate_snr, image_grid, cfft2, icfft2
from aspire.aspire.common.logger import logger
from aspire.aspire.utils.helpers import yellow, set_output_name


class Precomp:
    def __init__(self, n_theta, n_r, resolution, freqs):
        self.n_r = n_r
        self.n_theta = n_theta
        self.resolution = resolution
        self.freqs = freqs


class SpcaData:
    def __init__(self, eigval, freqs, radial_freqs, coeff, mean, bandlimit, supp_num, eig_im, fn0):
        self.eigval = eigval
        self.freqs = freqs
        self.radial_freqs = radial_freqs
        self.coeff = coeff
        self.mean = mean
        self.c = bandlimit
        self.r = supp_num
        self.eig_im = eig_im
        self.fn0 = fn0

    def save(self, obj_dir):
        os.mkdir(obj_dir)
        np.save(obj_dir + '/eigval.npy', self.eigval)
        np.save(obj_dir + '/freqs.npy', self.freqs)
        np.save(obj_dir + '/radial_freqs.npy', self.radial_freqs)
        np.save(obj_dir + '/coeff.npy', self.coeff)
        np.save(obj_dir + '/mean.npy', self.mean)
        np.save(obj_dir + '/c.npy', np.array([self.c]))
        np.save(obj_dir + '/r.npy', np.array([self.r]))
        np.save(obj_dir + '/eig_im.npy', self.eig_im)
        np.save(obj_dir + '/fn0.npy', self.fn0)

    def load(self, obj_dir, matlab=False):
        if matlab:
            files = os.listdir(obj_dir)
            vec_obj = ['c', 'r', 'eigval', 'mean', 'radial_freqs', 'freqs']
            curr_dir = os.getcwd()
            target_dir = os.path.join(curr_dir, obj_dir)
            os.chdir(target_dir)
            for f in files:
                if f[-1] == 't':
                    name = f[:-4]
                    if name in vec_obj:
                        curr_obj = mat_to_npy_vec(name)
                    else:
                        curr_obj = mat_to_npy(name)
                    np.save(name, curr_obj)
            os.chdir(curr_dir)

        self.eigval = np.load(obj_dir + '/eigval.npy')
        self.freqs = np.load(obj_dir + '/freqs.npy')
        self.radial_freqs = np.load(obj_dir + '/radial_freqs.npy')
        self.coeff = np.load(obj_dir + '/coeff.npy')
        self.mean = np.load(obj_dir + '/mean.npy')
        self.c = np.load(obj_dir + '/c.npy')[0]
        self.r = np.load(obj_dir + '/r.npy')[0]
        self.eig_im = np.load(obj_dir + '/eig_im.npy')
        self.fn0 = np.load(obj_dir + '/fn0.npy')


class Basis:
    def __init__(self, phi_ns, angular_freqs, radian_freqs, n_theta):
        self.phi_ns = phi_ns
        self.angular_freqs = angular_freqs
        self.radian_freqs = radian_freqs
        self.n_theta = n_theta


class SamplePoints:
    def __init__(self, x, w):
        self.x = x
        self.w = w


class FastRotatePrecomp:
    def __init__(self, phi, mx, my, mult90):
        self.phi = phi
        self.mx = mx
        self.my = my
        self.mult90 = mult90


def fast_rotate_precomp(szx, szy, phi):
    phi, mult90 = adjust_rotate(phi)

    phi = np.pi * phi / 180
    phi = -phi

    if szy % 2:
        cy = (szy + 1) // 2
        sy = 0
    else:
        cy = szy // 2 + 1
        sy = 0.5

    if szx % 2:
        cx = (szx + 1) // 2
        sx = 0
    else:
        cx = szx // 2 + 1
        sx = 0.5

    my = np.zeros((szy, szx), dtype='complex128')
    r = np.arange(cy)
    r_t = np.arange(szy, cy, -1) - 1
    u = (1 - np.cos(phi)) / np.sin(phi + np.finfo(float).eps)
    alpha1 = 2 * np.pi * 1j * r / szy
    for x in range(szx):
        ux = u * (x + 1 - cx + sx)
        my[r, x] = np.exp(alpha1 * ux)
        my[r_t, x] = np.conj(my[1: cy - 2 * sy, x])

    my = my.T

    mx = np.zeros((szx, szy), dtype='complex128')
    r = np.arange(cx)
    r_t = np.arange(szx, cx, -1) - 1
    u = -np.sin(phi)
    alpha2 = 2 * np.pi * 1j * r / szx
    for y in range(szy):
        uy = u * (y + 1 - cy + sy)
        mx[r, y] = np.exp(alpha2 * uy)
        mx[r_t, y] = np.conj(mx[1: cx - 2 * sx, y])

    # because I am using real fft I take only part of mx and my
    return FastRotatePrecomp(phi, mx[:szx // 2 + 1].copy(), my[:, :szy // 2 + 1].copy(), mult90)


def adjust_rotate(phi):
    phi = phi % 360
    mult90 = 0
    phi2 = phi

    if 45 <= phi < 90:
        mult90 = 1
        phi2 = -(90 - phi)
    elif 90 <= phi < 135:
        mult90 = 1
        phi2 = phi - 90
    elif 135 <= phi < 180:
        mult90 = 2
        phi2 = -(180 - phi)
    elif 180 <= phi < 225:
        mult90 = 2
        phi2 = phi - 180
    elif 215 <= phi < 270:
        mult90 = 3
        phi2 = -(270 - phi)
    elif 270 <= phi < 315:
        mult90 = 3
        phi2 = phi - 270
    elif 315 <= phi < 360:
        mult90 = 0
        phi2 = phi - 360
    return phi2, mult90


def fast_rotate_image(image, phi, tmps=None, plans=None, m=None):
    """
    Could make it faster without the flag 'FFTW_UNALIGNED' if I could make
    :param image:
    :param phi:
    :param tmps:
    :param plans:
    :param m:
    :return:
    """
    szx, szy = image.shape

    if m is None:
        m = fast_rotate_precomp(szx, szy, phi)

    mx = m.mx
    my = m.my

    image[:] = np.rot90(image, m.mult90)

    if tmps is None:
        const_size0 = image.shape[0] // 2 + 1
        const_size1 = image.shape[1] // 2 + 1
        tmp1 = np.empty((len(image), const_size1), dtype='complex128')
        tmp2 = np.empty((const_size0, len(image)), dtype='complex128')
    else:
        tmp1 = tmps[0]
        tmp2 = tmps[1]
    if plans is None:
        tmp01 = pyfftw.empty_aligned(tmp1.shape, tmp1.dtype)
        tmp02 = pyfftw.empty_aligned(tmp2.shape, tmp1.dtype)
        tmp03 = pyfftw.empty_aligned(image.shape, image.dtype)
        plans = [pyfftw.FFTW(tmp03, tmp01), pyfftw.FFTW(tmp01, tmp03, direction='FFTW_BACKWARD'),
                 pyfftw.FFTW(tmp03, tmp02, axes=(0,)), pyfftw.FFTW(tmp02, tmp03, axes=(0,), direction='FFTW_BACKWARD')]

    # first pass
    plan = plans[0]
    plan(image, tmp1)
    tmp1 *= my
    plan = plans[1]
    plan(tmp1, image)

    # second pass
    plan = plans[2]
    plan(image, tmp2)
    tmp2 *= mx
    plan = plans[3]
    plan(tmp2, image)

    # first pass
    plan = plans[0]
    plan(image, tmp1)
    tmp1 *= my
    plan = plans[1]
    plan(tmp1, image)


def get_fast_rotate_vars(resolution):
    tmp1 = np.empty((resolution, resolution // 2 + 1), dtype='complex128')
    tmp2 = np.empty((resolution // 2 + 1, resolution), dtype='complex128')
    tmps = tmp1, tmp2
    tmp01 = pyfftw.empty_aligned(tmp1.shape, tmp1.dtype)
    tmp02 = pyfftw.empty_aligned(tmp2.shape, tmp1.dtype)
    tmp03 = pyfftw.empty_aligned((resolution, resolution), 'float64')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    plans = [pyfftw.FFTW(tmp03, tmp01, flags=flags),
             pyfftw.FFTW(tmp01, tmp03, direction='FFTW_BACKWARD', flags=flags),
             pyfftw.FFTW(tmp03, tmp02, axes=(0,), flags=flags),
             pyfftw.FFTW(tmp02, tmp03, axes=(0,), direction='FFTW_BACKWARD', flags=flags)]

    return tmps, plans


def sort_list_weights_wrefl(classes, corr, rot, refl):
    n_theta = 360
    n, n_nbor = classes.shape
    refl = refl.flatten(order='F')
    classes = classes.flatten(order='F')
    rot = rot.flatten(order='F')
    corr = corr.flatten(order='F')
    refl_is_two = refl == 2
    classes[refl_is_two] += n
    c_list = np.tile(np.arange(n, dtype='int'), n_nbor)
    c_list = np.stack((classes, c_list), axis=1)
    classes[~refl_is_two] += n
    classes[refl_is_two] -= n
    tmp = np.tile(np.arange(n, 2 * n, dtype='int'), n_nbor)
    tmp = np.stack((classes, tmp), axis=1)
    c_list = np.concatenate((c_list, tmp), axis=0)

    x = np.concatenate((corr, corr), axis=0)
    rot = np.concatenate((rot, -rot), axis=0)
    cc_list = np.stack((c_list[:, 1], c_list[:, 0]), axis=1)

    ia, ib = union_row_idx(c_list, cc_list)

    rows = np.concatenate((c_list[ia, 0], cc_list[ib, 0]), axis=0)
    cols = np.concatenate((c_list[ia, 1], cc_list[ib, 1]), axis=0)
    rot_mat = rot.flatten(order='F')
    rot_mat = np.concatenate((rot_mat[ia], -rot_mat[ib]), axis=0)
    x = np.concatenate((x[ia], x[ib]), axis=0)

    rows_smaller_cols = rows < cols
    rows = rows[rows_smaller_cols]
    cols = cols[rows_smaller_cols]
    x = x[rows_smaller_cols]
    rot_mat = rot_mat[rows_smaller_cols]
    ah = np.exp(rot_mat * 2 * np.pi * 1j / n_theta)
    return x, ah, rows, cols


def union_row_idx(a, b):
    _, idx = np.unique(np.concatenate((a, b), axis=0), return_index=True, axis=0)
    # just for debug
    idx = np.sort(idx)
    a_rows = a.shape[0]
    ia = idx[idx < a_rows]
    ib = idx[idx >= a_rows] - a_rows

    return ia, ib


def script_find_graph_weights_v3(x, c, n, k):
    t = len(x)

    a_eq = np.zeros((n, t))
    for i in range(n):
        id1 = np.where(c[:, 0] == i)[0]
        id2 = np.where(c[id1, 1] > i)[0]
        id3 = np.where(c[:, 1] == i)[0]
        id4 = np.where(c[id3, 0] < i)[0]
        idx = np.concatenate((id1[id2], id3[id4]), axis=0)
        a_eq[i, idx] = 1

    b_ub = np.concatenate((np.ones(t), np.zeros(t)), axis=0)
    b_eq = k * np.ones(n)
    a_ub = np.concatenate((np.eye(t), -np.eye(t)), axis=0)
    ans = optim.linprog(x, a_ub, b_ub, a_eq, b_eq)

    return ans.x


def vdm_lp(h1, num_eig):
    """

    :param h1:
    :param num_eig:
    :return:

    (This version of VDM still doesn't work on MATLAB)
    """

    h = np.abs(h1)
    m = h.shape[0]
    c = np.array(h.sum(axis=1)).reshape(m)
    d = sps.csr_matrix((1 / np.sqrt(c), (np.arange(m), np.arange(m))), shape=(m, m))
    h = d.dot(h.dot(d))
    h1 = d.dot(h1.dot(d))
    d1, vv1 = spsl.eigs(h1, k=num_eig)
    d2, vv2 = spsl.eigs(h, k=num_eig + 1)
    vv1 = fix_signs(vv1)
    vv2 = fix_signs(vv2)

    v4 = vv1
    v3 = vv2[:, 1:]

    d1 = d1.real
    d2 = d2.real

    if np.min(d1) < 0:
        keep = np.abs(d1) > np.abs(np.min(d1))
        d1 = d1[keep]
        v4 = v4[:, keep]
        keep = np.abs(d2) > np.abs(np.min(d2))
        d2 = d2[keep]
        v3 = vv2[:, keep]

    delta = 0.1
    t = np.ceil(np.log(delta) / np.log(np.min(np.abs(d1))))
    v4 = np.array([np.power(d1, t)] * len(v4)) * v4
    v3 = np.array([np.power(d2[1:], t)] * len(v3)) * v3
    tmp_list = np.zeros((num_eig * (num_eig + 1) // 2, 2), dtype='int')
    c = np.zeros(num_eig * (num_eig + 1) // 2, dtype='int')
    count = 0
    for i in range(num_eig):
        for j in range(i, num_eig):
            tmp_list[count, 0] = i
            tmp_list[count, 1] = j
            if i == j:
                c[count] = 1
            count += 1

    r_vdm = np.conj(v4[:, tmp_list[:, 0]]) * v4[:, tmp_list[:, 1]]
    r_dm = v3
    r_vdm[:, c == 0] = r_vdm[:, c == 0] * np.sqrt(2)

    r_vdm = (r_vdm.T / np.linalg.norm(r_vdm, axis=1)).T
    r_dm = (r_dm.T / np.linalg.norm(r_dm, axis=1)).T

    return r_vdm, r_dm, v4


def vdm_angle_v2(v, t):
    angle = np.sum(v[t[:, 1]] * np.conj(v[t[:, 0]]), axis=1)
    angle = np.arctan2(angle.imag, angle.real)
    angle = angle * 180 / np.pi
    return angle


def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


def bispec_2drot_large(coeff, freqs, eigval):
    alpha = 1.0 / 3
    freqs_not_zero = freqs != 0
    coeff_norm = np.log(np.power(np.absolute(coeff[freqs_not_zero]), alpha))
    check = coeff_norm == np.float('-inf')
    # should assert if there is a -inf in the coeff norm
    if True in check:
        return 0

    phase = coeff[freqs_not_zero] / np.absolute(coeff[freqs_not_zero])
    phase = np.arctan2(np.imag(phase), np.real(phase))
    eigval = eigval[freqs_not_zero]
    o1, o2 = bispec_operator_1(freqs[freqs_not_zero])

    # don't know if its a bug or not, why 4000?
    n = 4000
    m = np.exp(o1 * np.log(np.power(eigval, alpha)))
    p_m = m / m.sum()
    x = np.random.rand(len(m))
    m_id = np.where(x < n * p_m)[0]
    o1 = o1[m_id]
    o2 = o2[m_id]
    m = np.exp(o1 * coeff_norm + 1j * o2 * phase)

    # svd of the reduced bispectrum
    u, s, v = pca_y(m, 300)
    coeff_b = np.einsum('i, ij -> ij', s, np.conjugate(v))
    coeff_b_r = np.conjugate(u.T).dot(np.conjugate(m))

    coeff_b = coeff_b / np.linalg.norm(coeff_b, axis=0)
    coeff_b_r = coeff_b_r / np.linalg.norm(coeff_b_r, axis=0)
    return coeff_b, coeff_b_r, 0


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

    full_list = np.zeros((count, 3), dtype='int')
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
                full_list[count: count + nd, :2] = tmp1
                full_list[count: count + nd, 2] = tmp2
                count += nd

    val = np.ones(full_list.shape)
    val[:, 2] = -1
    n_col = count
    full_list = full_list.flatten('F')
    val = val.flatten('F')
    col = np.tile(np.arange(n_col), 3)
    o1 = sps.csr_matrix((np.ones(len(full_list)), (col, full_list)), shape=(n_col, len(freqs)))
    o2 = sps.csr_matrix((val, (col, full_list)), shape=(n_col, len(freqs)))
    return o1, o2


# very slow function compared to matlab
def rot_align(m, coeff, pairs):
    n_theta = 360.0
    p = pairs.shape[0]
    c = np.zeros((m + 1, p), dtype='complex128')
    m_list = np.arange(1, m + 1)

    for i in range(m + 1):
        c[i] = np.einsum('ij, ij -> j', np.conj(coeff[i][:, pairs[:, 0]]), coeff[i][:, pairs[:, 1]])

    c2 = np.flipud(np.conj(c[1:]))
    b = (2 * m + 1) * np.real(icfft(np.concatenate((c2, c), axis=0)))
    rot = np.argmax(b, axis=0)
    rot = (rot - m) * n_theta / (2 * m + 1)

    x_old = - np.ones(p)
    x_new = rot
    precision = 0.001
    num_iter = 0

    m_list_ang = m_list * np.pi / 180
    m_list_ang_1j = 1j * m_list_ang
    c_for_f_prime_1 = np.einsum('i, ij -> ji', m_list_ang, c[1:]).copy()
    c_for_f_prime_2 = np.einsum('i, ji -> ji', m_list_ang, c_for_f_prime_1).copy()

    diff = np.absolute(x_new - x_old)
    while np.max(diff) > precision:
        diff = np.absolute(x_new - x_old)
        indices = np.where(diff > precision)[0]
        x_old1 = x_new[indices]
        tmp = np.exp(np.outer(m_list_ang_1j, x_old1))

        delta = np.imag(np.einsum('ji, ij -> j', c_for_f_prime_1[indices], tmp)) / \
                np.real(np.einsum('ji, ij -> j', c_for_f_prime_2[indices], tmp))
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


def cryo_image_contrast(projs, r=None):
    n = projs.shape[0]
    if r is None:
        r = n // 2

    indices = np.where(cart2rad(n) <= r)
    contrast = np.std(projs[indices], axis=0, ddof=1)

    return contrast


def cryo_select_subset(classes, size_output, priority=None, to_image=None, n_skip=None):
    num_images = classes.shape[0]
    num_neighbors = classes.shape[1]
    if to_image is None:
        to_image = num_images

    if n_skip is None:
        n_skip = min(to_image // size_output, num_neighbors)
    # else:
    #     if n_skip > min(to_image // size_output, num_neighbors):
    #         n_skip = min(to_image // size_output, num_neighbors)

    if priority is None:
        priority = np.arange(num_images)

    mask = np.zeros(num_images, dtype='int')
    selected = []
    curr_image_idx = 0

    while len(selected) <= size_output and curr_image_idx < to_image:
        while curr_image_idx < to_image and mask[priority[curr_image_idx]] == 1:
            curr_image_idx += 1
        if curr_image_idx < to_image:
            selected.append(priority[curr_image_idx])
            mask[classes[priority[curr_image_idx], :n_skip]] = 1
            curr_image_idx += 1
    return np.array(selected, dtype='int')[:min(size_output, len(selected))]


def cryo_smart_select_subset(classes, size_output, priority=None, to_image=None):
    num_images = classes.shape[0]
    num_neighbors = classes.shape[1]
    if to_image is None:
        to_image = num_images

    if priority is None:
        priority = np.arange(num_images)

    n_skip = min(to_image // size_output, num_neighbors)
    for i in range(num_neighbors, n_skip - 1, -1):
        selected = cryo_select_subset(classes, size_output, priority, to_image, i)
        if len(selected) == size_output:
            return selected
    return cryo_select_subset(classes, size_output, priority, to_image)


def lgwt(n, a, b):
    """
    Get leggauss points in interval [a, b]

    :param n: number of points
    :param a: interval starting point
    :param b: interval end point
    :returns SamplePoints(x, w): sample points, weight
    """

    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return SamplePoints(x, w)


def bessel_ns_radial(bandlimit, support_size, x):
    bessel = np.load(ClassAveragesConfig.bessel_file)
    bessel = bessel[bessel[:, 3] <= 2 * np.pi * bandlimit * support_size, :]
    angular_freqs = bessel[:, 0]
    max_ang_freq = int(np.max(angular_freqs))
    n_theta = int(np.ceil(16 * bandlimit * support_size))
    if n_theta % 2 == 1:
        n_theta += 1

    radian_freqs = bessel[:, 1]
    r_ns = bessel[:, 2]
    phi_ns = np.zeros((len(x), len(angular_freqs)))
    phi = {}

    pb = ProgressBar(total=100, prefix='bessel_ns_radial', suffix='completed',
                     decimals=0, length=100, fill='%')
    angular_freqs_length = len(angular_freqs)
    for i in range(angular_freqs_length):
        pb.print_progress_bar((i + 1) / angular_freqs_length * 100)
        r0 = x * r_ns[i] / bandlimit
        f = sp.jv(angular_freqs[i], r0)
        # probably the square and the sqrt not needed
        tmp = np.pi * np.square(sp.jv(angular_freqs[i] + 1, r_ns[i]))
        phi_ns[:, i] = f / (bandlimit * np.sqrt(tmp))

    for i in range(max_ang_freq + 1):
        phi[i] = phi_ns[:, angular_freqs == i]

    return Basis(phi, angular_freqs, radian_freqs, n_theta)


def fbcoeff_nfft(split_images, support_size, basis, sample_points, num_threads):
    image_size = split_images[0].shape[0]
    orig = int(np.floor(image_size / 2))
    new_image_size = int(2 * support_size)

    # unpacking input
    phi_ns = basis.phi_ns
    angular_freqs = basis.angular_freqs
    max_angular_freqs = int(np.max(angular_freqs))
    n_theta = basis.n_theta
    x = sample_points.x
    w = sample_points.w
    w = w * x

    # sampling points in the fourier domain
    freqs = pft_freqs(x, n_theta)
    precomp = Precomp(n_theta, len(x), new_image_size, freqs)
    scale = 2 * np.pi / n_theta

    coeff_pos_k = []
    pos_k = []

    pb = ProgressBar(total=100, prefix='fbcoeff_nfft', suffix='completed',
                     decimals=0, length=100, fill='%')
    for i in range(num_threads):
        pb.print_progress_bar((i + 1) / num_threads * 100)
        curr_images = split_images[i]
        # start_pixel = orig - support_size
        # end_pixel = orig + support_size
        # print(start_pixel, end_pixel, curr_images.shape)
        # curr_images = curr_images[start_pixel:end_pixel, start_pixel:end_pixel, :]
        tmp = cryo_pft_nfft(curr_images, precomp)
        pf_f = scale * np.fft.fft(tmp, axis=1)
        pos_k.append(pf_f[:, :max_angular_freqs + 1, :])

    pos_k = np.concatenate(pos_k, axis=2)

    pb = ProgressBar(total=100, prefix='appending', suffix='completed',
                     decimals=0, length=100, fill='%')
    for i in range(max_angular_freqs + 1):
        pb.print_progress_bar((i + 1) / (max_angular_freqs + 1) * 100)
        coeff_pos_k.append(np.einsum('ki, k, kj -> ij', phi_ns[i], w, pos_k[:, i]))

    return coeff_pos_k


def pft_freqs(x, n_theta):
    n_r = len(x)
    d_theta = 2 * np.pi / n_theta

    # sampling points in the fourier domain
    freqs = np.zeros((n_r * n_theta, 2))
    for i in range(n_theta):
        freqs[i * n_r:(i + 1) * n_r, 0] = x * np.sin(i * d_theta)
        freqs[i * n_r:(i + 1) * n_r, 1] = x * np.cos(i * d_theta)

    return freqs


def cryo_pft_nfft(projections, precomp):
    freqs = precomp.freqs
    m = len(freqs)

    n_theta = precomp.n_theta
    n_r = precomp.n_r
    num_projections = projections.shape[2]
    x = -2 * np.pi * freqs.T
    x = x.copy()
    # using nufft
    # import time
    # tic = time.time()
    pf = np.empty((x.shape[1], num_projections), dtype='complex128', order='F')
    finufftpy.nufft2d2many(x[0], x[1], pf, -1, 1e-15, projections)
    # toc = time.time()

    # TODO is it a reference we want to keep?
    # using nudft around 5x slower didn't ret to optimize
    # grid_x, grid_y = image_grid(projections.shape[1])
    # pts = np.array([grid_y.flatten('F'), grid_x.flatten('F')])
    # if projections.shape[1] % 2 == 0:
    #     pts -= 0.5
    #
    # # maybe can do it with less memory by splitting the exponent to several parts
    # pf = np.dot(np.exp(-1j * np.dot(x.T, pts)),
    #           projections.reshape((projections.shape[0] * projections.shape[0]), num_projections,
    #                                 order='F'))
    pf = pf.reshape((n_r, n_theta, num_projections), order='F')

    return pf


def test(im, freqs):
    nj = freqs.shape[1]

    xj = np.random.rand(nj) * 2 * np.pi - np.pi
    yj = np.random.rand(nj) * 2 * np.pi - np.pi

    cj = np.zeros([nj], dtype=np.complex128)
    finufftpy.nufft2d2(xj, yj, cj, -1, 1e-15, im)

    ref = nudft2(im, np.array([xj, yj]))
    logger.info(np.linalg.norm(cj - ref) / np.linalg.norm(ref))


def test2(im, freqs):
    nj = freqs.shape[1]

    if not (np.all(-np.pi <= freqs) and np.all(freqs < np.pi)):
        logger.error('bad frequencies')
        # TODO Itay, should we quit here?

    xj = freqs[0].copy()
    yj = freqs[1].copy()

    cj = np.empty([nj], dtype=np.complex128)
    finufftpy.nufft2d2(xj, yj, cj, -1, 1e-15, im)

    ref = nudft2(im, np.array([xj, yj]))
    logger.info(np.linalg.norm(cj - ref) / np.linalg.norm(ref))


def nudft2(im, freqs):
    grid_x, grid_y = image_grid(im.shape[0])
    pts = np.array([grid_y.flatten('F'), grid_x.flatten('F')])
    if im.shape[0] % 2 == 0:
        pts -= 0.5

    # maybe can do it with less memory by splitting the exponent to several parts
    pf = np.dot(np.exp(-1j * np.dot(freqs.T, pts)), im.flatten('F'))
    return pf


def nufft2(im, freqs):
    freqs = np.mod(freqs + np.pi, 2 * np.pi) - np.pi
    out = np.empty(freqs.shape[1], dtype='complex128')
    finufftpy.nufft2d2(freqs[0], freqs[1], out, -1, 1e-15, im)
    return out


def spca_whole(coeff, var_hat):
    max_ang_freq = len(coeff) - 1
    n_p = coeff[0].shape[1]
    u = []
    d = []
    spca_coeff = []
    mean_coeff = np.mean(coeff[0], axis=1)
    lr = len(coeff)

    pb = ProgressBar(total=100, prefix='spca_whole', suffix='completed',
                     decimals=0, length=100, fill='%')
    for i in range(max_ang_freq + 1):
        pb.print_progress_bar((i + 1) / (max_ang_freq + 1) * 100)
        tmp = coeff[i]
        if i == 0:
            tmp = (tmp.T - mean_coeff).T
            lambda_var = float(lr) / n_p
        else:
            lambda_var = float(lr) / (2 * n_p)

        c1 = np.real(np.einsum('ij, kj -> ik', tmp, np.conj(tmp))) / n_p
        curr_d, curr_u = np.linalg.eig(c1)
        sorted_indices = np.argsort(-curr_d)
        curr_u = fix_signs(curr_u)
        curr_d = curr_d[sorted_indices]
        curr_u = curr_u[:, sorted_indices]

        if var_hat != 0:
            k = len(np.where(curr_d > var_hat * np.square(1 + np.sqrt(lambda_var)))[0])
            if k != 0:
                curr_d = curr_d[:k]
                curr_u = curr_u[:, :k]
                d.append(curr_d)
                u.append(curr_u)
                l_k = 0.5 * ((curr_d - (lambda_var + 1) * var_hat) + np.sqrt(
                    np.square((lambda_var + 1) * var_hat - curr_d) - 4 * lambda_var * np.square(var_hat)))
                snr_i = l_k / var_hat
                snr = (np.square(snr_i) - lambda_var) / (snr_i + lambda_var)
                weight = 1 / (1 + 1 / snr)
                spca_coeff.append(np.einsum('i, ji, jk -> ik', weight, curr_u, tmp))

        else:
            u.append(curr_u)
            d.append(curr_d)
            spca_coeff.append(np.einsum('ji, jk -> ik', curr_u, tmp))

    return u, d, spca_coeff, mean_coeff


def ift_fb(support_size, bandlimit):
    support_size = int(support_size)
    x, y = np.meshgrid(np.arange(-support_size, support_size), np.arange(-support_size, support_size))
    r = np.sqrt(np.square(x) + np.square(y))
    inside_circle = r <= support_size
    theta = np.arctan2(x, y)
    theta = theta[inside_circle]
    r = r[inside_circle]

    bessel = np.load(ClassAveragesConfig.bessel_file)
    bessel = bessel[bessel[:, 3] <= 2 * np.pi * bandlimit * support_size, :]
    k_max = int(np.max(bessel[:, 0]))
    fn = []

    computation1 = 2 * np.pi * bandlimit * r
    computation2 = np.square(computation1)
    c_sqrt_pi_2 = 2 * bandlimit * np.sqrt(np.pi)
    bessel_freqs = bessel[:, 0]
    bessel2 = bessel[:, 2]
    for i in range(k_max + 1):
        bessel_k = bessel2[bessel_freqs == i]
        tmp = np.zeros((2 * support_size, 2 * support_size, len(bessel_k)), dtype='complex128')

        f_r_base = c_sqrt_pi_2 * np.power(-1j, i) * sp.jv(i, computation1)
        f_theta = np.exp(1j * i * theta)

        tmp[inside_circle, :] = np.outer(f_r_base * f_theta, np.power(-1, np.arange(1, len(bessel_k) + 1)) * bessel_k) \
                                / np.subtract.outer(computation2, np.square(bessel_k))

        fn.append(np.transpose(tmp, axes=(1, 0, 2)))

    return fn


def pca_y(x, k, num_iters=2):
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
    if x.dtype == np.dtype('complex'):
        h = operator((2 * np.random.random((k + 2, n)).T - ones) + 1j * (2 * np.random.random((k + 2, n)).T - ones))
    else:
        h = operator(2 * np.random.random((k + 2, n)).T - ones)

    f = [h]
    for i in range(num_iters):
        h = operator_transpose(h)
        h = operator(h)
        f.append(h)

    f = np.concatenate(f, axis=1)

    # f has e-16 error, q has e-13
    q, _, _ = scl.qr(f, mode='economic', pivoting=True)
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


def comp(a, b):
    logger.info(max_dif(a, b))


def max_dif(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def comp_sparse(a, b):
    logger.info(spsl.norm(a - b) / spsl.norm(a))


def max_dif_matrices_sign_invariant(a, b):
    curr_max = 0
    for i in range(a.shape[0]):
        curr_max = max(curr_max, min(max_dif(a[i], b[i]), max_dif(a[i], -b[i])))
    return curr_max


def fix_signs(u):
    """
    makes the matrix coloumn sign be by the biggest value
    :param u: matrix
    :return: matrix
    """
    b = np.argmax(np.absolute(u), axis=0)
    b = np.array([np.linalg.norm(u[b[k], k]) / u[b[k], k] for k in range(len(b))])
    u = u * b
    return u


def fix_svd(u, v):
    b = np.argmax(np.absolute(np.real(u)), axis=0)
    b = np.diag(np.array([np.linalg.norm(u[b[k], k]) / u[b[k], k] for k in range(len(b))]))
    u = u.dot(b)
    v = b.dot(v)
    return u, v


def comp_array_of_npy(a, b):
    n = min(len(a), len(b))
    max_diff = 0
    for i in range(n):
        max_diff = max(max_dif(a[i], b[i]), max_diff)

    return max_diff


def spca_data_compare(a, b):
    logger.info('eigval difference = {}'.format(max_dif(a.eigval, b.eigval)))
    logger.info('freqs difference = {}'.format(max_dif(a.freqs, b.freqs)))
    logger.info('radial_freqs difference = {}'.format(max_dif(a.radial_freqs, b.radial_freqs)))
    logger.info('coeff difference = {}'.format(max_dif(a.coeff, b.coeff)))
    logger.info('mean difference = {}'.format(max_dif(a.mean, b.mean)))
    logger.info('c difference = {}'.format(a.c - b.c))
    logger.info('r difference = {}'.format(a.r - b.r))
    logger.info('eig_im difference = {}'.format(max_dif(a.eig_im.T, b.eig_im.T)))
    logger.info('fn0 difference = {}\n'.format(max_dif(a.fn0, b.fn0)))


def compare_relative_to_classes(a1, b1, a2, b2):
    a1 = a1.copy()
    a2 = a2.copy()
    dif = [0.0] * (len(b1) + 1)
    for i in range(a1.shape[0]):
        for j in range(len(a1[i])):
            t = np.where(a1[i, j] == a2[i])[0]
            if len(t) > 0:
                for k in range(len(b1)):
                    # dif[k + 1] = max(np.abs(b1[k][i, j] - b2[k][i, t[0]]), dif[k + 1])
                    dif[k + 1] += np.abs(b1[k][i, j] - b2[k][i, t[0]])
                a1[i, j] = -1
                a2[i, t[0]] = -1
            else:
                dif[0] += 1
    return dif


def initial_class_comp(a1, a2, b1, b2, c1, c2, d1=None, d2=None):
    # comparing vdm
    if d1 is None:
        dif = compare_relative_to_classes(a1, [b1, c1], a2 - 1, [b2, c2])
        logger.info('classes vdm difference = {}'.format(dif[0]))
        logger.info('classes refl vdm difference = {}'.format(dif[1]))
        logger.info('angle vdm difference = {}\n'.format(dif[2]))

    # comparing initial classses classification
    else:
        dif = compare_relative_to_classes(a1, [b1, c1, d1], a2 - 1, [b2, c2, d2])
        logger.info('classes difference = {}'.format(dif[0]))
        logger.info('classes refl difference = {}'.format(dif[1]))
        logger.info('rot difference = {}'.format(dif[2]))
        logger.info('corr difference = {}\n'.format(dif[3]))


def cart2rad(n):
    n = int(np.floor(n))
    p = (n - 1) / 2
    x, y = np.meshgrid(np.arange(-p, p + 1), np.arange(-p, p + 1))
    center_polar_samples = np.sqrt(np.square(x) + np.square(y))
    return center_polar_samples


class ClassAverages:

    @classmethod
    def run(cls, input_images, output_images, n_nbor=100, nn_avg=50):

        # check if output file already exists
        if os.path.exists(output_images):
            raise Exception(f'outstack already exists! ({yellow(output_images)}) '
                            'please remove or use flag -o for a different outstack.')

        # convert images to numpy based on their type
        images = load_stack_from_file(input_images, c_contiguous=False)

        # TODO adjust all funcitons to work in the same indexing
        images = c_to_fortran(images)

        # estimate snr
        logger.info('estimating snr..')
        snr, signal, noise = estimate_snr(images)

        # spca data
        logger.info('calculating spca data..')
        spca_data = cls.compute_spca(images, noise)

        # initial classification fd update
        logger.info('running initial classification..')
        classes, class_refl, rot, corr, _ = cls.initial_classification_fd_update(spca_data, n_nbor)

        # VDM
        logger.info('skipping vdm..')
        # class_vdm, class_vdm_refl, angle = cls.vdm(classes, np.ones(classes.shape), rot,
        #                                            class_refl, 50, False, 50)

        # align main
        list_recon = np.arange(images.shape[2])
        use_em = True
        logger.info('aligning main..')
        shifts, corr, unsorted_averages_fname, norm_variance = cls.align_main(images, rot,
                                                                              classes,
                                                                              class_refl,
                                                                              spca_data, nn_avg, 15,
                                                                              list_recon,
                                                                              'my_tmpdir',
                                                                              use_em)

        # picking images for abinitio
        logger.info('picking images for align_main')
        contrast = cryo_image_contrast(images)
        contrast_priority = contrast.argsort()
        # num images for abinitio
        size_output = 1000
        to_image = min(len(contrast), 20000)

        # num_neighbors = class_vdm.shape[1]
        # n_skip = min(to_image // size_output, num_neighbors)
        # indices = cryo_select_subset(class_vdm, size_output, contrast_priority, to_image, n_skip)
        indices = cryo_smart_select_subset(classes, size_output, contrast_priority, to_image)

        with mrcfile.new(output_images) as mrc:
            mrc.set_data(unsorted_averages_fname.transpose((2, 1, 0)).astype('float32'))

        subset_projs_name = set_output_name(output_images, 'subset')
        with mrcfile.new(subset_projs_name) as mrc:
            mrc.set_data(unsorted_averages_fname[:, :, indices].transpose((2, 1, 0)).astype('float32'))

    @classmethod
    def compute_spca(cls, images, noise_v_r, adaptive_support=False):
        num_images = images.shape[2]
        resolution = images.shape[0]

        if adaptive_support:
            # TODO debug this
            energy_thresh = 0.99

            # Estimate bandlimit and compact support size
            [bandlimit, support_size] = cls.choose_support_v6(cfft2(images), energy_thresh)
            # Rescale between 0 and 0.5
            bandlimit = bandlimit * 0.5 / np.floor(resolution / 2.0)

        else:
            bandlimit = 0.5
            support_size = int(np.floor(resolution / 2.0))

        n_r = int(np.ceil(4 * bandlimit * support_size))
        basis, sample_points = cls.precompute_fb(n_r, support_size, bandlimit)
        _, coeff, mean_coeff, spca_coeff, u, d = cls.jobscript_ffbspca(images, support_size,
                                                                       noise_v_r,
                                                                       basis, sample_points)

        ang_freqs = []
        rad_freqs = []
        vec_d = []
        pb = ProgressBar(total=100, prefix='compute_spca(1/2)', suffix='completed',
                         decimals=0, length=100, fill='%')
        for i in range(len(d)):
            pb.print_progress_bar((i + 1) / len(d) * 100)
            if len(d[i]) != 0:
                ang_freqs.extend(np.ones(len(d[i]), dtype='int') * i)
                rad_freqs.extend(np.arange(len(d[i])) + 1)
                vec_d.extend(d[i])

        ang_freqs = np.array(ang_freqs)
        rad_freqs = np.array(rad_freqs)
        d = np.array(vec_d)
        k = min(len(d), 400)  # keep the top 400 components
        sorted_indices = np.argsort(-d)
        sorted_indices = sorted_indices[:k]
        d = d[sorted_indices]
        ang_freqs = ang_freqs[sorted_indices]
        rad_freqs = rad_freqs[sorted_indices]

        s_coeff = np.zeros((len(d), num_images), dtype='complex128')
        pb = ProgressBar(total=100, prefix='spca_coeff', suffix='completed',
                         decimals=0, length=100, fill='%')

        for i in range(len(d)):
            pb.print_progress_bar((i + 1) / len(d) * 100)
            s_coeff[i] = spca_coeff[ang_freqs[i]][rad_freqs[i] - 1]

        fn = ift_fb(support_size, bandlimit)

        eig_im = np.zeros((np.square(2 * support_size), len(d)), dtype='complex128')

        # TODO it might be possible to do this faster
        pb = ProgressBar(total=100, prefix='compute_spca(2/2)', suffix='completed',
                         decimals=0, length=100, fill='%')

        for i in range(len(d)):
            pb.print_progress_bar((i + 1) / len(d) * 100)
            tmp = fn[ang_freqs[i]]
            tmp = tmp.reshape((int(np.square(2 * support_size)), tmp.shape[2]), order='F')
            eig_im[:, i] = np.dot(tmp, u[ang_freqs[i]][:, rad_freqs[i] - 1])

        fn0 = fn[0].reshape((int(np.square(2 * support_size)), fn[0].shape[2]), order='F')

        spca_data = SpcaData(d, ang_freqs, rad_freqs, s_coeff, mean_coeff, bandlimit, support_size,
                             eig_im, fn0)
        return spca_data

    @classmethod
    def initial_classification_fd_update(cls, spca_data, n_nbor, is_rand=False):
        # TODO might have a bug here, with less than 10,000 images the error is very large, with more its 0
        # unpacking spca_data
        coeff = spca_data.coeff
        freqs = spca_data.freqs
        eigval = spca_data.eigval

        n_im = coeff.shape[1]
        coeff[freqs == 0] /= np.sqrt(2)
        # could possibly do it faster
        for i in range(n_im):
            coeff[:, i] /= np.linalg.norm(coeff[:, i])

        coeff[freqs == 0] *= np.sqrt(2)
        coeff_b, coeff_b_r, _ = bispec_2drot_large(coeff, freqs, eigval)

        concat_coeff = np.concatenate((coeff_b, coeff_b_r), axis=1)
        del coeff_b_r

        # TODO check if there is a better implementation to NN, use transpose coeff_b might be faster
        if n_im <= 10000:
            # could use einsum
            corr = np.real(np.dot(np.conjugate(coeff_b[:, :n_im]).T, concat_coeff))
            range_arr = np.arange(n_im)
            corr = corr - sps.csr_matrix((np.ones(n_im), (range_arr, range_arr)),
                                         shape=(n_im, 2 * n_im))
            classes = np.argsort(-corr, axis=1)
            classes = classes[:, :n_nbor].A

        else:
            if not is_rand:
                batch_size = 2000
                num_batches = int(np.ceil(1.0 * n_im / batch_size))
                classes = np.zeros((n_im, n_nbor), dtype='int')
                for i in range(num_batches):
                    start = i * batch_size
                    finish = min((i + 1) * batch_size, n_im)
                    corr = np.real(np.dot(np.conjugate(coeff_b[:, start: finish]).T, concat_coeff))
                    classes[start: finish] = np.argsort(-corr, axis=1)[:, 1: n_nbor + 1]
            else:
                # TODO implement random nn
                logger.warning('random nearest neighbors not implemented yet '
                               'using regular one instead')

                batch_size = 2000
                num_batches = int(np.ceil(n_im / batch_size))
                classes = np.zeros((n_im, n_nbor), dtype='int')
                for i in range(num_batches):
                    start = i * batch_size
                    finish = min((i + 1) * batch_size, n_im)
                    corr = np.real(np.dot(np.conjugate(coeff_b[:, start: finish]).T, concat_coeff))
                    classes[start: finish] = np.argsort(-corr, axis=1)[:, 1: n_nbor + 1]

        del coeff_b, concat_coeff
        max_freq = np.max(freqs)
        cell_coeff = []
        for i in range(max_freq + 1):
            cell_coeff.append(
                np.concatenate((coeff[freqs == i], np.conjugate(coeff[freqs == i])), axis=1))

        # maybe pairs should also be transposed
        pairs = np.stack((classes.flatten('F'), np.tile(np.arange(n_im), n_nbor)), axis=1)
        corr, rot = rot_align(max_freq, cell_coeff, pairs)

        rot = rot.reshape((n_im, n_nbor), order='F')
        classes = classes.reshape((n_im, n_nbor), order='F')  # this should already be in that shape
        corr = corr.reshape((n_im, n_nbor), order='F')
        id_corr = np.argsort(-corr, axis=1)
        for i in range(n_im):
            corr[i] = corr[i, id_corr[i]]
            classes[i] = classes[i, id_corr[i]]
            rot[i] = rot[i, id_corr[i]]

        class_refl = np.ceil((classes + 1.0) / n_im).astype('int')
        classes[classes >= n_im] = classes[classes >= n_im] - n_im
        rot[class_refl == 2] = np.mod(rot[class_refl == 2] + 180, 360)
        return classes, class_refl, rot, corr, 0

    @classmethod
    def vdm(cls, classes, corr, rot, class_refl, k, flag, n_nbor):
        n = classes.shape[0]

        x, ah, rows, cols = \
            sort_list_weights_wrefl(classes[:, :k], np.sqrt(2 - 2 * np.real(corr[:, :k])),
                                    rot[:, :k], class_refl[:, :k])
        if flag:
            # TODO bug here, with memory allocation for a_ub, same for matlab, can try use sparse!
            w = script_find_graph_weights_v3(x, np.stack((rows, cols), axis=1), 2 * n, 5)
        else:
            w = np.ones(len(x))

        w2 = w * ah
        w_bigger = w > 0.001
        h2 = sps.csr_matrix((w2[w_bigger], (rows[w_bigger], cols[w_bigger])), shape=(2 * n, 2 * n))
        h2 += np.conj(h2.T)

        r_vdm_lp, _, vv_lp = vdm_lp(h2, 24)

        if n <= 1e4:
            range_n = np.arange(n)
            corr_vdm = r_vdm_lp[:n].dot(np.conj(r_vdm_lp.T))
            corr_vdm = np.real(
                corr_vdm - sps.csr_matrix((np.ones(n), (range_n, range_n)), shape=(n, 2 * n)))
            class_vdm = np.argsort(-corr_vdm, axis=1)
            class_vdm = class_vdm[:n, :n_nbor]

        else:
            n_max = 5000
            i_max = int(np.ceil(1.0 * n / n_max))
            class_vdm = np.zeros((n, n_nbor), dtype='int')
            for i in range(i_max):
                start = i * n_max
                end = min((i + 1) * n_max, n)
                corr_vdm = r_vdm_lp[start:end].dot(np.conjugate(r_vdm_lp.T))
                corr_vdm = np.real(corr_vdm)
                tmp = np.argsort(-corr_vdm, axis=1)
                class_vdm[start:end] = tmp[:, 1:n_nbor + 1]

        class_vdm = np.array(class_vdm)
        class_vdm_refl = np.ceil((class_vdm + 1.) / n).astype('int')
        class_vdm_le_n = class_vdm >= n
        class_vdm[class_vdm_le_n] = class_vdm[class_vdm_le_n] - n
        class_vdm = class_vdm.astype('int')
        flatten_classes = class_vdm.flatten(order='F')
        flatten_classes_refl = class_vdm_refl.flatten(order='F') - 1
        tmp_list = np.column_stack(
            (np.tile(np.arange(n), n_nbor), flatten_classes + flatten_classes_refl * n))
        angle = vdm_angle_v2(vv_lp[:, :10], tmp_list)

        angle = angle.reshape((n, n_nbor), order='F')

        return class_vdm, class_vdm_refl, angle

    @classmethod
    def align_main(cls, data, angle, class_vdm, refl, spca_data, k, max_shifts, list_recon, tmpdir,
                   use_em):
        data = data.swapaxes(0, 2)
        data = data.swapaxes(1, 2)
        data = np.ascontiguousarray(data)
        resolution = data.shape[1]

        if class_vdm.shape[1] < k:
            # raise error
            pass

        shifts = np.zeros((len(list_recon), k + 1), dtype='complex128')
        corr = np.zeros((len(list_recon), k + 1), dtype='complex128')
        norm_variance = np.zeros(len(list_recon))

        m = np.fix(resolution * 1.0 / 2)
        omega_x, omega_y = np.mgrid[-m:m + 1, -m:m + 1]
        omega_x = -2 * np.pi * omega_x / resolution
        omega_y = -2 * np.pi * omega_y / resolution
        omega_x = omega_x.flatten('F')
        omega_y = omega_y.flatten('F')
        a = np.arange(-max_shifts, max_shifts + 1)
        num = len(a)
        a1 = np.tile(a, num)
        a2 = np.repeat(a, num)
        shifts_list = np.column_stack((a1, a2))

        phase = np.ascontiguousarray(
            np.conj(np.exp(1j * (np.outer(omega_x, a1) + np.outer(omega_y, a2))).T))

        angle = np.round(-angle).astype('int')
        angle[angle < 0] += 360

        angle[angle == 360] = 0
        m = []
        for i in range(1, 360):
            m.append(fast_rotate_precomp(resolution, resolution, i))

        n = resolution // 2
        r = spca_data.r
        coeff = spca_data.coeff
        eig_im = spca_data.eig_im
        freqs = spca_data.freqs
        mean_im = np.dot(spca_data.fn0, spca_data.mean)
        output = np.zeros(data.shape)

        # pre allocating stuff
        images = np.zeros((k + 1, resolution, resolution), dtype='float64')
        images2 = np.zeros((k + 1, resolution, resolution), dtype='complex128')
        tmp_alloc = np.zeros((resolution, resolution), dtype='complex128')
        tmp_alloc2 = np.zeros((resolution, resolution), dtype='complex128')
        pf_images = np.zeros((resolution * resolution, k + 1), dtype='complex128')
        pf2 = np.zeros(phase.shape, dtype='complex128')
        c = np.zeros((phase.shape[0], k + 1), dtype='complex128')
        var = np.zeros(resolution * resolution, dtype='float64')
        mean = np.zeros(resolution * resolution, dtype='complex128')
        pf_images_shift = np.zeros((resolution * resolution, k + 1), dtype='complex128')
        tmps, plans = get_fast_rotate_vars(resolution)

        angle_j = np.zeros((k + 1), dtype='int')
        refl_j = np.ones((k + 1), dtype='int')
        index = np.zeros((k + 1), dtype='int')
        import time
        rotate_time = 0
        mult_time = 0
        cfft_time = 0
        multiply_time = 0
        dot_time = 0
        rest_time = 0
        for j in range(len(list_recon)):
            logger.info('starting image {}'.format(j))
            angle_j[1:] = angle[list_recon[j], :k]
            refl_j[1:] = refl[list_recon[j], :k]
            index[1:] = class_vdm[list_recon[j], :k]
            index[0] = list_recon[j]

            for i in range(k + 1):
                if refl_j[i] == 2:
                    images[i] = np.flipud(data[index[i]])
                else:
                    images[i] = data[index[i]]

            tic0 = time.time()
            # 2610 sec for 9000 images, 1021 for matlab
            for i in range(k + 1):
                if angle_j[i] != 0:
                    fast_rotate_image(images[i], angle_j[i], tmps, plans, m[angle_j[i] - 1])
            tic1 = time.time()

            # 190 sec for 9000 images, 103 for matlab
            tmp = np.dot(eig_im[:, freqs == 0], coeff[freqs == 0, list_recon[j]]) + 2 * np.real(
                np.dot(eig_im[:, freqs != 0], coeff[freqs != 0, list_recon[j]])) + mean_im
            tic2 = time.time()

            # 1170 sec for 9000 images, 375 for matlab
            tmp_alloc[n - r:n + r, n - r:n + r] = np.reshape(tmp, (2 * r, 2 * r), 'F')

            pf1 = cfft2(tmp_alloc).flatten('F')
            for i in range(k + 1):
                images2[i] = cfft2(images[i])
            tic3 = time.time()

            # 651 sec for 9000 images, 261 for matlab
            pf_images[:] = images2.reshape((k + 1, resolution * resolution), order='F').T
            np.multiply(phase, np.conj(pf1), out=pf2)
            tic4 = time.time()

            # 313 sec for 9000 images, 233 for matlab
            np.dot(pf2, pf_images, out=c)
            tic5 = time.time()

            # 307 sec for 9000 images, 100 for matlab
            ind = np.lexsort((np.angle(c), np.abs(c)), axis=0)[-1]
            ind_for_c = ind, np.arange(len(ind))
            corr[j] = c[ind_for_c]

            np.multiply(pf_images, phase[ind].T, out=pf_images_shift)
            np.var(pf_images_shift, 1, ddof=1, out=var)
            norm_variance[j] = np.linalg.norm(var)
            np.mean(pf_images_shift, axis=1, out=mean)
            tmp_alloc2[:] = np.reshape(mean, (resolution, resolution), 'F')

            output[j] = np.real(icfft2(tmp_alloc2))
            shifts[j] = -shifts_list[ind, 0] - 1j * shifts_list[ind, 1]
            tic6 = time.time()

            rotate_time += tic1 - tic0
            mult_time += tic2 - tic1
            cfft_time += tic3 - tic2
            multiply_time += tic4 - tic3
            dot_time += tic5 - tic4
            rest_time += tic6 - tic5

        output = output.swapaxes(1, 2)
        output = output.swapaxes(0, 2)
        output = np.ascontiguousarray(output)

        return shifts, corr, output, norm_variance

    @classmethod
    def choose_support_v6(cls, proj_ctf_noisy, energy_threshold):
        """
        Determine sizes of the compact support in both real and Fourier space.

        :param proj_ctf_noisy:
        :param energy_threshold:
        :return: (c_limit, R_limit):
            c_limit: Size of support in Fourier space
            R_limit: Size of support in real space

        We scale the images in real space by L, so that the noise variance in
        both real and Fourier domains is the same.
        """

        L = proj_ctf_noisy.data.shape[1]
        N = int(np.floor(L / 2))
        P = proj_ctf_noisy.data.shape[0]
        x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
        r = np.sqrt(np.square(x) + np.square(y))
        r_flat = r.flatten()
        r_max = N

        img_f = proj_ctf_noisy.data.astype(np.float64)
        img = (icfft2(img_f)) * L
        mean_data = np.mean(img, axis=0)  # Remove mean from the data
        img = img - mean_data

        # Compute the variance of the noise in two different way. See below for the reason.
        img_corner = np.reshape(img, (P, L * L))
        img_corner = img_corner[:, r_flat > r_max]
        img_corner = img_corner.flatten()
        var_img = np.var(img_corner, ddof=1)

        imgf_corner = np.reshape(img_f, (P, L * L))
        imgf_corner = imgf_corner[:, r_flat > r_max]
        imgf_corner = imgf_corner.flatten()
        var_imgf = np.var(imgf_corner, ddof=1)

        noise_var = np.min([var_img, var_imgf])  # Note, theoretical img_f and
        # img should give the same variance but there is a small difference,
        # choose the smaller one so that you don't get a negative variance or power
        # spectrum in 46,47

        variance_map = np.var(img, axis=0, ddof=1)
        variance_map = variance_map.transpose()

        # Mean 2D variance radial function
        radial_var = np.zeros(N)
        for i in range(N):
            radial_var[i] = np.mean(variance_map[np.logical_and(r >= i, r < i + 1)])

        img_ps = np.square(np.abs(img_f))
        pspec = np.mean(img_ps, 0)
        pspec = pspec.transpose()
        radial_pspec = np.zeros(N)

        # Compute the radial power spectrum
        for i in range(N):
            radial_pspec[i] = np.mean(pspec[np.logical_and(r >= i, r < i + 1)])

        # Subtract the noise variance
        radial_pspec = radial_pspec - noise_var
        radial_var = radial_var - noise_var

        # compute the cumulative variance and power spectrum.
        c = np.linspace(0, 0.5, N)
        R = np.arange(0, N)
        cum_pspec = np.zeros(N)
        cum_var = np.zeros(N)

        for i in range(N):
            cum_pspec[i] = np.sum(np.multiply(radial_pspec[0:i + 1], c[0:i + 1]))
            cum_var[i] = np.sum(np.multiply(radial_var[0:i + 1], R[0:i + 1]))

        cum_pspec = cum_pspec / cum_pspec[-1]
        cum_var = cum_var / cum_var[-1]

        cidx = np.where(cum_pspec > energy_threshold)
        c_limit = c[cidx[0][0] - 1] * L
        Ridx = np.where(cum_var > energy_threshold)
        R_limit = R[Ridx[0][0] - 1]

        return c_limit, R_limit

    @classmethod
    def precompute_fb(cls, n_r, support_size, bandlimit):
        sample_points = lgwt(n_r, 0, bandlimit)
        basis = bessel_ns_radial(bandlimit, support_size, sample_points.x)
        return basis, sample_points

    @classmethod
    def jobscript_ffbspca(cls, images, support_size, noise_var, basis, sample_points, num_threads=10):
        split_images = np.array_split(images, num_threads, axis=2)
        del images

        coeff = fbcoeff_nfft(split_images, support_size, basis, sample_points, num_threads)
        u, d, spca_coeff, mean_coeff = spca_whole(coeff, noise_var)
        return 0, coeff, mean_coeff, spca_coeff, u, d
