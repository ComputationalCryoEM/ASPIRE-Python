import math
import time
import finufftpy

import numpy as np
import scipy.special as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from pyfftw.interfaces import numpy_fft

from aspire.aspire.common.config import AbinitioConfig
from aspire.aspire.common.exceptions import WrongInput
from aspire.aspire.common.logger import logger
from aspire.aspire.utils.data_utils import validate_square_projections, c_to_fortran


class DiracBasis:
    # for now doesn't work with mask, probably there is no need for it too.
    def __init__(self, sz, mask=None):
        if mask is None:
            mask = np.ones(sz)

        self.type = 0  # TODO define a constant for this

        self.sz = sz
        self.mask = mask
        self.count = mask.size
        self.sz_prod = np.count_nonzero(sz)

    def evaluate(self, x):
        if x.shape[0] != self.count:
            raise ValueError('First dimension of input must be of size basis.count')

        return x.reshape(self.sz, order='F').copy()

    def expand(self, x):
        if len(x.shape) < len(self.sz) or x.shape[:len(self.sz)] != self.sz:
            raise ValueError('First {} dimension of input must be of size basis.count'.format(len(self.sz)))

        return x.flatten('F').copy()

    def evaluate_t(self, x):
        return self.expand(x)

    def expand_t(self, x):
        return self.expand(x)


class Abinitio:
    """ Abinitio class contains all functions related to Abinitio algorithm. """

    @classmethod
    def cryo_abinitio_c1_worker(cls, stack):

        stack = c_to_fortran(stack)
        validate_square_projections(stack)

        resolution = stack.shape[1]
        num_projections = stack.shape[2]

        n_r = math.ceil(AbinitioConfig.n_r * resolution)
        max_shift = math.ceil(AbinitioConfig.max_shift * resolution)

        mask_radius = resolution * 0.45
        # mask_radius is of the form xxx.5
        if mask_radius * 2 == int(mask_radius * 2):
            mask_radius = math.ceil(mask_radius)
        # mask is not of the form xxx.5
        else:
            mask_radius = int(round(mask_radius))

        # mask projections
        m = fuzzy_mask(resolution, mask_radius)
        masked_projs = stack.copy()
        masked_projs = masked_projs.transpose((2, 0, 1))
        masked_projs *= m
        masked_projs = masked_projs.transpose((1, 2, 0)).copy()

        # compute polar fourier transform
        pf, _ = cryo_pft(masked_projs, n_r, AbinitioConfig.n_theta)

        # find common lines from projections
        clstack, _, _, _, _ = cryo_clmatrix_cpu(pf, num_projections, 1,
                                                max_shift, AbinitioConfig.shift_step)

        if AbinitioConfig.algo == 2:
            s = cryo_syncmatrix_vote(clstack, AbinitioConfig.n_theta)
            rotations = cryo_sync_rotations(s)
        else:
            raise NotImplementedError('AbinitioConfig.algo currently support only "2"!')

        est_shifts, _ = cryo_estimate_shifts(pf, rotations, max_shift, AbinitioConfig.shift_step)

        # reconstruct downsampled volume with no CTF correction
        n = stack.shape[1]

        params = fill_struct()
        params.rot_matrices = rotations
        params.ctf = np.ones((n, n))
        params.ctf_idx = np.array([True] * stack.shape[2])
        params.shifts = est_shifts
        params.ampl = np.ones(stack.shape[2])

        basis = DiracBasis((n, n, n))
        v1, _ = cryo_estimate_mean(stack, params, basis)
        ii1 = np.linalg.norm(v1.imag) / np.linalg.norm(v1)
        logger.info(f'Relative norm of imaginary components = {ii1}')
        v1 = v1.real
        return v1


def cryo_estimate_mean(im, params, basis=None, mean_est_opt=None):
    """
    1e-5 error from matlab
    :param im:
    :param params:
    :param basis:
    :param mean_est_opt:
    :return:
    """
    resolution = im.shape[1]
    n = im.shape[2]

    if basis is None:
        basis = DiracBasis((resolution, resolution, resolution))

    mean_est_opt = fill_struct(mean_est_opt, {'precision': 'float64', 'preconditioner': 'circulant'})

    if not check_imaging_params(params, resolution, n):
        raise ValueError('bad Params')

    kernel_f = cryo_mean_kernel_f(resolution, params, mean_est_opt)

    precond_kernel_f = []
    if mean_est_opt.preconditioner == 'circulant':
        precond_kernel_f = 1 / circularize_kernel_f(kernel_f)
    elif mean_est_opt.preconditioner != 'none':
        raise ValueError('Invalid preconditioner type')

    def identity(x):
        return x

    mean_est_opt.preconditioner = identity
    im_bp = cryo_mean_backproject(im, params, mean_est_opt)

    mean_est, cg_info = cryo_conj_grad_mean(kernel_f, im_bp, basis, precond_kernel_f, mean_est_opt)
    return mean_est, cg_info


def cryo_conj_grad_mean(kernel_f, im_bp, basis, precond_kernel_f=None, mean_est_opt=None):
    mean_est_opt = fill_struct(mean_est_opt)
    resolution = im_bp.shape[0]
    if len(im_bp.shape) != 3 or im_bp.shape[1] != resolution or im_bp.shape[2] != resolution:
        raise ValueError('im_bp must be as array of size LxLxL')

    def fun(vol_basis):
        return apply_mean_kernel(vol_basis, kernel_f, basis)

    if precond_kernel_f is not None:
        def precond_fun(vol_basis):
            return apply_mean_kernel(vol_basis, precond_kernel_f, basis)

        mean_est_opt.preconditioner = precond_fun

    im_bp_basis = basis.evaluate_t(im_bp)
    mean_est_basis, _, cg_info = conj_grad(fun, im_bp_basis, mean_est_opt)
    mean_est = basis.evaluate(mean_est_basis)
    return mean_est, cg_info


def conj_grad(a_fun, b, cg_opt=None, init=None):

    def identity(input_x):
        return input_x

    cg_opt = fill_struct(cg_opt, {'max_iter': 50, 'verbose': 0, 'iter_callback': [], 'preconditioner': identity,
                                  'rel_tolerance': 1e-15, 'store_iterates': False})
    init = fill_struct(init, {'x': None, 'p': None})
    if init.x is None:
        x = np.zeros(b.shape)
    else:
        x = init.x

    b_norm = np.linalg.norm(b)
    r = b.copy()
    s = cg_opt.preconditioner(r)

    if np.any(x != 0):
        if cg_opt.verbose:
            print('[CG] Calculating initial residual')
        a_x = a_fun(x)
        r = r-a_x
        s = cg_opt.preconditioner(r)
    else:
        a_x = np.zeros(x.shape)

    obj = np.real(np.sum(x.conj() * a_x, 0) - 2 * np.real(np.sum(np.conj(b * x), 0)))

    if init.p is None:
        p = s
    else:
        p = init.p

    info = fill_struct(att_vals={'iter': [0], 'res': [np.linalg.norm(r)], 'obj': [obj]})
    if cg_opt.store_iterates:
        info = fill_struct(info, att_vals={'x': [x], 'r': [r], 'p': [p]})

    if cg_opt.verbose:
        print('[CG] Initialized. Residual: {}. Objective: {}'.format(np.linalg.norm(info.res[0]), np.sum(info.obj[0])))

    if b_norm == 0:
        print('b_norm == 0')
        return

    i = 0
    for i in range(1, cg_opt.max_iter):
        if cg_opt.verbose:
            print('[CG] Applying matrix & preconditioner')

        a_p = a_fun(p)
        old_gamma = np.real(np.sum(s.conj() * r))

        alpha = old_gamma / np.real(np.sum(p.conj() * a_p))
        x += alpha * p
        a_x += alpha * a_p

        r -= alpha * a_p
        s = cg_opt.preconditioner(r)
        new_gamma = np.real(np.sum(r.conj() * s))
        beta = new_gamma / old_gamma
        p *= beta
        p += s

        obj = np.real(np.sum(x.conj() * a_x, 0) - 2 * np.real(np.sum(np.conj(b * x), 0)))
        res = np.linalg.norm(r)
        info.iter.append(i)
        info.res.append(res)
        info.obj.append(obj)
        if cg_opt.store_iterates:
            info.x.append(x)
            info.r.append(r)
            info.p.append(p)

        if cg_opt.verbose:
            print('[CG] Initialized. Residual: {}. Objective: {}'.format(np.linalg.norm(info.res[0]), np.sum(info.obj[0])))

        if np.all(res < b_norm * cg_opt.rel_tolerance):
            break

    # if i == cg_opt.max_iter - 1:
    #     raise Warning('Conjugate gradient reached maximum number of iterations!')
    return x, obj, info


def apply_mean_kernel(vol_basis, kernel_f, basis):
    vol = basis.evaluate(vol_basis)
    vol = cryo_conv_vol(vol, kernel_f)
    vol_basis = basis.evaluate_t(vol)
    return vol_basis


def cryo_conv_vol(x, kernel_f):
    n = x.shape[0]
    n_ker = kernel_f.shape[0]

    if np.any(np.array(x.shape) != n):
        raise ValueError('Volume in `x` must be cubic')

    if np.any(np.array(kernel_f.shape) != n_ker):
        raise ValueError('Convolution kernel in `kernel_f` must be cubic')

    is_singleton = len(x.shape) == 3

    shifted_kernel_f = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(kernel_f, 0), 1), 2)

    if is_singleton:
        x = numpy_fft.fftn(x, [n_ker] * 3)
    else:
        x = numpy_fft.fft(x, n=n_ker, axis=0)
        x = numpy_fft.fft(x, n=n_ker, axis=1)
        x = numpy_fft.fft(x, n=n_ker, axis=2)

    x *= shifted_kernel_f

    if is_singleton:
        x = numpy_fft.ifftn(x)
        x = x[:n, :n, :n]
    else:
        x = numpy_fft.ifft(x, axis=0)
        x = numpy_fft.ifft(x, axis=1)
        x = numpy_fft.ifft(x, axis=2)

    x = x.real
    return x


def cryo_mean_backproject(im, params, mean_est_opt=None):
    """
    1e-7 error from matlab
    :param im:
    :param params:
    :param mean_est_opt:
    :return:
    """
    mean_est_opt = fill_struct(mean_est_opt, {'precision': 'float64', 'half_pixel': False, 'batch_size': 0})
    if im.shape[0] != im.shape[1] or im.shape[0] == 1 or len(im.shape) != 3:
        raise ValueError('im must be 3 dimensional LxLxn where L > 1')

    resolution = im.shape[1]
    n = im.shape[2]

    check_imaging_params(params, resolution, n)
    # TODO debug, might be a problem with the first 2 lines
    if mean_est_opt.batch_size != 0:
        batch_size = mean_est_opt.batch_size
        mean_est_opt.batch_size = 0

        batch_ct = np.ceil(n / batch_size)
        im_bp = np.zeros([2 * resolution] * 3, dtype=mean_est_opt.precision)

        for batch in range(batch_ct):
            start = batch_size * batch
            end = min((batch_size + 1) * batch, n)

            batch_params = subset_params(params, np.arange(start, end))
            batch_im = im[:, :, start:end]
            batch_im_bp = cryo_mean_kernel_f(batch_im, batch_params, mean_est_opt)
            im_bp += (end - start) / n * batch_im_bp

        return im_bp

    if mean_est_opt.precision == 'float32' or mean_est_opt.precision == 'single':
        im = im.astype('float32')

    filter_f = np.einsum('ij, k -> ijk', params.ctf, np.ones(np.count_nonzero(params.ctf_idx)))
    im = im * params.ampl
    im = im_translate(im, -params.shifts)
    im = im_filter(im, filter_f)
    im = im_backproject(im, params.rot_matrices, mean_est_opt.half_pixel)
    im /= n
    return im


def im_backproject(im, rot_matrices, half_pixel=False):
    resolution = im.shape[1]
    n = im.shape[2]
    if im.shape[0] != resolution:
        raise ValueError('im must be squared - LxLxN')
    if rot_matrices.shape[2] != n:
        raise ValueError('The number of rotation matrices must match the number of images')

    pts_rot = rotated_grids(resolution, rot_matrices, half_pixel)
    pts_rot = pts_rot.reshape((3, -1), order='F')

    if resolution % 2 == 0 and half_pixel:
        grid = np.arange(-resolution / 2, resolution / 2)
        y, x = np.meshgrid(grid, grid)
        phase_shift = 2 * np.pi * (x + y) / (2 * resolution)
        im = np.einsum('ijk, ij -> ijk', im, np.exp(1j * phase_shift))

    im = im.transpose((1, 0, 2))
    im_f = cfft2(im) / resolution ** 2

    if resolution % 2 == 0:
        if half_pixel:
            grid = np.arange(-resolution / 2, resolution / 2)
            y, x = np.meshgrid(grid, grid)
            phase_shift = 2 * np.pi * (x + y) / (2 * resolution)
            phase_shift += - np.reshape(pts_rot.sum(0), (resolution, resolution, n)) / 2
            im_f = np.einsum('ijk, ij -> ijk', im_f, np.exp(1j * phase_shift))
        else:
            im_f[0] = 0
            im_f[:, 0] = 0

    im_f = im_f.flatten('F')
    vol = anufft3(im_f, pts_rot, [resolution] * 3)
    vol = vol.real
    return vol


def im_filter(im, filter_f):
    n_im = im.shape[2]
    n_filter = filter_f.shape[2]

    if n_filter != 1 and n_filter != n_im:
        raise ValueError('The number of filters must be 1 or match the number of images')
    if im.shape[:2] != filter_f.shape[:2]:
        raise ValueError('The size of the images and filters must match')

    im_f = cfft2(im)
    im_filtered_f = im_f * filter_f
    im_filtered = np.real(icfft2(im_filtered_f))
    return im_filtered


def im_translate(im, shifts):
    n_im = im.shape[2]
    n_shifts = shifts.shape[1]

    if shifts.shape[0] != 2:
        raise ValueError('Input `shifts` must be of size 2-by-n')

    if n_shifts != 1 and n_shifts != n_im:
        raise ValueError('The number of shifts must be 1 or match the number of images')

    if im.shape[0] != im.shape[1]:
        raise ValueError('Images must be square')

    resolution = im.shape[1]
    grid = np.fft.ifftshift(np.ceil(np.arange(-resolution / 2, resolution / 2)))
    om_y, om_x = np.meshgrid(grid, grid)
    phase_shifts = np.einsum('ij, k -> ijk', om_x, shifts[0]) + np.einsum('ij, k -> ijk', om_y, shifts[1])
    phase_shifts /= resolution

    mult_f = np.exp(-2 * np.pi * 1j * phase_shifts)
    im_f = np.fft.fft2(im, axes=(0, 1))
    im_translated_f = im_f * mult_f
    im_translated = np.real(np.fft.ifft2(im_translated_f, axes=(0, 1)))
    return im_translated


def circularize_kernel_f(kernel_f):
    kernel = mdim_fftshift(np.fft.ifftn(mdim_ifftshift(kernel_f)))

    for dim in range(len(kernel_f.shape)):
        kernel = circularize_kernel_1d(kernel, dim)

    kernel = mdim_fftshift(np.fft.fftn(mdim_ifftshift(kernel)))
    return kernel


def circularize_kernel_1d(kernel, dim):
    sz = kernel.shape
    if dim >= len(sz):
        raise ValueError('dim exceeds kernal dimensions')

    n = sz[dim] // 2

    mult = np.arange(n) / n
    if dim == 0:
        kernel_circ = np.einsum('i, ijk -> ijk', mult, kernel[:n])
    elif dim == 1:
        kernel_circ = np.einsum('j, ijk -> ijk', mult, kernel[:, :n])
    else:
        kernel_circ = np.einsum('k, ijk -> ijk', mult, kernel[:, :, :n])

    mult = np.arange(n, 0, -1) / n
    if dim == 0:
        kernel_circ += np.einsum('i, ijk -> ijk', mult, kernel[n:])
    elif dim == 1:
        kernel_circ += np.einsum('j, ijk -> ijk', mult, kernel[:, n:])
    else:
        kernel_circ += np.einsum('k, ijk -> ijk', mult, kernel[:, :, n:])

    kernel_circ = np.fft.fftshift(kernel_circ, dim)
    return kernel_circ


def cryo_mean_kernel_f(resolution, params, mean_est_opt=None):
    """
    8e-14 error from matlab
    :param resolution:
    :param params:
    :param mean_est_opt:
    :return:
    """
    mean_est_opt = fill_struct(mean_est_opt, {'precision': 'float64', 'half_pixel': False, 'batch_size': 0})
    n = params.rot_matrices.shape[2]

    # TODO debug, might be a problem with the first 2 lines
    if mean_est_opt.batch_size != 0:
        batch_size = mean_est_opt.batch_size
        mean_est_opt.batch_size = 0

        batch_ct = np.ceil(n / batch_size)
        mean_kernel_f = np.zeros([2 * resolution] * 3, dtype=mean_est_opt.precision)

        for batch in range(batch_ct):
            start = batch_size * batch
            end = min((batch_size + 1) * batch, n)

            batch_params = subset_params(params, np.arange(start, end))
            batch_kernel_f = cryo_mean_kernel_f(resolution, batch_params, mean_est_opt)
            mean_kernel_f += (end - start) / n * batch_kernel_f

        return mean_kernel_f

    pts_rot = rotated_grids(resolution, params.rot_matrices, mean_est_opt.half_pixel)
    tmp = np.square(np.abs(params.ctf.flatten('F')))
    filt = np.empty((tmp.shape[0], params.ampl.shape[0]), dtype=mean_est_opt.precision)
    np.outer(tmp, np.square(params.ampl), out=filt)

    if resolution % 2 == 0 and not mean_est_opt.half_pixel:
        pts_rot = pts_rot[:, 1:, 1:]
        filt = filt[1:, 1:]

    # Reshape inputs into appropriate sizes and apply adjoint NUFFT
    pts_rot = pts_rot.reshape((3, -1), order='F')
    filt = filt.flatten('F')
    mean_kernel = anufft3(filt, pts_rot, [2 * resolution] * 3)
    mean_kernel /= n * resolution ** 2

    # Ensure symmetric kernel
    mean_kernel[0] = 0
    mean_kernel[:, 0] = 0
    mean_kernel[:, :, 0] = 0

    mean_kernel = mean_kernel.copy()
    # Take the Fourier transform since this is what we want to use when convolving
    mean_kernel = np.fft.ifftshift(mean_kernel)
    mean_kernel = np.fft.fftn(mean_kernel)
    mean_kernel = np.fft.fftshift(mean_kernel)
    mean_kernel = np.real(mean_kernel)
    return mean_kernel


def rotated_grids(resolution, rot_matrices, half_pixel=False):
    mesh2d = mesh_2d(resolution)

    if resolution % 2 == 0 and half_pixel:
        mesh2d.x += 1 / resolution
        mesh2d.y += 1 / resolution

    num_pts = resolution ** 2
    num_rots = rot_matrices.shape[2]

    pts = np.pi * np.stack((mesh2d.x.flatten('F'), mesh2d.y.flatten('F'), np.zeros(num_pts)))
    pts_rot = np.einsum('ilk, lj -> ijk', rot_matrices, pts)

    pts_rot = pts_rot.reshape((3, resolution, resolution, num_rots), order='F')
    return pts_rot


def subset_params(params, ind):
    if not check_imaging_params(params):
        raise ValueError('bad Params')

    # TODO check input
    batch_params = fill_struct()

    params.rot_matrices = params.rot_matrices[:, :, ind]
    params.ctf_idx = params.ctf_idx[:, ind]
    params.ampl = params.ampl[:, ind]
    params.shifts = params.shifts[:, ind]

    return batch_params


def cryo_estimate_shifts(pf, rotations, max_shift, shift_step=1, memory_factor=10000, shifts_2d_ref=None, verbose=0):
    if memory_factor < 0 or (memory_factor > 1 and memory_factor < 100):
        raise ValueError('subsamplingfactor must be between 0 and 1 or larger than 100')

    n_theta = pf.shape[1] // 2
    n_projs = pf.shape[2]
    pf = np.concatenate((np.flip(pf[1:, n_theta:], 0), pf[:, :n_theta]), 0).copy()

    n_equations_total = int(np.ceil(n_projs * (n_projs - 1) / 2))
    memory_total = n_equations_total * 2 * n_projs * 8

    if memory_factor <= 1:
        n_equations = int(np.ceil(n_projs * (n_projs - 1) * memory_factor / 2))
    else:
        subsampling_factor = (memory_factor * 10 ** 6) / memory_total
        if subsampling_factor < 1:
            n_equations = int(np.ceil(n_projs * (n_projs - 1) * subsampling_factor / 2))
        else:
            n_equations = n_equations_total

    if n_equations < n_projs:
        Warning('Too few equations. Increase memory_factor. Setting n_equations to n_projs')
        n_equations = n_projs

    if n_equations < 2 * n_projs:
        Warning('Number of equations is small. Consider increase memory_factor.')

    shift_i = np.zeros(4 * n_equations + n_equations)
    shift_j = np.zeros(4 * n_equations + n_equations)
    shift_eq = np.zeros(4 * n_equations + n_equations)
    shift_b = np.zeros(n_equations)

    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    r_max = (pf.shape[0] - 1) // 2
    rk = np.arange(-r_max, r_max + 1)
    rk2 = rk[:r_max]
    shift_phases = np.exp(
        np.outer(-2 * np.pi * 1j * rk2 / (2 * r_max + 1), np.arange(-max_shift, -max_shift + n_shifts * shift_step)))

    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * (r_max / 4) ** 2))

    d_theta = np.pi / n_theta

    idx_i = []
    idx_j = []
    for i in range(n_projs):
        tmp_j = range(i + 1, n_projs)
        idx_i.extend([i] * len(tmp_j))
        idx_j.extend(tmp_j)
    idx_i = np.array(idx_i, dtype='int')
    idx_j = np.array(idx_j, dtype='int')
    rp = np.random.choice(np.arange(len(idx_j)), size=n_equations, replace=False)

    # might be able to vectorize this
    for shift_eq_idx in range(n_equations):
        i = idx_i[rp[shift_eq_idx]]
        j = idx_j[rp[shift_eq_idx]]

        r_i = rotations[:, :, i]
        r_j = rotations[:, :, j]
        c_ij, c_ji = common_line_r(r_i.T, r_j.T, 2 * n_theta)

        if c_ij >= n_theta:
            c_ij -= n_theta
            c_ji -= n_theta
        if c_ji < 0:
            c_ji += 2 * n_theta

        c_ij = int(c_ij)
        c_ji = int(c_ji)
        is_pf_j_flipped = 0
        if c_ji < n_theta:
            pf_j = pf[:, c_ji, j]
        else:
            pf_j = pf[:, c_ji - n_theta, j]
            is_pf_j_flipped = 1
        pf_i = pf[:, c_ij, i]

        pf_i *= h
        pf_i[r_max - 1:r_max + 2] = 0
        pf_i /= np.linalg.norm(pf_i)
        pf_i = pf_i[:r_max]

        pf_j *= h
        pf_j[r_max - 1:r_max + 2] = 0
        pf_j /= np.linalg.norm(pf_j)
        pf_j = pf_j[:r_max]

        pf_i_flipped = np.conj(pf_i)
        pf_i_stack = np.einsum('i, ij -> ij', pf_i, shift_phases)
        pf_i_flipped_stack = np.einsum('i, ij -> ij', pf_i_flipped, shift_phases)

        c1 = 2 * np.real(np.dot(np.conj(pf_i_stack.T), pf_j))
        c2 = 2 * np.real(np.dot(np.conj(pf_i_flipped_stack.T), pf_j))

        sidx1 = np.argmax(c1)
        sidx2 = np.argmax(c2)

        if c1[sidx1] > c2[sidx2]:
            dx = -max_shift + sidx1 * shift_step
        else:
            dx = -max_shift + sidx2 * shift_step

        idx = np.arange(4 * shift_eq_idx, 4 * shift_eq_idx + 4)
        shift_alpha = c_ij * d_theta
        shift_beta = c_ji * d_theta
        shift_i[idx] = shift_eq_idx
        shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        shift_b[shift_eq_idx] = dx

        # check with compare cl
        if not is_pf_j_flipped:
            shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]
        else:
            shift_beta -= np.pi
            shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]

    t = 4 * n_equations
    shift_eq[t: t + n_equations] = shift_b
    shift_i[t: t + n_equations] = np.arange(n_equations)
    shift_j[t: t + n_equations] = 2 * n_projs
    tmp = np.where(shift_eq != 0)[0]
    shift_eq = shift_eq[tmp]
    shift_i = shift_i[tmp]
    shift_j = shift_j[tmp]
    shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)), shape=(n_equations, 2 * n_projs + 1))

    est_shifts = np.linalg.lstsq(shift_equations[:, :-1].todense(), shift_b)[0]
    est_shifts = est_shifts.reshape((2, n_projs), order='F')

    if shifts_2d_ref is not None:
        raise NotImplementedError

    if verbose != 0:
        raise NotImplementedError

    return est_shifts, shift_equations


def common_line_r(r1, r2, l):
    ut = np.dot(r2, r1.T)
    alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
    alpha_ji = np.arctan2(ut[0, 2], -ut[1, 2]) + np.pi

    l_ij = alpha_ij * l / (2 * np.pi)
    l_ji = alpha_ji * l / (2 * np.pi)

    l_ij = np.mod(np.round(l_ij), l)
    l_ji = np.mod(np.round(l_ji), l)
    return l_ij, l_ji


def cryo_sync_rotations(s, rots_ref=None, verbose=0):
    """
    3e-14 err from matlab
    :param s:
    :param rots_ref:
    :param verbose:
    :return:
    """
    tol = 1e-14
    ref = 0 if rots_ref is None else 1

    sz = s.shape
    if len(sz) != 2:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] != sz[1]:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] % 2 == 1:
        raise ValueError('clmatrix must be a square matrix of size 2Kx2K')

    k = sz[0] // 2

    d, v = sps.linalg.eigs(s, 10)
    d = np.real(d)
    sort_idx = np.argsort(-d)

    if verbose:
        print('Top eigenvalues:')
        print(d[sort_idx])

    v = np.real(v[:, sort_idx[:3]])
    # v = fix_signs(np.real(v[:, sort_idx[:3]]))
    v1 = v[:2*k:2].T.copy()
    v2 = v[1:2*k:2].T.copy()

    equations = np.zeros((3*k, 9))
    for i in range(3):
        for j in range(3):
            equations[0::3, 3*i+j] = v1[i] * v1[j]
            equations[1::3, 3*i+j] = v2[i] * v2[j]
            equations[2::3, 3*i+j] = v1[i] * v2[j]
    truncated_equations = equations[:, [0, 1, 2, 4, 5, 8]]

    b = np.ones(3 * k)
    b[2::3] = 0

    ata_vec = np.linalg.lstsq(truncated_equations, b)[0]
    ata = np.zeros((3, 3))
    ata[0, 0] = ata_vec[0]
    ata[0, 1] = ata_vec[1]
    ata[0, 2] = ata_vec[2]
    ata[1, 0] = ata_vec[1]
    ata[1, 1] = ata_vec[3]
    ata[1, 2] = ata_vec[4]
    ata[2, 0] = ata_vec[2]
    ata[2, 1] = ata_vec[4]
    ata[2, 2] = ata_vec[5]

    # numpy returns lower, matlab upper
    a = np.linalg.cholesky(ata)

    r1 = np.dot(a, v1)
    r2 = np.dot(a, v2)
    r3 = np.cross(r1, r2, axis=0)

    rotations = np.empty((k, 3, 3))
    rotations[:, :, 0] = r1.T
    rotations[:, :, 1] = r2.T
    rotations[:, :, 2] = r3.T
    u, _, v = np.linalg.svd(rotations)
    np.einsum('ijk, ikl -> ijl', u, v, out=rotations)
    rotations = rotations.transpose((1, 2, 0)).copy()

    # make sure that we got rotations
    # for i in range(k):
    #     r = rotations[:, :, i]
    #     err = np.linalg.norm(np.dot(r, r.T) - np.eye(3))
    #     if err > tol:
    #         Warning('Trnaformation {} is not orthogonal'.format(i))
    #
    #     err = np.abs(np.linalg.det(r) - 1)
    #     if err > tol:
    #         Warning('Determinant of {} is not 1'.format(i))
    #
    #     u, _, v = np.linalg.svd(r)
    #
    #     rotations[:, :, i] = np.dot(u, v)
    #
    # if ref:
    #     raise NotImplementedError

    return rotations


def cryo_syncmatrix_vote(clmatrix, l, rots_ref=None, is_perturbed=0):
    """
    3e-16 error from matlab
    :param clmatrix:
    :param l:
    :param rots_ref:
    :param is_perturbed:
    :return:
    """
    sz = clmatrix.shape
    if len(sz) != 2:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] != sz[1]:
        raise ValueError('clmatrix must be a square matrix')

    k = sz[0]
    s = np.eye(2 * k)

    for i in range(k - 1):
        stmp = np.zeros((2, 2, k))
        for j in range(i + 1, k):
            stmp[:, :, j] = cryo_syncmatrix_ij_vote(clmatrix, i, j, np.arange(k), l, rots_ref, is_perturbed)

        for j in range(i + 1, k):
            r22 = stmp[:, :, j]
            s[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = r22
            s[2 * j:2 * (j + 1), 2 * i:2 * (i + 1)] = r22.T
    return s


def cryo_syncmatrix_ij_vote(clmatrix, i, j, k, l, rots_ref=None, is_perturbed=None):
    tol = 1e-12
    ref = 0 if rots_ref is None else 1

    good_k, _, _ = cryo_vote_ij(clmatrix, l, i, j, k, rots_ref, is_perturbed)

    rs, good_rotations = rotratio_eulerangle_vec(clmatrix, i, j, good_k, l)

    if rots_ref is not None:
        reflection_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        raise NotImplementedError

    if len(good_rotations) > 0:
        rk = np.mean(rs, 2)
        tmp_r = rs[:2, :2]
        diff = tmp_r - rk[:2, :2, np.newaxis]
        err = np.linalg.norm(diff) / np.linalg.norm(tmp_r)
        if err > tol:
            pass
    else:
        rk = np.zeros((3, 3))
        if rots_ref is not None:
            raise NotImplementedError

    r22 = rk[:2, :2]
    return r22


def rotratio_eulerangle_vec(cl, i, j, good_k, n_theta):
    r = np.zeros((3, 3, len(good_k)))
    if i == j:
        return 0, 0

    tol = 1e-12

    idx1 = cl[good_k, j] - cl[good_k, i]
    idx2 = cl[j, good_k] - cl[j, i]
    idx3 = cl[i, good_k] - cl[i, j]

    a = np.cos(2 * np.pi * idx1 / n_theta)
    b = np.cos(2 * np.pi * idx2 / n_theta)
    c = np.cos(2 * np.pi * idx3 / n_theta)

    cond = 1 + 2 * a * b * c - (np.square(a) + np.square(b) + np.square(c))
    too_small_idx = np.where(cond <= 1.0e-5)[0]
    good_idx = np.where(cond > 1.0e-5)[0]

    a = a[good_idx]
    b = b[good_idx]
    c = c[good_idx]
    idx2 = idx2[good_idx]
    idx3 = idx3[good_idx]
    c_alpha = (a - b * c) / np.sqrt(1 - np.square(b)) / np.sqrt(1 - np.square(c))

    ind1 = np.logical_or(idx3 > n_theta / 2 + tol, np.logical_and(idx3 < -tol, idx3 > -n_theta / 2))
    ind2 = np.logical_or(idx2 > n_theta / 2 + tol, np.logical_and(idx2 < -tol, idx2 > -n_theta / 2))
    c_alpha[np.logical_xor(ind1, ind2)] = -c_alpha[np.logical_xor(ind1, ind2)]

    aa = cl[i, j] * 2 * np.pi / n_theta
    bb = cl[j, i] * 2 * np.pi / n_theta
    alpha = np.arccos(c_alpha)

    ang1 = np.pi - bb
    ang2 = alpha
    ang3 = aa - np.pi
    sa = np.sin(ang1)
    ca = np.cos(ang1)
    sb = np.sin(ang2)
    cb = np.cos(ang2)
    sc = np.sin(ang3)
    cc = np.cos(ang3)

    r[0, 0, good_idx] = cc * ca - sc * cb * sa
    r[0, 1, good_idx] = -cc * sa - sc * cb * ca
    r[0, 2, good_idx] = sc * sb
    r[1, 0, good_idx] = sc * ca + cc * cb * sa
    r[1, 1, good_idx] = -sa * sc + cc * cb * ca
    r[1, 2, good_idx] = -cc * sb
    r[2, 0, good_idx] = sb * sa
    r[2, 1, good_idx] = sb * ca
    r[2, 2, good_idx] = cb

    if len(too_small_idx) > 0:
        r[:, :, too_small_idx] = 0

    return r, good_idx


def cryo_vote_ij(clmatrix, l, i, j, k, rots_ref, is_perturbed):
    ntics = 60
    x = np.linspace(0, 180, ntics, True)
    phis = np.zeros((len(k), 2))
    rejected = np.zeros(len(k))
    idx = 0
    rej_idx = 0
    if i != j and clmatrix[i, j] != -1:
        l_idx12 = clmatrix[i, j]
        l_idx21 = clmatrix[j, i]
        k = k[np.logical_and(np.logical_and(k != i, clmatrix[i, k] != -1), clmatrix[j, k] != -1)]

        l_idx13 = clmatrix[i, k]
        l_idx31 = clmatrix[k, i]
        l_idx23 = clmatrix[j, k]
        l_idx32 = clmatrix[k, j]

        theta1 = (l_idx13 - l_idx12) * 2 * np.pi / l
        theta2 = (l_idx21 - l_idx23) * 2 * np.pi / l
        theta3 = (l_idx32 - l_idx31) * 2 * np.pi / l

        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c3 = np.cos(theta3)

        cond = 1 + 2 * c1 * c2 * c3 - (np.square(c1) + np.square(c2) + np.square(c3))

        good_idx = np.where(cond > 1e-5)[0]
        bad_idx = np.where(cond <= 1e-5)[0]

        cos_phi2 = (c3[good_idx] - c1[good_idx] * c2[good_idx]) / (np.sin(theta1[good_idx]) * np.sin(theta2[good_idx]))
        check_idx = np.where(np.abs(cos_phi2) > 1)[0]
        if np.any(np.abs(cos_phi2) - 1 > 1e-12):
            Warning('GCAR:numericalProblem')
        elif len(check_idx) == 0:
            cos_phi2[check_idx] = np.sign(cos_phi2[check_idx])

        phis[:idx + len(good_idx), 0] = cos_phi2
        phis[:idx + len(good_idx), 1] = k[good_idx]
        idx += len(good_idx)

        rejected[: rej_idx + len(bad_idx)] = k[bad_idx]
        rej_idx += len(bad_idx)

    phis = phis[:idx]
    rejected = rejected[:rej_idx]

    good_k = []
    peakh = -1
    alpha = []

    if idx > 0:
        angles = np.arccos(phis[:, 0]) * 180 / np.pi
        sigma = 3.0

        tmp = np.add.outer(np.square(angles), np.square(x))
        h = np.sum(np.exp((2 * np.multiply.outer(angles, x) - tmp) / (2 * sigma ** 2)), 0)
        peak_idx = h.argmax()
        peakh = h[peak_idx]
        idx = np.where(np.abs(angles - x[peak_idx]) < 360 / ntics)[0]
        good_k = phis[idx, 1]
        alpha = phis[idx, 0]

        if rots_ref is not None:
            raise NotImplementedError
    return good_k.astype('int'), peakh, alpha


def cryo_clmatrix_cpu(pf, nk=None, verbose=1, max_shift=15, shift_step=1, map_filter_radius=0, ref_clmatrix=0, ref_shifts_2d=0):
    n_projs = pf.shape[2]
    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    n_theta = pf.shape[1]
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')
    n_theta = n_theta // 2

    pf = np.concatenate((np.flip(pf[1:, n_theta:], 0), pf[:, :n_theta]), 0).copy()

    found_ref_clmatrix = 0
    if not np.isscalar(ref_clmatrix):
        found_ref_clmatrix = 1

    found_ref_shifts = 0
    if not np.isscalar(ref_shifts_2d):
        found_ref_shifts = 1

    verbose_plot_shifts = 0
    verbose_detailed_debugging = 0
    verbose_progress = 0

    # Allocate variables
    clstack = np.zeros((n_projs, n_projs)) - 1
    corrstack = np.zeros((n_projs, n_projs))
    clstack_mask = np.zeros((n_projs, n_projs))

    # Allocate variables used for shift estimation
    shifts_1d = np.zeros((n_projs, n_projs))
    shift_i = np.zeros(4 * n_projs * nk)
    shift_j = np.zeros(4 * n_projs * nk)
    shift_eq = np.zeros(4 * n_projs * nk)
    shift_equations_map = np.zeros((n_projs, n_projs))
    shift_equation_idx = 0
    shift_b = np.zeros(n_projs * (n_projs - 1) // 2)
    dtheta = np.pi / n_theta

    # search for common lines between pairs of projections
    r_max = int((pf.shape[0] - 1) / 2)
    rk = np.arange(-r_max, r_max + 1)
    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * np.square(r_max / 4)))

    pf3 = np.empty(pf.shape, dtype=pf.dtype)
    np.einsum('ijk, i -> ijk', pf, h, out=pf3)
    pf3[r_max - 1:r_max + 2] = 0
    pf3 /= np.linalg.norm(pf3, axis=0)
    pf3 = pf3[:r_max]
    pf3 = pf3.transpose((2, 1, 0)).copy()

    rk2 = rk[:r_max]

    all_shift_phases = np.zeros((n_shifts, r_max), 'complex128')
    shifts = np.array([-max_shift + i * shift_step for i in range(n_shifts)], dtype='int')
    for i in range(n_shifts):
        shift = shifts[i]
        all_shift_phases[i] = np.exp(-2 * np.pi * 1j * rk2 * shift / (2 * r_max + 1))

    for i in range(n_projs):
        n2 = min(n_projs - i, nk)

        if n_projs - i - 1 == 0:
            subset_k2 = []
        else:
            subset_k2 = np.sort(np.random.choice(n_projs - i - 1, n2 - 1, replace=False) + i + 1)

        proj1 = pf3[i]
        p1 = proj1
        p1_real = np.real(p1)
        p1_imag = np.imag(p1)

        for j in subset_k2:
            proj2 = pf3[j]
            p2_flipped = np.conj(proj2)

            # second ver for now this is slower
            # tic = time.time()
            # all_p2_shifted_flipped = np.einsum('ij, kj -> kji', p2_flipped, all_shift_phases)
            # all_part1 = np.einsum('ij, kjl -> kil', p1_real, np.real(all_p2_shifted_flipped))
            # all_part2 = np.einsum('ij, kjl -> kil', p1_imag, np.imag(all_p2_shifted_flipped))
            #
            # all_c1 = all_part1 - all_part2
            # sidx = all_c1.argmax()
            # shift, cl1, cl2 = np.unravel_index(sidx, all_c1.shape)
            # sval = all_c1[shift, cl1, cl2]
            #
            # all_c2 = all_part1 + all_part2
            # sidx = all_c2.argmax()
            # shift_2, cl1_2, cl2_2 = np.unravel_index(sidx, all_c2.shape)
            # sval2 = all_c2[shift_2, cl1_2, cl2_2]
            #
            # if sval2 > sval:
            #     shift = shift_2
            #     cl1 = cl1_2
            #     cl2 = cl2_2 + n_theta
            #     sval = 2 * sval2
            # else:
            #     sval *= 2
            #
            # clstack[i, j] = cl1
            # clstack[j, i] = cl2
            # corrstack[i, j] = sval
            # shifts_1d[i, j] = shifts[shift]
            # toc = time.time()
            # print(i, j, shift_equation_idx, corrstack[i, j], shifts_1d[i, j], toc - tic)

            tic = time.time()
            for shift in range(n_shifts):
                shift_phases = all_shift_phases[shift]
                p2_shifted_flipped = (shift_phases * p2_flipped).T
                part1 = p1_real.dot(np.real(p2_shifted_flipped))
                part2 = p1_imag.dot(np.imag(p2_shifted_flipped))

                c1 = part1 - part2
                sidx = c1.argmax()
                cl1, cl2 = np.unravel_index(sidx, c1.shape)
                sval = c1[cl1, cl2]

                c2 = part1 + part2
                sidx = c2.argmax()
                cl1_2, cl2_2 = np.unravel_index(sidx, c2.shape)
                sval2 = c2[cl1_2, cl2_2]

                if sval2 > sval:
                    cl1 = cl1_2
                    cl2 = cl2_2 + n_theta
                    sval = 2 * sval2
                else:
                    sval *= 2

                if sval > corrstack[i, j]:
                    clstack[i, j] = cl1
                    clstack[j, i] = cl2
                    corrstack[i, j] = sval
                    shifts_1d[i, j] = shifts[shift]

            toc = time.time()
            # Create a shift equation for the projections pair (i, j).
            idx = np.arange(4 * shift_equation_idx, 4 * shift_equation_idx + 4)
            shift_alpha = clstack[i, j] * dtheta
            shift_beta = clstack[j, i] * dtheta
            shift_i[idx] = shift_equation_idx
            shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            shift_b[shift_equation_idx] = shifts_1d[i, j]

            # Compute the coefficients of the current equation.
            if shift_beta < np.pi:
                shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]
            else:
                shift_beta -= np.pi
                shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]

            shift_equations_map[i, j] = shift_equation_idx
            print(i, j, shift_equation_idx, toc - tic)
            shift_equation_idx += 1


    tmp = np.where(corrstack != 0)
    corrstack[tmp] = 1 - corrstack[tmp]
    l = 4 * shift_equation_idx
    shift_eq[l: l + shift_equation_idx] = shift_b
    shift_i[l: l + shift_equation_idx] = np.arange(shift_equation_idx)
    shift_j[l: l + shift_equation_idx] = 2 * n_projs
    tmp = np.where(shift_eq != 0)[0]
    shift_eq = shift_eq[tmp]
    shift_i = shift_i[tmp]
    shift_j = shift_j[tmp]
    l += shift_equation_idx
    shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)), shape=(shift_equation_idx, 2 * n_projs + 1))

    return clstack, corrstack, shift_equations, shift_equations_map, clstack_mask


def cryo_clmatrix_cpu_ref(pf, nk=None, verbose=1, max_shift=15, shift_step=1, map_filter_radius=0, ref_clmatrix=0, ref_shifts_2d=0):
    n_projs = pf.shape[2]
    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    n_theta = pf.shape[1]
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')
    n_theta = n_theta // 2

    pf = np.concatenate((np.flip(pf[1:, n_theta:], 0), pf[:, :n_theta]), 0).copy()

    found_ref_clmatrix = 0
    if not np.isscalar(ref_clmatrix):
        found_ref_clmatrix = 1

    found_ref_shifts = 0
    if not np.isscalar(ref_shifts_2d):
        found_ref_shifts = 1

    verbose_plot_shifts = 0
    verbose_detailed_debugging = 0
    verbose_progress = 0

    # Allocate variables
    clstack = np.zeros((n_projs, n_projs)) - 1
    corrstack = np.zeros((n_projs, n_projs))
    clstack_mask = np.zeros((n_projs, n_projs))
    refcorr = np.zeros((n_projs, n_projs))
    thetha_diff = np.zeros((n_projs, n_projs))

    # Allocate variables used for shift estimation
    shifts_1d = np.zeros((n_projs, n_projs))
    ref_shifts_1d = np.zeros((n_projs, n_projs))
    shifts_estimation_error = np.zeros((n_projs, n_projs))
    shift_i = np.zeros(4 * n_projs * nk)
    shift_j = np.zeros(4 * n_projs * nk)
    shift_eq = np.zeros(4 * n_projs * nk)
    shift_equations_map = np.zeros((n_projs, n_projs))
    shift_equation_idx = 0
    shift_b = np.zeros(n_projs * (n_projs - 1) // 2)
    dtheta = np.pi / n_theta

    # Debugging handles and variables - not implemented
    pass

    # search for common lines between pairs of projections
    r_max = int((pf.shape[0] - 1) / 2)
    rk = np.arange(-r_max, r_max + 1)
    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * np.square(r_max / 4)))

    pf3 = np.empty(pf.shape, dtype=pf.dtype)
    np.einsum('ijk, i -> ijk', pf, h, out=pf3)
    pf3[r_max - 1:r_max + 2] = 0
    pf3 /= np.linalg.norm(pf3, axis=0)

    rk2 = rk[:r_max]
    for i in range(n_projs):
        n2 = min(n_projs - i, nk)

        if n_projs - i - 1 == 0:
            subset_k2 = []
        else:
            subset_k2 = np.sort(np.random.choice(n_projs - i - 1, n2 - 1, replace=False) + i + 1)

        proj1 = pf3[:, :, i]
        p1 = proj1[:r_max].T
        p1_flipped = np.conj(p1)

        if np.linalg.norm(proj1[r_max]) > 1e-13:
            raise ValueError('DC component of projection is not zero.')

        for j in subset_k2:
            proj2 = pf3[:, :, j]
            p2 = proj2[:r_max]

            if np.linalg.norm(proj2[r_max]) > 1e-13:
                raise ValueError('DC component of projection is not zero.')

            if verbose_plot_shifts and found_ref_clmatrix:
                raise NotImplementedError

            tic = time.time()
            for shift in range(-max_shift, n_shifts, shift_step):
                shift_phases = np.exp(-2 * np.pi * 1j * rk2 * shift / (2 * r_max + 1))
                p1_shifted = shift_phases * p1
                p1_shifted_flipped = shift_phases * p1_flipped
                c1 = 2 * np.real(np.dot(p1_shifted.conj(), p2))
                c2 = 2 * np.real(np.dot(p1_shifted_flipped.conj(), p2))
                c = np.concatenate((c1, c2), 1)

                if map_filter_radius > 0:
                    raise NotImplementedError
                    # c = cryo_average_clmap(c, map_filter_radius)

                sidx = c.argmax()
                cl1, cl2 = np.unravel_index(sidx, c.shape)
                sval = c[cl1, cl2]
                improved_correlation = 0

                if sval > corrstack[i, j]:
                    clstack[i, j] = cl1
                    clstack[j, i] = cl2
                    corrstack[i, j] = sval
                    shifts_1d[i, j] = shift
                    improved_correlation = 1

                if verbose_detailed_debugging and found_ref_clmatrix and found_ref_shifts:
                    raise NotImplementedError

                if verbose_plot_shifts and improved_correlation:
                    raise NotImplementedError

                if verbose_detailed_debugging:
                    raise NotImplementedError

                if verbose_detailed_debugging:
                    raise NotImplementedError

            toc = time.time()
            # Create a shift equation for the projections pair (i, j).
            idx = np.arange(4 * shift_equation_idx, 4 * shift_equation_idx + 4)
            shift_alpha = clstack[i, j] * dtheta
            shift_beta = clstack[j, i] * dtheta
            shift_i[idx] = shift_equation_idx
            shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            shift_b[shift_equation_idx] = shifts_1d[i, j]

            # Compute the coefficients of the current equation.
            if shift_beta < np.pi:
                shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]
            else:
                shift_beta -= np.pi
                shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]

            shift_equations_map[i, j] = shift_equation_idx
            print(i, j, shift_equation_idx, corrstack[i, j], toc - tic)
            shift_equation_idx += 1

            if verbose_progress:
                raise NotImplementedError

    if verbose_detailed_debugging and found_ref_clmatrix:
        raise NotImplementedError

    tmp = np.where(corrstack != 0)
    corrstack[tmp] = 1 - corrstack[tmp]
    l = 4 * shift_equation_idx
    shift_eq[l: l + shift_equation_idx] = shift_b
    shift_i[l: l + shift_equation_idx] = np.arange(shift_equation_idx)
    shift_j[l: l + shift_equation_idx] = 2 * n_projs
    tmp = np.where(shift_eq != 0)[0]
    shift_eq = shift_eq[tmp]
    shift_i = shift_i[tmp]
    shift_j = shift_j[tmp]
    l += shift_equation_idx
    shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)), shape=(shift_equation_idx, 2 * n_projs + 1))

    if verbose_detailed_debugging:
        raise NotImplementedError

    return clstack, corrstack, shift_equations, shift_equations_map, clstack_mask


def cryo_ray_normalize(pf):
    n_theta = pf.shape[1]

    def normalize_one(p):
        for j in range(n_theta):
            nr = np.linalg.norm(p[:, j])
            if nr < 1e-3:
                Warning('Ray norm is close to zero.')
            p[:, j] /= nr
        return p

    if len(pf.shape) == 2:
        pf = normalize_one(pf)
    else:
        for k in range(pf.shape[2]):
            pf[:, :, k] = normalize_one(pf[:, :, k])
    return pf


# utils
def check_imaging_params(params, resolution=None, n=None):
    # TODO
    return True


def cryo_pft(p, n_r, n_theta):
    """
    Compute the polar Fourier transform of projections with resolution n_r in the radial direction
    and resolution n_theta in the angular direction.
    :param p:
    :param n_r: Number of samples along each ray (in the radial direction).
    :param n_theta: Angular resolution. Number of Fourier rays computed for each projection.
    :return:
    """
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')

    n_projs = p.shape[2]
    omega0 = 2 * np.pi / (2 * n_r - 1)
    dtheta = 2 * np.pi / n_theta

    freqs = np.zeros((2, n_r * n_theta // 2))
    for i in range(n_theta // 2):
        freqs[0, i * n_r: (i + 1) * n_r] = np.arange(n_r) * np.sin(i * dtheta)
        freqs[1, i * n_r: (i + 1) * n_r] = np.arange(n_r) * np.cos(i * dtheta)

    freqs *= omega0
    # finufftpy require it to be aligned in fortran order
    pf = np.empty((n_r * n_theta // 2, n_projs), dtype='complex128', order='F')
    finufftpy.nufft2d2many(freqs[0], freqs[1], pf, 1, 1e-15, p)
    pf = pf.reshape((n_r, n_theta // 2, n_projs), order='F')
    pf = np.concatenate((pf, pf.conj()), axis=1).copy()
    return pf, freqs


def fuzzy_mask(n, r0, origin=None):
    if isinstance(n, int):
        n = np.array([n])

    if isinstance(r0, int):
        r0 = np.array([r0])

    k = 1.782 / AbinitioConfig.rise_time  # move constant to config

    if AbinitioConfig.fuzzy_mask_dims == 2:
        if origin is None:
            origin = np.floor(n / 2) + 1
            origin = origin.astype('int')
        if len(n) == 1:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1]

        if len(r0) < 2:
            r = np.sqrt(np.square(x) + np.square(y))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]))

    else:
        raise WrongInput(f"only 2D is allowed!")

    m = 0.5 * (1 - sp.erf(k * (r - r0[0])))
    return m


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


def fill_struct(obj=None, att_vals=None, overwrite=None):
    """
    Fill object with attributes in a dictionary.
    If a struct is not given a new object will be created and filled.
    If the given struct has a field in att_vals, the original field will stay, unless specified otherwise in overwrite.
    att_vals is a dictionary with string keys, and for each key:
    if hasattr(s, key) and key in overwrite:
        pass
    else:
        setattr(s, key, att_vals[key])
    :param obj:
    :param att_vals:
    :param overwrite
    :return:
    """
    # TODO should consider making copy option - i.e that the input won't change
    if obj is None:
        class DisposableObject:
            pass

        obj = DisposableObject()

    if att_vals is None:
        return obj

    if overwrite is None or not overwrite:
        overwrite = []
    if overwrite is True:
        overwrite = list(att_vals.keys())

    for key in att_vals.keys():
        if hasattr(obj, key) and key not in overwrite:
            continue
        else:
            setattr(obj, key, att_vals[key])

    return obj


def mesh_2d(resolution, inclusive=False):
    if inclusive:
        cons = (resolution - 1) / 2
        grid = np.arange(-cons, cons + 1) / cons
    else:
        cons = resolution / 2
        grid = np.ceil(np.arange(-cons, cons)) / cons

    mesh = fill_struct()
    mesh.y, mesh.x = np.meshgrid(grid, grid)  # reversed from matlab
    mesh.phi, mesh.r, _ = cart2pol(mesh.x, mesh.y)
    return mesh


def cart2pol(x, y, z=None):
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    return th, r, z


def anufft3(vol_f, fourier_pts, sz):
    if len(sz) != 3:
        raise ValueError('sz must be 3')
    if len(fourier_pts.shape) != 2:
        raise ValueError('fourier_pts must be 2D with shape 3x_')
    if fourier_pts.shape[0] != 3:
        raise ValueError('fourier_pts must be 2D with shape 3x_')
    if not fourier_pts.flags.c_contiguous:
        fourier_pts = fourier_pts.copy()
    if not vol_f.flags.c_contiguous:
        vol_f = vol_f.copy()

    x = fourier_pts[0]
    y = fourier_pts[1]
    z = fourier_pts[2]
    isign = 1
    eps = 1e-15
    ms, mt, mu = sz
    f = np.empty(sz, dtype='complex128', order='F')
    finufftpy.nufft3d1(x, y, z, vol_f, isign, eps, ms, mt, mu, f)
    return f.copy()


def cfft2(x):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, (0, 1))
        y = np.transpose(y, (2, 0, 1))
        y = np.fft.fft2(y)
        y = np.transpose(y, (1, 2, 0))
        y = np.fft.fftshift(y, (0, 1))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft2(x):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, (0, 1))
        y = np.transpose(y, (2, 0, 1))
        y = np.fft.ifft2(y)
        y = np.transpose(y, (1, 2, 0))
        y = np.fft.fftshift(y, (0, 1))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def mdim_ifftshift(x, dims=None):
    if dims is None:
        dims = np.arange(len(x.shape))

    x = np.fft.ifftshift(x, dims)
    return x


def mdim_fftshift(x, dims=None):
    if dims is None:
        dims = np.arange(len(x.shape))

    x = np.fft.fftshift(x, dims)
    return x


def comp(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)
