import numpy as np
import pyfftw
from pynfft import NFFT
# from pynufft.pynufft import NUFFT_cpu

# import sys
# sys.path.insert(0, '/home/itays/Desktop/aspire-python')
# sys.path.insert(0, '/home/itays/Desktop/aspire-python/lib')
# sys.path.insert(0, '/home/itays/Desktop/aspire-python/extern')
# from lib.nufft_cims import py_nufft


def forward(images, fast_model, start, finish):
    nfft_res = compute_nfft_potts(images, fast_model, start, finish)
    return pswf_integration(nfft_res, fast_model)


def pswf_integration(images_nufft, fast_model):
    # unpacking needed variables
    n_max = fast_model.n_max
    r_quad_indices = fast_model.r_quad_indices
    num_angular_pts = fast_model.num_angular_pts
    indices_for_n = fast_model.indices_for_n
    numel_for_n = fast_model.numel_for_n
    blk_r = fast_model.blk_r

    num_images = images_nufft.shape[1]

    n_max_float = float(n_max) / 2

    quad_rule_radial_wts = fast_model.quad_rule_radial_wts

    r_n_eval_mat = np.zeros((len(fast_model.radial_quad_pts), n_max, num_images), dtype='complex128')

    for i in range(len(fast_model.radial_quad_pts)):
        curr_r_mat = images_nufft[r_quad_indices[i]: r_quad_indices[i] + num_angular_pts[i], :]
        curr_r_mat = np.concatenate((curr_r_mat, np.conj(curr_r_mat)))
        fft_plan = pyfftw.builders.fft(curr_r_mat, axis=0, overwrite_input=True, auto_contiguous=True,
                                       auto_align_input=False, avoid_copy=True, planner_effort='FFTW_ESTIMATE')
        angular_eval = fft_plan() * quad_rule_radial_wts[i]

        r_n_eval_mat[i, :, :] = np.tile(angular_eval, (int(max(1, np.ceil(n_max_float / num_angular_pts[i]))), 1))[:n_max, :]

    r_n_eval_mat = r_n_eval_mat.reshape((len(fast_model.radial_quad_pts) * n_max, num_images), order='F')
    coeff_vec_quad = np.zeros((len(fast_model.angular_frequency), num_images), dtype='complex128')
    m = len(fast_model.pswf_radial_quad)
    for i in range(n_max):
        coeff_vec_quad[indices_for_n[i] + np.arange(numel_for_n[i]), :] =\
            np.dot(blk_r[i], r_n_eval_mat[i * m:(i + 1)*m, :])

    return coeff_vec_quad


def compute_nfft_potts(images, fast_model, start, finish):
    x = fast_model.us_fft_pts
    n = fast_model.size_x
    points_inside_circle = fast_model.points_inside_circle
    num_images = finish - start

    # pynufft
    # m = x.shape[0]
    # nufft_obj = NUFFT_cpu()
    # nufft_obj.plan(x, (n, n), (2*n, 2*n), (10, 10))
    # shift = np.exp(x * fast_model.resolution * 1j)
    # shift = np.sum(shift, axis=1)

    # gal nufft
    # m = x.shape[1]
    # nufft_obj = py_nufft.factory('nufft')

    # pynfft
    m = x.shape[0]
    plan = NFFT(N=[n, n], M=m)
    plan.x = x
    plan.precompute()

    images_nufft = np.zeros((m, num_images), dtype='complex128')
    current_image = np.zeros((n, n))
    for i in range(start, finish):

        current_image[points_inside_circle] = images[:, i]

        # images_nufft[:, i - start] = nufft_obj.forward(current_image) * shift

        plan.f_hat = current_image
        images_nufft[:, i - start] = plan.trafo()

        # images_nufft[:, i - start] = nufft_obj.forward2d(current_image.T, x, iflag=-1)[0]

    return images_nufft
