import numpy as np
from Cn.abinitio_cn import abinitio_c2, abinitio_cn
from scipy.io import loadmat

# This script illustrates the basic ab initio reconstruction functionality of
# the ASPIRE toolbox on Cn symmetric simulated data.


def mat_to_npy(file_name):
    return loadmat(file_name + '.mat')[file_name]


def mat_to_npy_vec(file_name):
    a = mat_to_npy(file_name)
    return a.reshape(a.shape[0] * a.shape[1])


def main():
    print('loading mat files')
    projs = mat_to_npy('projs')
    rots_gt = mat_to_npy('rots_gt')
    n_symm = mat_to_npy_vec('n_symm')[0]
    max_shift = mat_to_npy_vec('max_shift')[0]
    shift_step = mat_to_npy_vec('shift_step')[0]
    inplane_rot_res_deg = mat_to_npy_vec('inplane_rot_res')[0]
    n_r = 45
    n_theta = 360
    rots_gt = np.transpose(rots_gt, axes=(2, 0, 1))
    if n_symm == 2:
        rots = abinitio_c2(projs, n_r, n_theta, max_shift, shift_step, rots_gt)
    elif n_symm in [3, 4]:
        # cache file is not needed so supply None
        rots = abinitio_cn(n_symm, projs, n_r, n_theta, max_shift, shift_step, inplane_rot_res_deg, None, rots_gt)
    else:
        # cache_file_name = 'cn_cache_points200_ntheta360_res1_tmp_gt.pckl'
        # cache_file_name = 'cn_cache_points500_ntheta360_res1_gt.pckl'  # assign None if wish to create a new one
        cache_file_name = 'cn_cache_points1000_ntheta360_res1.pckl'  # assign None if wish to create a new one
        rots = abinitio_cn(n_symm, projs, n_r, n_theta, max_shift, shift_step, inplane_rot_res_deg, cache_file_name, rots_gt)
    return rots


if __name__ == "__main__":
    main()
