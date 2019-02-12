from Cn.abinitio_cn import abinitio_c2, abinitio_cn
import Cn.utils as utils
import numpy as np

# This script illustrates the basic ab initio reconstruction functionality of
# the ASPIRE toolbox on Cn symmetric simulated data.

print('loading mat files')
projs = utils.mat_to_npy('projs')
rots_gt = utils.mat_to_npy('rots_gt')
n_symm = utils.mat_to_npy_vec('n_symm')[0]
max_shift = utils.mat_to_npy_vec('max_shift')[0]
shift_step = utils.mat_to_npy_vec('shift_step')[0]
inplane_rot_res_deg = utils.mat_to_npy_vec('inplane_rot_res')[0]
n_r = 45
n_theta = 360

projs = np.transpose(projs, axes=(2, 0, 1))
rots_gt = np.transpose(rots_gt, axes=(2, 0, 1))
if n_symm == 2:
    rots, vol = abinitio_c2(projs, n_r, n_theta, max_shift, shift_step, rots_gt)
elif n_symm in [3, 4]:
    # cache file is not needed so supply None
    rots = abinitio_cn(n_symm, projs, n_r, n_theta, max_shift, shift_step, inplane_rot_res_deg, None, rots_gt)
else:
    # cache_file_name = 'cn_cache_points200_ntheta360_res1_tmp_gt.pckl'
    # cache_file_name = 'cn_cache_points500_ntheta360_res1_gt.pckl'  # assign None if wish to create a new one
    cache_file_name = 'cn_cache_points1000_ntheta360_res1.pckl'  # assign None if wish to create a new one
    rots, vol = abinitio_cn(n_symm, projs, n_r, n_theta, max_shift, shift_step, inplane_rot_res_deg, cache_file_name, rots_gt)

