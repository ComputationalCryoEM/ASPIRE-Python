import numpy as np
import Cn.utils as utils
import aspire.abinitio as abinitio
from Cn.config_symm import AbinitioSymmConfig

from Cn.estimate_relative_viewing_directions import estimate_relative_viewing_directions
from Cn.estimate_third_rows import estimate_third_rows
from Cn.estimate_rots_from_third_rows import estimate_rots_from_third_rows
from Cn.handedness_sync import handedness_sync


def main():
    dict_mat_file_names = {
        'projections': ['projs', 'projs_shifted'],
        'symmetry': ['n_symm', 'n_symm_shifted'],
        'rotation matrices gt': ['rots_gt', 'rots_gt_shifted']
    }

    sim_type = 1 if AbinitioSymmConfig.is_load_shifted_projs else 0
    projs = utils.mat_to_npy(dict_mat_file_names['projections'][sim_type])
    AbinitioSymmConfig.n_symm = utils.mat_to_npy_vec(dict_mat_file_names['symmetry'][sim_type])[0]
    rots_gt = utils.mat_to_npy(dict_mat_file_names['rotation matrices gt'][sim_type])
    rots_gt = np.transpose(rots_gt, axes=(2, 0, 1))

    if AbinitioSymmConfig.is_load_shifted_projs:
        AbinitioSymmConfig.max_shift = utils.mat_to_npy_vec('max_shift')[0]
        AbinitioSymmConfig.shift_step = utils.mat_to_npy_vec('shift_step')[0]
    else:
        AbinitioSymmConfig.max_shift = 0
        AbinitioSymmConfig.shift_step = 1

    n_r = AbinitioSymmConfig.n_r
    n_theta = AbinitioSymmConfig.n_theta
    n_symm = AbinitioSymmConfig.n_symm
    cache_file_name = AbinitioSymmConfig.cache_file_name
    npf, _ = abinitio.cryo_pft(projs, n_r, n_theta)
    npf = np.transpose(npf, axes=(2, 1, 0))
    viis, vijs = estimate_relative_viewing_directions(npf, cache_file_name, rots_gt)
    viis, vijs, sign_J = handedness_sync(viis, vijs)
    vis = estimate_third_rows(vijs, viis, rots_gt)
    rots, _ = estimate_rots_from_third_rows(npf, vis)
    mse, rots_aligned, _ = utils.check_rotations_error(rots, n_symm, rots_gt)
    err_in_degrees = utils.check_degrees_error(rots_aligned, n_symm, n_theta, rots_gt)

    print('median error in degrees: %e' % np.median(err_in_degrees))
    print('mean error in degrees: %e' % np.mean(err_in_degrees))
    print('std error in degrees: %e' % np.std(err_in_degrees))
    print('min error in degrees: %e' % np.min(err_in_degrees))
    print('max error in degrees: %e' % np.max(err_in_degrees))
    print("mse=" + str(mse))


if __name__ == "__main__":
    main()
