import numpy as np
import Cn.utils as utils
import aspire.abinitio as abinitio
from Cn.config_symm import AbinitioSymmConfig

from Cn.estimate_third_rows import estimate_third_rows
from Cn.estimate_rots_from_third_rows import estimate_rots_from_third_rows
from Cn.estimate_relative_viewing_directions_c3_c4 import estimate_relative_viewing_directions_c3_c4
from Cn.estimate_relative_viewing_directions_cn import estimate_relative_viewing_directions_cn
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

    npf, _ = abinitio.cryo_pft(projs, AbinitioSymmConfig.n_r, AbinitioSymmConfig.n_theta)
    npf = np.transpose(npf, axes=(2, 1, 0))
    # if AbinitioSymmConfig.n_symm in [3, 4]:
    if AbinitioSymmConfig.n_symm == 3:
        viis, vijs = estimate_relative_viewing_directions_c3_c4(AbinitioSymmConfig.n_symm, npf, AbinitioSymmConfig.n_theta, rots_gt)
    else:
        print('Estimating relative viewing directions for n>4')
        viis, vijs = estimate_relative_viewing_directions_cn(AbinitioSymmConfig.n_symm, npf, rots_gt)

    viis, vijs, sign_J = handedness_sync(viis, vijs, rots_gt)
    vis = estimate_third_rows(vijs, viis, rots_gt)
    rots = estimate_rots_from_third_rows(AbinitioSymmConfig.n_symm, npf, vis, rots_gt)
    mse, rots_aligned, _ = utils.check_rotations_error(rots, AbinitioSymmConfig.n_symm, rots_gt)
    err_in_degrees = utils.check_degrees_error(rots_aligned, AbinitioSymmConfig.n_symm, AbinitioSymmConfig.n_theta, rots_gt)

    print("mse=" + str(mse))


if __name__ == "__main__":
    main()

