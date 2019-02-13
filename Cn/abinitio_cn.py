import numpy as np
import Cn.utils as utils
import aspire.abinitio as abinitio
from Cn.estimate_relative_viewing_directions import estimate_relative_viewing_directions, estimate_relative_rotations_c2
from Cn.estimate_third_rows import estimate_third_rows
from Cn.estimate_rots_from_third_rows import estimate_rots_from_third_rows, estimate_rots_from_third_rows_c2
from Cn.handedness_sync import handedness_sync
from Cn.config_symm import AbinitioSymmConfig
from Cn.reconstruct_cn import reconstruct_cn


def abinitio_c2(projs, n_r=45, n_theta=360, max_shift=15, shift_step=1, rots_gt=None):
    AbinitioSymmConfig.n_symm = 2
    AbinitioSymmConfig.n_r = n_r
    AbinitioSymmConfig.n_theta = n_theta
    AbinitioSymmConfig.max_shift = max_shift
    AbinitioSymmConfig.shift_step = shift_step
    print('running nufft on projections')
    npf, _ = abinitio.cryo_pft_pystyle(projs, n_r, n_theta)
    print('running abinitio c2')
    Rijs = estimate_relative_rotations_c2(npf, rots_gt)
    vijs = np.mean(Rijs, axis=1)
    viis = np.zeros((len(npf), 3, 3))
    _, _, signs_J = handedness_sync(viis, vijs)
    for ij, sign_j in enumerate(signs_J):
        if sign_j == 1:
            Rijs[ij, 0] = utils.J_conjugate(Rijs[ij, 0])
            Rijs[ij, 1] = utils.J_conjugate(Rijs[ij, 1])
    vijs = np.mean(Rijs, axis=1)
    vis = estimate_third_rows(vijs, viis, rots_gt)
    rots, _ = estimate_rots_from_third_rows_c2(vis, Rijs)
    vol = reconstruct_cn(projs, rots)
    if rots_gt is not None:
        mse, rots_aligned, _ = utils.check_rotations_error(rots, AbinitioSymmConfig.n_symm, rots_gt)
        err_in_degrees = utils.check_degrees_error(rots_aligned, AbinitioSymmConfig.n_symm, n_theta, rots_gt)
        print('median error in degrees: %e' % np.median(err_in_degrees))
        print('mean error in degrees: %e' % np.mean(err_in_degrees))
        print('std error in degrees: %e' % np.std(err_in_degrees))
        print('min error in degrees: %e' % np.min(err_in_degrees))
        print('max error in degrees: %e' % np.max(err_in_degrees))
        print("mse=" + str(mse))
    return rots, vol


def abinitio_cn(n_symm, projs, n_r=45, n_theta=360, max_shift=15, shift_step=1, inplane_rot_res_deg=1, cache_file_name=None, rots_gt=None):
    AbinitioSymmConfig.n_symm = n_symm
    AbinitioSymmConfig.n_r = n_r
    AbinitioSymmConfig.n_theta = n_theta
    AbinitioSymmConfig.max_shift = max_shift
    AbinitioSymmConfig.shift_step = shift_step
    AbinitioSymmConfig.inplane_rot_res_deg = inplane_rot_res_deg
    AbinitioSymmConfig.cache_file_name = cache_file_name

    print('running nufft on projections')
    npf, _ = abinitio.cryo_pft_pystyle(projs, n_r, n_theta)
    print('running abinitio c' + str(n_symm))
    viis, vijs = estimate_relative_viewing_directions(npf, cache_file_name, rots_gt)
    viis, vijs, sign_J = handedness_sync(viis, vijs)
    vis = estimate_third_rows(vijs, viis, rots_gt)
    rots, _ = estimate_rots_from_third_rows(npf, vis)

    vol = reconstruct_cn(projs, rots)
    if rots_gt is not None:
        mse, rots_aligned, _ = utils.check_rotations_error(rots, n_symm, rots_gt)
        err_in_degrees = utils.check_degrees_error(rots_aligned, n_symm, n_theta, rots_gt)
        print('median error in degrees: %e' % np.median(err_in_degrees))
        print('mean error in degrees: %e' % np.mean(err_in_degrees))
        print('std error in degrees: %e' % np.std(err_in_degrees))
        print('min error in degrees: %e' % np.min(err_in_degrees))
        print('max error in degrees: %e' % np.max(err_in_degrees))
        print("mse=" + str(mse))
    return rots, vol


if __name__ == "__main__":
    averages_nn50_group1 = utils.mat_to_npy('averages_nn50_group1')
    averages_nn50_group1 = np.transpose(averages_nn50_group1, axes=(2, 0, 1))
    n_symm = 4
    abinitio_cn(n_symm, averages_nn50_group1)