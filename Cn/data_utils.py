import Cn.utils as utils
import numpy as np
import aspire.abinitio as abinitio
import Cn.utils as utils
from Cn.config_symm import AbinitioSymmConfig


if __name__ == "__main__":
    
    projs = utils.mat_to_npy('projs')
    rots_gt = utils.mat_to_npy('rots_gt')
    # projs = np.transpose(projs, axes=(2, 0, 1))
    rots_gt = np.transpose(rots_gt, axes=(2, 0, 1))
    # plt.imshow(projs[0], cmap='gray')
    # plt.show()

    n_r = 45
    n_theta = 360
    max_shift = 0
    shift_step = 1

    n_symm = 4
    angle_tol_err_deg = 5

    n_images = len(projs)
    pf, _ = abinitio.cryo_pft(projs, n_r, n_theta)
    # find common lines from projections
    clmatrix, _, _, _, _ = abinitio.cryo_clmatrix_cpu(pf, n_images, 1, max_shift, shift_step)
    utils.cl_detection_rate_single(n_symm, clmatrix, rots_gt, n_theta, AbinitioSymmConfig.angle_tol_err_deg)

    npf = np.transpose(pf, axes=(2, 1, 0))
    clmatrix2, _, _, _, _ = abinitio.cryo_clmatrix_cpu_pystyle(npf, n_images, 0, max_shift, shift_step)
    utils.cl_detection_rate_single(n_symm, clmatrix2, rots_gt, n_theta, AbinitioSymmConfig.angle_tol_err_deg)

    Rijs = abinitio.cryo_syncmatrix_vote_3n(clmatrix, n_theta)
    utils.detection_rate_relative_rots(Rijs, n_symm, rots_gt)

