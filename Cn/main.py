import numpy as np
import Cn.utils as utils
import aspire.abinitio as abinitio
from Cn.config_symm import AbinitioSymmConfig

from Cn.estimate_third_rows import estimate_third_rows
from Cn.estimate_rots_from_third_rows import estimate_rots_from_third_rows
from Cn.estimate_relative_viewing_directions_c3_c4 import estimate_relative_viewing_directions_c3_c4
from Cn.handedness_sync import handedness_sync


def main():

    projs = utils.mat_to_npy('projs')
    AbinitioSymmConfig.n_symm = 4  # TODO: this should come from matlab. not hard-coded
    rots_gt = utils.mat_to_npy('rots_gt')
    rots_gt = np.transpose(rots_gt, axes=(2, 0, 1))

    # plt.imshow(projs[0], cmap='gray')
    # plt.show()
    npf, _ = abinitio.cryo_pft(projs, AbinitioSymmConfig.n_r, AbinitioSymmConfig.n_theta)
    npf = np.transpose(npf, axes=(2, 1, 0))
    if AbinitioSymmConfig.n_symm in [3, 4]:
        viis, vijs = estimate_relative_viewing_directions_c3_c4(AbinitioSymmConfig.n_symm, npf, AbinitioSymmConfig.n_theta, rots_gt)
    else:
        pass
    viis, vijs, sign_J = handedness_sync(viis, vijs, rots_gt)
    vis = estimate_third_rows(vijs, viis, rots_gt)
    rots = estimate_rots_from_third_rows(AbinitioSymmConfig.n_symm, npf, vis, rots_gt)
    mse = utils.check_rotations_error(rots, AbinitioSymmConfig.n_symm, rots_gt)

    print("mse=" + str(mse))


if __name__ == "__main__":
    main()

