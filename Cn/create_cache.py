import pickle
import numpy as np
import Cn.utils as utils
from tqdm import tqdm
import os




if __name__ == "__main__":
    cache_mat_full_file_name, Ris_tilde, R_theta_ijs, cijs_inds = create_cache(base_dir=".", n_points_sphere=30, n_theta=360, inplane_rot_res_deg=1)
    cijs_inds_, Ris_tilde_, R_theta_ijs_, n_theta_ = read_cache(cache_mat_full_file_name)

    assert np.all(cijs_inds == cijs_inds_)
    assert np.all(Ris_tilde == Ris_tilde_)
    assert np.all(R_theta_ijs == R_theta_ijs_)
    assert np.all(360 == n_theta_)

