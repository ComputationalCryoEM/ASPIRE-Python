import pickle
import numpy as np
import Cn.utils as utils
from tqdm import tqdm
import os


def create_cache(base_dir, n_points_sphere=1000, n_theta=360, inplane_rot_res_deg=1, rots_gt=None):
    print('creating cache')
    Ris_tilde, R_theta_ijs = generate_cand_rots(n_points_sphere, inplane_rot_res_deg, rots_gt)
    cijs_inds = compute_cls_inds(Ris_tilde, R_theta_ijs, n_theta)

    filename = "cn_cache_points%d_ntheta%d_res%d.pckl" % (n_points_sphere, n_theta, inplane_rot_res_deg)
    cache_mat_full_file_name = os.path.join(base_dir, filename)
    print('Saving data to cache file=%s' % cache_mat_full_file_name)
    f = open(cache_mat_full_file_name, 'wb')
    pickle.dump((cijs_inds, Ris_tilde, R_theta_ijs, n_theta), f)
    f.close()
    print('Cache file has been successfully saved!')
    return cache_mat_full_file_name, Ris_tilde, R_theta_ijs, cijs_inds


def generate_cand_rots(n_points_sphere, inplane_rot_res_deg, rots_gt=None):

    # Step 1: construct Ris_tilde (i.e. Ris modulo the in-plane rotation matrices)
    vis = generate_cand_rots_third_rows(n_points_sphere, rots_gt)
    Ris_tilde = np.array([utils.complete_third_row_to_rot(vi) for vi in vis])
    theta_ij = np.arange(0, 360, inplane_rot_res_deg) * np.pi / 180
    
    # Step 2: construct all in-plane rotation matrices R(theta_ij)
    cos_theta_ijs = np.cos(theta_ij)
    sin_theta_ijs = np.sin(theta_ij)
    R_theta_ijs = np.array([[[c, -s, 0], [s, c, 0], [0, 0, 1]] for (c, s) in zip(cos_theta_ijs, sin_theta_ijs)])

    return Ris_tilde, R_theta_ijs


def generate_cand_rots_third_rows(n_points_sphere, rots_gt=None):

    if rots_gt is None:
        third_rows = np.random.randn(n_points_sphere, 3)
        third_rows = np.array([third_row/np.linalg.norm(third_row) for third_row in third_rows])
    else:
        third_rows_gt = np.array([rot[-1] for rot in rots_gt])
        third_rows_other = np.random.randn(n_points_sphere - len(third_rows_gt), 3)
        third_rows_other = np.array([third_row / np.linalg.norm(third_row) for third_row in third_rows_other])
        third_rows = np.vstack((third_rows_gt, third_rows_other))
    return third_rows


def compute_cls_inds(Ris_tilde, R_theta_ijs, n_theta):

    n_points_sphere = len(Ris_tilde)
    n_theta_ijs = len(R_theta_ijs)
    cij_inds = np.zeros([n_points_sphere, n_points_sphere, n_theta_ijs, 2], dtype=np.uint16)

    with tqdm(total=n_points_sphere) as pbar:
        for i in range(n_points_sphere):
            for j in range(n_points_sphere):
                print(str(i), str(j))
                R_cands = np.array([np.linalg.multi_dot([Ris_tilde[i].T, R_theta_ij, Ris_tilde[j]])
                                     for R_theta_ij in R_theta_ijs])

                c1s = np.array([(-R_cand[1, 2],  R_cand[0, 2]) for R_cand in R_cands])
                c2s = np.array([( R_cand[2, 1], -R_cand[2, 0]) for R_cand in R_cands])

                c1s = utils.clAngles2Ind__(c1s, n_theta)
                c2s = utils.clAngles2Ind__(c2s, n_theta)

                inds = np.where(c1s >= n_theta//2)
                c1s[inds] -= n_theta//2
                c2s[inds] += n_theta//2
                c2s[inds] = np.mod(c2s[inds], n_theta)

                cij_inds[i, j, :, 0] = c1s
                cij_inds[i, j, :, 1] = c2s
            # update the bar
            if np.mod(i, 10) == 0:
                pbar.update(10)
    return cij_inds


def read_cache(cache_mat_full_file_name):
    f = open(cache_mat_full_file_name, 'rb')
    cijs_inds, Ris_tilde, R_theta_ijs, n_theta = pickle.load(f)
    f.close()
    print('Cache file has been successfully loaded!')
    return cijs_inds, Ris_tilde, R_theta_ijs, n_theta


if __name__ == "__main__":
    cache_mat_full_file_name, Ris_tilde, R_theta_ijs, cijs_inds = create_cache(base_dir=".", n_points_sphere=30, n_theta=360, inplane_rot_res_deg=1)
    cijs_inds_, Ris_tilde_, R_theta_ijs_, n_theta_ = read_cache(cache_mat_full_file_name)

    assert np.all(cijs_inds == cijs_inds_)
    assert np.all(Ris_tilde == Ris_tilde_)
    assert np.all(R_theta_ijs == R_theta_ijs_)
    assert np.all(360 == n_theta_)

