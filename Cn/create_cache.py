import pickle
import numpy as np
import Cn.utils as utils
from tqdm import tqdm
import os

def create_cache(base_dir, n_points_sphere = 1000, n_theta = 360, inplane_rot_res = 1, rots_gt = None):

    Ris_tilde, R_theta_ijs = generate_cand_rots(n_points_sphere, inplane_rot_res)
    cijs_inds              = compute_cls_inds(Ris_tilde, R_theta_ijs, n_theta)

    filename = "cn_cache_points%d_ntheta%d_res%d.pckl" % (n_points_sphere, n_theta, inplane_rot_res)
    cache_mat_full_file_name = os.path.join(base_dir, filename)
    print('Saving data to cache file=%s' % cache_mat_full_file_name)
    f = open(cache_mat_full_file_name, 'wb')
    pickle.dump((cijs_inds, Ris_tilde, R_theta_ijs, n_theta), f)
    f.close()
    print('Cache file has been successfuly saved!')
    return cache_mat_full_file_name

def generate_cand_rots(n_points_sphere, inplane_rot_res, rots_gt=None):

    # Step 1: construct Ris_tilde (i.e. Ris modulo the inplane rotation matrices)
    vis = generate_cand_rots_third_rows(n_points_sphere, rots_gt)
    Ris_tilde = np.array([utils.complete_third_row_to_rot(vi) for vi in vis])
    theta_ij = np.arange(0, 360-inplane_rot_res, inplane_rot_res)*np.pi/180
    
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
        third_rows_bad = np.random.randn(3, n_points_sphere - len(rots_gt))
        third_rows = np.concatenate((third_rows_gt, third_rows_bad), axis=0)
    return third_rows


def compute_cls_inds(Ris_tilde, R_theta_ijs, n_theta):

    n_points_sphere = len(Ris_tilde)
    inplane_rot_res = len(R_theta_ijs)
    cijs = np.zeros([n_points_sphere, n_points_sphere, inplane_rot_res, 2], dtype = np.uint16)

    with tqdm(total=n_points_sphere) as pbar:
        for i in np.arange(n_points_sphere):
            for j in np.arange(n_points_sphere):
                    R_cands = np.array([np.linalg.multi_dot([Ris_tilde[i].T, R_theta_ij, Ris_tilde[j]])
                                         for R_theta_ij in R_theta_ijs])
                
                    c1s = np.array([(-R_cand[1, 2],  R_cand[0, 2]) for R_cand in R_cands])
                    c2s = np.array([( R_cand[2, 1], -R_cand[2, 0]) for R_cand in R_cands])
                    
                    c1s = utils.clAngles2Ind__(c1s, n_theta)
                    c2s = utils.clAngles2Ind__(c2s, n_theta)

                    inds = np.where(c1s > n_theta/2)
                    c1s[inds] -= n_theta/2 
                    c2s[inds] += n_theta/2
                    c2s[inds] = np.mod(c2s[inds], n_theta) 
                    
                    cijs[i, j, :, 0] = c1s
                    cijs[i, j, :, 1] = c2s
            # update the bar
            if np.mod(i,10) == 0:
                pbar.update(10)
    return cijs

def read_cache(cache_mat_full_file_name):
    f = open(cache_mat_full_file_name, 'rb')
    cijs_inds, Ris_tilde, R_theta_ijs, n_theta = pickle.load(f)
    f.close()
    print('Cache file has been successfuly loaded!')
    return cijs_inds, Ris_tilde, R_theta_ijs, n_theta

if __name__ == "__main__":
    cache_mat_full_file_name = create_cache(base_dir="C:\\cn_cache", n_points_sphere = 1000, n_theta = 360, inplane_rot_res = 1)
    cijs_inds, Ris_tilde, R_theta_ijs, n_theta = read_cache(cache_mat_full_file_name)

    # assert np.all(cijs_inds == cijs_inds_)
    # assert np.all(Ris_tilde == Ris_tilde_)
    # assert np.all(R_theta_ijs == R_theta_ijs_)
    # assert np.all(n_theta == n_theta_)