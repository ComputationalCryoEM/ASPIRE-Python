# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:44:21 2019

@author: Gabi
"""
import numpy as np
import Cn.utils as utils
from scipy.special import comb
import scipy.linalg
import scipy


def estimate_third_rows(vijs, viis, rots_gt=None):
    """ Find the third row of each rotation matrix.
    Input parameters:
        vijs       A 3x3xnchoose2 array where each 3x3 slice holds the
                   third rows outer product of the corresponding pair of matrices.
        viis       A 3x3xn array where the i-th 3x3 slice holds the outer 
                   product of the third row of Ri with itself
     Output parameters:
         vis       A 3xnImages matrix whose i-th column equals the 
                   transpose of the third row of the rotation matrix Ri."""

    # TODO: handle is_conjugate_with_vii
    n_images = len(viis)
    assert comb(n_images, 2).astype(int) == len(vijs), "size not compatible"

    # V is a 3mx3m matrix whose (i,j)-th block of size 3x3 holds
    # the outer product vij
    V = np.zeros((3 * n_images, 3 * n_images))

    k = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            V[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = vijs[k]
            k += 1

    V = V + V.T  # since vij^{T} = vji

    for i in np.arange(n_images):
        V[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = viis[i]

    # calc the top 5 eigs
    eig_vals, eig_vecs = scipy.linalg.eigh(V, eigvals=(3 * n_images - 5, 3 * n_images - 1))

    print("V top 5 eigenvalues are " + str(eig_vals))
    # In a clean setting V is of rank 1 whose sole eigenvector is the concatenation
    # of the third rows of all rotation matrices or the negation thereof
    evect1 = eig_vecs[:, -1]

    vis = np.zeros((n_images, 3))
    for i in np.arange(n_images):
        vi = evect1[3 * i:3 * (i + 1)]
        vi /= np.linalg.norm(vi)
        vis[i] = vi

    # vis = np.array([rot_gt[2] for rot_gt in rots_gt])
    # we should retrieve either all third rows or 
    # the negation thereof (the latter correponds to 
    # an in-plane rotation of pi radians about either the x-axis or the y-axis)
    if rots_gt is not None:
        # diffs = np.zeros(4)
        mses = np.zeros(4)
        vis_gt = [rot_gt[2] for rot_gt in rots_gt]
        vis_gt_J = [rot_gt[2] * [-1, -1, 1] for rot_gt in rots_gt]

        diffs_0 = np.array([np.linalg.norm(vi - vi_gt) for (vi, vi_gt) in zip(vis, vis_gt)])
        diffs_1 = np.array([np.linalg.norm(-1*vi - vi_gt) for (vi, vi_gt) in zip(vis, vis_gt)])
        diffs_2 = np.array([np.linalg.norm(vi - vi_gt_J) for (vi, vi_gt_J) in zip(vis, vis_gt_J)])
        diffs_3 = np.array([np.linalg.norm(-1*vi - vi_gt_J) for (vi, vi_gt_J) in zip(vis, vis_gt_J)])
        mses[0] = np.mean(diffs_0 ** 2)
        mses[1] = np.mean(diffs_1 ** 2)
        mses[2] = np.mean(diffs_2 ** 2)
        mses[3] = np.mean(diffs_3 ** 2)
        min_mse = np.argmin(mses)
        # diffs[0] = np.linalg.norm(vis - vis_gt, 'fro')
        # diffs[1] = np.linalg.norm(-1 * vis - vis_gt, 'fro')
        #
        # diffs[2] = np.linalg.norm(vis - vis_gt_J, 'fro')
        # diffs[3] = np.linalg.norm(-1 * vis - vis_gt_J, 'fro')

        # min_diff = np.argmin(diffs)

        if min_mse == 1 or min_mse == 3:
            # TODO: This is a hack. Getting the negated third rows (due to the factorization of V), means that
            # depending on te impl of complete_third_row_to_rot, we are paying O_x (an in-plane rotation of 180
            # degrees about the x-axis), or O_y (an in-plane rotation of 180 degrees about the y-axis) As a result
            # our estimated Ri are such that Ri = g^{s_i}*O_x*Ri_gt and currently g_sync does not handle this. Once
            # g_sync is fixed, this hack can be removed
            print("third rows found with minus sign, negating all third rows")
            vis = -1 * vis
        # diff = diffs[min_diff]
        # print("third rows diff = " + str(diff))
        mse = mses[min_mse]
        print("MSE of vis=%10f" % mse)
    return vis


def test_estimate_third_rows(n_images=100):
    rots_gt = utils.generate_rots(n_images)
    vis_gt = np.zeros((n_images, 3))
    for i, rot_gt in enumerate(rots_gt):
        vis_gt[i] = rot_gt[-1]

    vijs = np.zeros((comb(n_images, 2).astype(int), 3, 3))
    k = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            vijs[k] = np.outer(vis_gt[i], vis_gt[j])
            k += 1

    viis = [np.outer(vi_gt, vi_gt) for vi_gt in vis_gt]

    # %% testing n_symm <=4
    estimate_third_rows(vijs, viis, rots_gt)


if __name__ == "__main__":
    test_estimate_third_rows(n_images=100)
