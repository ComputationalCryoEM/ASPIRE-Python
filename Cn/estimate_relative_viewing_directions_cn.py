from builtins import enumerate

import numpy as np
import Cn.utils as utils
import Cn.create_cache as create_cache
import scipy
from Cn.config_symm import AbinitioSymmConfig
from tqdm import tqdm


def estimate_relative_viewing_directions_cn(n_symm, npf, rots_gt):
    # base_dir = "."
    # n_points_sphere = 1000
    # n_theta = 360
    # inplane_rot_res = 1
    # cache_file_name, *_ = create_cache.create_cache(base_dir, n_points_sphere, n_theta, inplane_rot_res, None)
    cache_file_name = 'cn_cache_points1000_ntheta360_res1.pckl'
    print('loading line indeces cache %s.\n Please be patient...' % cache_file_name)
    cijs_inds, Ris_tilde, R_theta_ijs, n_theta = create_cache.read_cache(cache_file_name)
    AbinitioSymmConfig.n_theta = n_theta
    if n_theta != npf.shape[1]:
        raise ValueError('n_theta = %d for cache, while n_theta=%d for npf are not equal. '
                         'Either create a new cache or a new npf' % (n_theta, len(npf)))
    print('done loading indeces cache')
    n_images = len(npf)
    n_cands = len(Ris_tilde)
    # Step 1: precalculate the likelihood with respect to the self common-lines
    # note: cannot pre-compute the scls inds in cache since these depend on the symmetry class
    scls_inds = compute_scls_inds(Ris_tilde, n_symm, n_theta)
    scores_self_corrs = np.zeros((n_images, n_cands))
    for i, npf_i in enumerate(npf):
        # ignoring dc-term.
        npf_i[:, 0] = 0
        # normalize each ray to have norm equal to 1
        npf_i = np.array([ray / np.linalg.norm(ray) for ray in npf_i])
        # TODO: there is no conjugation here because when building scl_inds we impliceteley picked
        #  the antipodal line. Contrast this with the c3_c4 algorithm. Suggestion: since in c3_c3 we
        #  utilize the upfront known angles between scls fix here rather than there
        corrs_i = np.dot(npf_i[:n_theta//2, :], np.conj(npf_i).T)

        corrs_i_cands = np.array([corrs_i[scls_inds_cand[:, 0], scls_inds_cand[:, 1]] for scls_inds_cand in scls_inds])
        scores_self_corrs[i] = np.mean(np.real(corrs_i_cands), axis=1)

    cii_equators_inds = np.array([ind for (ind, Ri_tilde) in enumerate(Ris_tilde)
                                  if abs(np.arccos(Ri_tilde[2, 2]) - np.pi/2) < 10*np.pi/180])

    scores_self_corrs[:, cii_equators_inds] = 0

    # Step 2: likelihood wrt to pairwise images
    print('computing pairwise likelihood')
    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    vijs = np.zeros((m_choose_2, 3, 3))
    viis = np.zeros((n_images, 3, 3))
    g = utils.generate_g(n_symm)
    gs_s = np.array([np.linalg.matrix_power(g, s) for s in range(n_symm)])
    n_points_sphere, n_points_sphere, n_theta_ijs, _ = cijs_inds.shape
    c = 0
    e1 = [1, 0, 0]
    min_ii_norm = min_jj_norm = float('inf')
    # with tqdm(total=n_images) as pbar:
    for i in range(n_images):
        for j in range(i+1, n_images):
            print(str(i), str(j))
            npf_i = npf[i]
            npf_j = npf[j]
            npf_i[:, 0] = 0
            npf_j[:, 0] = 0
            # normalize each ray to have norm equal to 1
            npf_i = np.array([ray / np.linalg.norm(ray) for ray in npf_i])
            # normalize each ray to have norm equal to 1
            npf_j = np.array([ray / np.linalg.norm(ray) for ray in npf_j])
            corrs_ij = np.dot(npf_i[:n_theta // 2, :], np.conj(npf_j).T)
            corrs = corrs_ij[cijs_inds[..., 0], cijs_inds[..., 1]]
            corrs = np.reshape(corrs, (-1, n_symm, n_theta_ijs//n_symm))
            corrs = np.mean(corrs, axis=1)  # mean over all n_sym cls
            corrs = np.reshape(corrs, (n_points_sphere, n_points_sphere, n_theta_ijs//n_symm))
            #  the self common-lines are invariant to n_theta_ijs (i.e., in-plane rotation angles) so max them out
            opt_theta_ij_ind_per_sphere_points = np.argmax(corrs, axis=-1)
            corrs = np.max(corrs, axis=-1)
            # maximum likelihood while taking into consideration both cls and scls
            corrs = corrs * np.outer(scores_self_corrs[i], scores_self_corrs[j])

            opt_sphere_i, opt_sphere_j = np.unravel_index(np.argmax(corrs), corrs.shape)
            opt_theta_ij = opt_theta_ij_ind_per_sphere_points[opt_sphere_i, opt_sphere_j]

            opt_Ri_tilde = Ris_tilde[opt_sphere_i]
            opt_Rj_tilde = Ris_tilde[opt_sphere_j]
            opt_R_theta_ij = R_theta_ijs[opt_theta_ij]

            vii_j = np.mean(np.array([np.linalg.multi_dot([opt_Ri_tilde.T, gs, opt_Ri_tilde])
                                      for gs in gs_s]), axis=0)
            _, svals, _ = np.linalg.svd(vii_j)
            if np.linalg.norm(svals - e1, 2) < min_ii_norm:
                viis[i] = vii_j

            vjj_i = np.mean(np.array([np.linalg.multi_dot([opt_Rj_tilde.T, gs, opt_Rj_tilde])
                                      for gs in gs_s]), axis=0)
            _, svals, _ = np.linalg.svd(vjj_i)
            if np.linalg.norm(svals - e1, 2) < min_jj_norm:
                viis[j] = vjj_i

            vijs[c] = np.mean(np.array([np.linalg.multi_dot([opt_Ri_tilde.T, gs, opt_R_theta_ij, opt_Rj_tilde])
                                    for gs in gs_s]), axis=0)
            c += 1
                # update the bar
                # if np.mod(i, 10) == 0:
                #     pbar.update(10)
    if rots_gt is not None:
        # utils.detection_rate_self_relative_rots(Riis, n_symm, rots_gt)
        # utils.detection_rate_relative_rots(Rijs, n_symm, rots_gt)
        utils.detection_rate_viis(viis, n_symm, rots_gt)
        utils.detection_rate_vijs(vijs, n_symm, rots_gt)
    return viis, vijs


def compute_scls_inds(Ris_tilde, n_symm, n_theta):

    n_selfcl_pairs = (n_symm-1)//2
    n_cands = len(Ris_tilde)
    scls_inds = np.zeros((n_cands, n_selfcl_pairs, 2), dtype=np.uint16)
    g = utils.generate_g(n_symm)
    gs_s = np.array([np.linalg.matrix_power(g, s) for s in range(1, n_selfcl_pairs+1)])

    for i_cand in range(n_cands):
        Ri_tilde = Ris_tilde[i_cand]
        Riigs = np.array([np.linalg.multi_dot([Ri_tilde.T, gs, Ri_tilde]) for gs in gs_s])

        c1s = np.array([[-Riig[1, 2],  Riig[0, 2]] for Riig in Riigs])
        c2s = np.array([[ Riig[2, 1], -Riig[2, 0]] for Riig in Riigs])

        c1s_inds = utils.clAngles2Ind__(c1s, n_theta)
        c2s_inds = utils.clAngles2Ind__(c2s, n_theta)

        inds = np.where(c1s_inds >= (n_theta//2))
        c1s_inds[inds] -= (n_theta//2)
        c2s_inds[inds] += (n_theta//2)
        c2s_inds[inds] = np.mod(c2s_inds[inds], n_theta)

        scls_inds[i_cand, :, 0] = c1s_inds
        scls_inds[i_cand, :, 1] = c2s_inds
        # ciis[i] = np.ravel_multi_index([utils.clAngles2Ind__(c1s, n_theta),
        #                                 utils.clAngles2Ind__(c2s, n_theta)], (n_theta // 2, n_theta))
    return scls_inds

