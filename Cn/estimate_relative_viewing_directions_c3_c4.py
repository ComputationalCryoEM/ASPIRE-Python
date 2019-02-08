import numpy as np
import Cn.utils as utils
import scipy
import aspire.abinitio as abinitio
from Cn.config_symm import AbinitioSymmConfig


def estimate_relative_viewing_directions_c3_c4(n_symm, npf, n_theta, rots_gt=None):
    if AbinitioSymmConfig.is_use_gt:
        assert rots_gt is not None
        clmatrix = utils.find_cl_gt(n_symm, n_theta, rots_gt, is_simulate_J=True, single_cl=True)
        # clmatrix = utils.find_single_cl_gt(n_symm, n_theta, rots_gt)
        sclmatrix = utils.find_scl_gt(n_symm, n_theta, rots_gt,is_simulate_J=True, is_simulate_transpose=True)
    else:
        n_images = len(npf)
        max_shift_1d = np.ceil(2*np.sqrt(2)*AbinitioSymmConfig.max_shift)
        n_r = AbinitioSymmConfig.n_r
        shift_step = AbinitioSymmConfig.shift_step
        print('estimating common-lines')
        clmatrix, *_ = abinitio.cryo_clmatrix_cpu_pystyle(npf, n_images, 0,
                                                          max_shift_1d, AbinitioSymmConfig.shift_step)
        print('estimating self common-lines')
        sclmatrix, *_ = utils.find_scl(n_symm, npf, n_theta, n_r, max_shift_1d, shift_step, rots_gt)

    print('estimating relative orientations')
    Rijs = abinitio.cryo_syncmatrix_vote_3n(clmatrix, n_theta)
    print('estimating self relative orientations')
    Riis = estimate_self_relative_rots(n_symm, sclmatrix, n_theta, rots_gt)
    print('local handedness')
    viis, vijs = local_handedness_sync(n_symm, Riis, Rijs, rots_gt)
    if rots_gt is not None:
        utils.cl_detection_rate_single(n_symm, clmatrix, rots_gt, n_theta, AbinitioSymmConfig.angle_tol_err_deg)
        utils.scl_detection_rate(n_symm, sclmatrix, rots_gt, n_theta, AbinitioSymmConfig.angle_tol_err_deg)
        utils.detection_rate_self_relative_rots(Riis, n_symm, rots_gt)
        utils.detection_rate_relative_rots(Rijs, n_symm, rots_gt)
        utils.detection_rate_viis(viis, n_symm, rots_gt)
        utils.detection_rate_vijs(vijs, n_symm, rots_gt)
    return viis, vijs


def estimate_self_relative_rots(n_symm, sclmatrix, n_theta, rots_gt):
    # g = utils.generate_g(n_symm)
    # return np.array([np.linalg.multi_dot([Ri.T,g,Ri]) for Ri in rots_gt])

    # g = utils.generate_g(n_symm)
    # g_n_minus_1 = np.linalg.matrix_power(g,n_symm-1)  
    # n_images = len(rots_gt)
    # Riis = np.zeros((n_images, 3, 3))
    # for i, Ri in enumerate(rots_gt):
    #     if np.random.rand() > 0.5:
    #         Riis[i] = np.linalg.multi_dot([Ri.T,g,Ri])
    #     else:
    #         Riis[i] = np.linalg.multi_dot([Ri.T,g_n_minus_1,Ri])

    # return Riis

    assert n_symm == 3 or n_symm == 4, "supports only C3 or C4"

    cos_diff = np.cos((sclmatrix[:, 1] - sclmatrix[:, 0]) * 2 * np.pi / n_theta)

    if n_symm == 3:
        # cos_diff is supposed to be <= 0.5, but due to discretization that might be violated
        if np.max(cos_diff) > 0.5:
            print("cos(angular_diff) should be < 0.5. maximum found=%f. number of estimates exceeding=%d"
                  % (np.max(cos_diff), np.count_nonzero(cos_diff > 0.5)))
            cos_diff[cos_diff > 0.5] = 0.5
        gammas = np.arccos(cos_diff / (1 - cos_diff))
    else:
        # cos_diff is supposed to be <= 0, but due to discretization that might be violated
        if np.max(cos_diff) > 0:
            print("cos(angular-diff) should be < 0. maximum found=%f. number of estimates exceeding=%d"
                  % (np.max(cos_diff), np.count_nonzero(cos_diff > 0)))
            cos_diff[cos_diff > 0] = 0
        gammas = np.arccos((1 + cos_diff) / (1 - cos_diff))

    if rots_gt is not None:
        dr = utils.detection_rate_gammas(gammas, n_symm, rots_gt, angle_deg_tol_err=10)
        print("Rii detection rate gammas=%.2f, n_symm = %d" % (dr, n_symm))

    # calculate the remaining euler angles
    aa = sclmatrix[:, 0] * 2 * np.pi / n_theta
    bb = sclmatrix[:, 1] * 2 * np.pi / n_theta + np.pi  # TODO: get rid of the addition of pi by fixing ang_to_orth

    Riis = utils.ang_to_orth(-bb, gammas, aa)
    return Riis


def local_handedness_sync(n_symm, Riis, Rijs, rots_gt=None):
    assert n_symm == 3 or n_symm == 4

    n_images = len(Riis)
    assert scipy.special.comb(n_images, 2) == len(Rijs)

    viis = np.zeros((n_images, 3, 3))

    for i, Rii in enumerate(Riis):
        viis[i] = np.mean([np.linalg.matrix_power(Rii, s) for s in np.arange(n_symm)], axis=0)

    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    vijs = np.zeros((m_choose_2, 3, 3))
    e1 = [1, 0, 0]
    opts = np.zeros((8, 3, 3))  # holds all inner sync options per i<j
    scores_rank1 = np.zeros(8)
    min_idxs = np.zeros((m_choose_2, 3, 3))
    c = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            Rii = Riis[i]
            Rjj = Riis[j]
            Rij = Rijs[c]

            Rii_J = utils.J_conjugate(Rii)
            Rjj_J = utils.J_conjugate(Rjj)
            # testing 8 possibilities: 
            # a. whether or not to transpose Rii (so that Rii and Rjj match)
            # b. whether or not to J-conjugate Rii (so that it matches to Rij J-con class)
            # c. whether or not to J-conjugate Rjj (so that it matches to Rij J-con class)
            if n_symm == 3:
                opts[0] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj]) + np.linalg.multi_dot([Rii.T, Rij, Rjj.T])
                opts[1] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj]) + np.linalg.multi_dot([Rii_J.T, Rij, Rjj.T])
                opts[2] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj_J]) + np.linalg.multi_dot([Rii.T, Rij, Rjj_J.T])
                opts[3] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj_J]) + np.linalg.multi_dot([Rii_J.T, Rij, Rjj_J.T])

                opts[4] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj]) + np.linalg.multi_dot([Rii, Rij, Rjj.T])
                opts[5] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj]) + np.linalg.multi_dot([Rii_J, Rij, Rjj.T])
                opts[6] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj_J]) + np.linalg.multi_dot([Rii, Rij, Rjj_J.T])
                opts[7] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj_J]) + np.linalg.multi_dot([Rii_J, Rij, Rjj_J.T])

                opts = opts / 3  # normalize
            else:
                opts[0] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj])
                opts[1] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj])
                opts[2] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj_J])
                opts[3] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj_J])

                opts[4] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj])
                opts[5] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj])
                opts[6] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj_J])
                opts[7] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj_J])

                opts = opts / 2  # normalize

            for k, opt in enumerate(opts):
                _, svals, _ = np.linalg.svd(opt)
                scores_rank1[k] = np.linalg.norm(svals - e1, 2)
            min_idx = np.argmin(scores_rank1)
            vijs[c] = opts[min_idx]
            min_idxs[c] = min_idx
            c += 1

    hist, _ = np.histogram(min_idxs, np.arange(9))
    print("hist local handedness=" + str(hist))

    return viis, vijs


def test_local_handedness_sync(n_symm, n_images=100):
    rots_gt = utils.generate_rots(n_images, is_J_conj_random=False)

    viis_gt = np.zeros((n_images, 3, 3))
    for i, Ri in enumerate(rots_gt):
        vi = Ri[2]
        viis_gt[i] = np.outer(vi, vi)

    vijs_gt = np.zeros((scipy.special.comb(n_images, 2).astype(int), 3, 3))
    c = 0  # counter
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            vi = rots_gt[i, 2]
            vj = rots_gt[j, 2]
            vijs_gt[c] = np.outer(vi, vj)
            c = c + 1

    g = utils.generate_g(n_symm)
    Riis = [np.linalg.multi_dot([Ri.T, g, Ri]) for Ri in rots_gt]
    for i, Rii in enumerate(Riis):
        if np.random.random() > 0.5:
            Rii = utils.J_conjugate(Rii)
        if np.random.random() > 0.5:
            Rii = Rii.T
        Riis[i] = Rii

    Rijs = np.zeros((scipy.special.comb(n_images, 2).astype(int), 3, 3))
    c = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            Ri = rots_gt[i]
            Rj = rots_gt[j]
            s_ij = np.random.randint(n_symm)
            g_s_ij = np.linalg.matrix_power(g, s_ij)
            Rij = np.linalg.multi_dot([Ri.T, g_s_ij, Rj])
            if np.random.random() > 0.5:
                Rij = utils.J_conjugate(Rij)
            Rijs[c] = Rij
            c = c + 1

    viis, vijs = local_handedness_sync(n_symm, Riis, Rijs, rots_gt)

    diff_viis = np.zeros(n_images)
    for i, (vii, vii_gt) in enumerate(zip(viis, viis_gt)):
        diff_viis[i] = np.min([np.linalg.norm(vii - vii_gt, 'fro'), np.linalg.norm(utils.J_conjugate(vii) - vii_gt, 'fro')])
    mse_viis = np.sum(diff_viis ** 2) / n_images
    print("mse_viis=" + str(mse_viis))

    diff_vijs = np.zeros_like(vijs)
    for c, (vij, vij_gt) in enumerate(zip(vijs, vijs_gt)):
        diff_vijs[c] = np.min([np.linalg.norm(vij - vij_gt, 'fro'), np.linalg.norm(utils.J_conjugate(vij) - vij_gt, 'fro')])
    mse_vijs = np.sum(diff_vijs ** 2) / scipy.special.comb(n_images, 2)
    print("mse_vijs=" + str(mse_vijs))


def test_estimate_self_relative_rots(n_symm, n_theta, rots_gt):
    Riis = estimate_self_relative_rots(n_symm, n_theta, rots_gt)
    detection_rate_self_relative_rots(Riis, n_symm, rots_gt)


def test_estimate_relative_rots(n_symm, n_theta, rots_gt):
    Rijs = utils.estimate_relative_rots(n_symm, n_theta, rots_gt)
    utils.detection_rate_relative_rots(Rijs, n_symm, rots_gt)


if __name__ == "__main__":
    rots_gt = utils.generate_rots(n_images=100, is_J_conj_random=True)
    test_estimate_self_relative_rots(n_symm=3, rots_gt=rots_gt, n_theta=360)
    test_estimate_relative_rots(n_symm=3, rots_gt=rots_gt, n_theta=360)
    # test_local_handedness_sync(n_symm=3, n_images = 100)
    # test_local_handedness_sync(n_symm=4, n_images = 100)
