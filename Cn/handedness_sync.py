import numpy as np
import scipy
import Cn.utils as utils
from tqdm import tqdm
from scipy.sparse.linalg import eigs


def handedness_sync(viis, vijs):
    print('handedness synchronization')
    n_images = len(viis)
    m_choose_2 = len(vijs)

    assert m_choose_2 == scipy.special.comb(n_images, 2).astype(int)

    n_edges = 3 * scipy.special.comb(n_images, 3).astype(int)

    II = np.zeros(n_edges, dtype='int')
    JJ = np.zeros(n_edges, dtype='int')
    VV = np.zeros(n_edges)

    ii = 0
    with tqdm(total=n_images) as pbar:
        for i in np.arange(n_images):
            for j in np.arange(i + 1, n_images):
                for k in np.arange(j + 1, n_images):
                    ind_ij = uppertri_ijtoind(i, j, n_images)
                    ind_ik = uppertri_ijtoind(i, k, n_images)
                    ind_jk = uppertri_ijtoind(j, k, n_images)

                    vij = vijs[ind_ij]
                    vik = vijs[ind_ik]
                    vjk = vijs[ind_jk]

                    # compute 4 discrepancyRij values, the lowest one indicating the
                    # +-1 on the edges of the triangle (ij)-(jk)-(ki)
                    vij_J = utils.J_conjugate(vij)
                    vjk_J = utils.J_conjugate(vjk)
                    vik_J = utils.J_conjugate(vik)

                    y = np.zeros(4)
                    y[0] = discrepancy_vij(vij, vjk, vik)
                    y[1] = discrepancy_vij(vij_J, vjk, vik)
                    y[2] = discrepancy_vij(vij, vjk_J, vik)
                    y[3] = discrepancy_vij(vij, vjk, vik_J)

                    imin = np.argmin(y)
                    if imin == 0:
                        vals_edges = [1, 1, 1]
                    elif imin == 1:
                        vals_edges = [-1, 1, -1]
                    elif imin == 2:
                        vals_edges = [-1, -1, 1]
                    else:
                        vals_edges = [1, -1, -1]

                    II[ii] = ind_ij; JJ[ii] = ind_jk; VV[ii] = vals_edges[0]; ii += 1
                    II[ii] = ind_jk; JJ[ii] = ind_ik; VV[ii] = vals_edges[1]; ii += 1
                    II[ii] = ind_ik; JJ[ii] = ind_ij; VV[ii] = vals_edges[2]; ii += 1
            # update the bar
            if np.mod(i, 10) == 0:
                pbar.update(10)
    S = scipy.sparse.csr_matrix((VV, (II, JJ)))
    S = S + S.T
    # import time
    # tic1 = time.time()
    _, evect = power_iteration(S)
    # tic2 = time.time()
    # _, a = eigs(S, k=1, maxiter=1000, tol=0.01)
    # tic3 = time.time()
    #
    # a = np.sign(a[:, 0].astype('float'))
    # evect = np.sign(evect)
    # print(np.linalg.norm(a - evect))
    # print(tic2 - tic1)
    # print(tic3 - tic2)

    sign_J = [1 if x > 0 else -1 for x in evect]
    for i in np.arange(len(vijs)):
        if evect[i] < 0:
            vijs[i] = utils.J_conjugate(vijs[i])

    # step 2: viis J-sync
    for i in np.arange(n_images):
        vii = viis[i]
        total_J_conj = 0
        for j in np.arange(n_images):
            if j < i:
                vij = vijs[uppertri_ijtoind(j, i, n_images)]
            elif j == i:
                continue
            else:
                vij = vijs[uppertri_ijtoind(i, j, n_images)]

            err1 = np.linalg.norm(np.dot(vii, vij) - vij, 'fro')
            err2 = np.linalg.norm(np.dot(utils.J_conjugate(vii), vij) - vij, 'fro')
            if err1 < err2:
                total_J_conj -= 1
            else:
                total_J_conj += 1
        if total_J_conj > 0:
            viis[i] = utils.J_conjugate(viis[i])

    # viis = np.array([-1*vii for vii in viis])
    return viis, vijs, sign_J


def discrepancy_vij(vij, vjk, vik):
    return np.linalg.norm(np.dot(vij, vjk) - vik, 'fro') \
           + np.linalg.norm(np.dot(vik, vjk.T) - vij, 'fro') \
           + np.linalg.norm(np.dot(vij.T, vik) - vjk, 'fro')


def power_iteration(A):
    assert A.shape[0] == A.shape[1]  # matrix must be square
    n = A.shape[0]

    v = np.ones(n) / np.sqrt(n)
    ev = eigenvalue(A, v)

    max_its = 1000
    it = 0
    while it < max_its:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break
        v = v_new
        ev = ev_new
        it += 1

    if it == max_its:
        print("Warning: power iterations xceeded the maximum number of iterations=" + str(max_its))
    return ev_new, v_new


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


def uppertri_ijtoind(i, j, n):
    # (0,1) ->1, (n-2,n-1)->n*(n-1)/2
    return (2 * n - i - 1) * i // 2 + j - (i + 1)


def test_handedness_sync(n_images=100):
    rots_gt = utils.generate_rots(n_images)
    vis_gt = np.zeros((n_images, 3))
    for i, rot_gt in enumerate(rots_gt):
        vis_gt[i] = rot_gt[-1]

    vijs = np.zeros((scipy.special.comb(n_images, 2).astype(int), 3, 3))
    k = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            vijs[k] = np.outer(vis_gt[i], vis_gt[j])
            k += 1

    viis = [np.outer(vi_gt, vi_gt) for vi_gt in vis_gt]

    # test 1: all same class
    _, _, sign_J = handedness_sync(viis, vijs)
    assert np.all(sign_J == np.ones_like(sign_J)) or np.all(-1 * sign_J == np.ones_like(sign_J))

    sign_J_gt = np.random.rand(scipy.special.comb(n_images, 2).astype(int)) > 0.5
    vijs = np.array([utils.J_conjugate(vijs[i]) if sign_J_gt[i] else vijs[i] for i in np.arange(len(sign_J_gt))])
    sign_J_gt = np.array([1 if s else -1 for s in sign_J_gt])
    _, _, sign_J = handedness_sync(viis, vijs)

    assert np.all([s == ss for (s, ss) in zip(sign_J, sign_J_gt)]) \
           or np.all([s == -1 * ss for (s, ss) in zip(sign_J, sign_J_gt)])

    print("all tests passed")


if __name__ == "__main__":
    test_handedness_sync(n_images=100)
