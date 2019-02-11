# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:48:04 2019

@author: Gabi
"""
from scipy.stats import special_ortho_group
import numpy as np
import scipy
import math
import pickle


def generate_g(n_symm):
    #    a rotation of 360/n_symm degrees about the z-axis
    cos_a = np.cos(2*np.pi/n_symm)
    sin_a = np.sin(2*np.pi/n_symm)
    g = np.array([[cos_a, -sin_a, 0],
                  [sin_a, cos_a, 0],
                  [0, 0, 1]])
    return g


def generate_rots(n_images=100, is_J_conj_random=False):
    """ generates n_images rotation matrices. """
    # TODO: handle random_state
    rots = special_ortho_group.rvs(dim=3, size=n_images)
    
    if is_J_conj_random:
        inds_to_J_conj = np.nonzero(scipy.stats.bernoulli.rvs(size=n_images, p=0.5))
        rots[inds_to_J_conj] = J_conjugate(rots[inds_to_J_conj])
    return rots


def complete_third_row_to_rot(r3):
    """ Constructs a rotation matrix whose third row is equal to a given row vector.
        Input parameters:
                r3         A 1x3 vector of norm 1
        Output parameters:
                R          A rotation matrix whose third row is equal to r3 """
    assert np.abs(np.linalg.norm(r3)-1) < 1e-5, "input vector must have norm equal to 1"            
    
    # handle the case that the third row coincides with the z-axis
    if np.linalg.norm(r3-[0, 0, 1]) < 1e-05:
        return np.eye(3)
    
    # tmp is non-zero since r3 does not coincide with the z-axis
    tmp = np.sqrt(r3[0]**2 + r3[1]**2)
    # contruct an orthogonal row vector of norm 1
    r1 = np.array([r3[1]/tmp, -r3[0]/tmp, 0])
    # construct r2 so that r3 = r1xr2
    r2 = np.array([r3[0]*r3[2]/tmp, r3[1]*r3[2]/tmp, -tmp])

    return np.vstack((r1, r2, r3))


def test_complete_third_row_to_rot():

    r3 = np.random.randn(3)
    r3 /= np.linalg.norm(r3)
    R = complete_third_row_to_rot(r3)

    assert np.isclose(np.linalg.norm(np.dot(R, R.T) - np.eye(3), 'fro'), 0), "output matrix is not orthogonal"
    assert np.isclose(np.linalg.det(R), 1), "output matrix should have det 1"
    
    print("method test_complete_third_row_to_rot is successful")
    

def g_sync(rots, n_symm, rots_gt):
    """ Every calculated rotation might be a the rotated version of gs_i, 
    where s_{i} in [n] of the ground truth rotatation. This method synchronizes 
    all rotations so that only a possibly single global rotation should be applied 
    to all rotation. """
    assert len(rots) == len(rots_gt), "#rots is not equal to #rots_gt"
    n_images = len(rots)

    g = generate_g(n_symm)

    A_g = np.zeros((n_images, n_images), dtype=complex)
    
    for i in np.arange(n_images):
        for j in np.arange(i+1, n_images):
            Ri = rots[i]
            Rj = rots[j]
            Rij = np.dot(Ri.T, Rj)
            
            Ri_gt = rots_gt[i]
            Rj_gt = rots_gt[j]
            
            diffs = np.zeros(n_symm)
            for s in np.arange(n_symm):
                g_s = np.linalg.matrix_power(g, s)
                Rij_gt = np.linalg.multi_dot([Ri_gt.T, g_s, Rj_gt])
                diffs[s] = min([np.linalg.norm(Rij-Rij_gt, 'fro'),
                                np.linalg.norm(Rij-J_conjugate(Rij_gt), 'fro')])
            ind = np.argmin(diffs)
            A_g[i, j] = np.exp(-1j*2*np.pi/n_symm*ind)
    
    # A_g(k,l) is exp(-j(-theta_k+theta_j))
    A_g = A_g + np.conj(A_g).T
    # Diagonal elements correspond to exp(-i*0) so put 1. 
    # This is important only for verification purposes that spectrum is (K,0,0,0...,0)
    A_g = A_g + np.eye(n_images) 

    # calc the top 5 eigs
    eig_vals, eig_vecs = scipy.linalg.eigh(A_g, eigvals=(n_images-3, n_images-1))
    evect1 = eig_vecs[:, -1]
    
    print("g_sync top 5 eigenvalues are " + str(eig_vals))
    
    angles = np.exp(1j*2*np.pi/n_symm*np.arange(n_symm))
    sign_g_Ri = np.zeros(n_images)
    
    for ii in np.arange(n_images):
        zi = evect1[ii]
        zi = zi/np.abs(zi)  # rescale so it lies on unit circle
        # Since a ccw and a cw closest are just as good, 
        # we take the absolute value of the angle
        angleDists = np.abs(np.angle(zi/angles))
        ind = np.argmin(angleDists)
        sign_g_Ri[ii] = ind
    
#    return np.zeros(n_images).astype(int)
#    print("sign_g_Ri"+ str(sign_g_Ri.astype(int)))
    return sign_g_Ri.astype(int)


def check_rotations_error(rots, n_symm, rots_gt):
    """ Our estimate for each rotation matrix Ri may be g^{s}Ri for s in [n_symm] 
        independently of other rotation matrices. As such, for error analysis,
        we perform a g-synchronization. """
    assert len(rots) == len(rots_gt), "#rots is not equal to #rots_gt"
    n_images = len(rots)
    
    g = generate_g(n_symm)

    sign_g_Ri = g_sync(rots, n_symm, rots_gt)
    
    rots_stack = np.zeros((3*n_images, 3))
    rots_gt_stack_1 = np.zeros((3*n_images, 3))
    rots_gt_stack_2 = np.zeros((3*n_images, 3))
    for i, (rot, rot_gt) in enumerate(zip(rots, rots_gt)):
        rot_gt_i = np.dot(np.linalg.matrix_power(g, sign_g_Ri[i]), rot_gt)
        rots_stack[3*i:3*i+3] = rot.T
        rots_gt_stack_1[3*i:3*i+3] = rot_gt_i.T
        rots_gt_stack_2[3*i:3*i+3] = J_conjugate(rot_gt_i).T
    
    # Compute the two possible orthogonal matrices which register the
    # true rotations to the estimated 
    O1 = np.dot(rots_stack.T, rots_gt_stack_1)/n_images
    O2 = np.dot(rots_stack.T, rots_gt_stack_2)/n_images
    
    # We are registering one set of rotations (the estimated ones) to
    # another set of rotations (the true ones). Thus, the transformation
    # matrix between the two sets of rotations should be orthogonal. This
    # matrix is either O1 if we recover the non-reflected solution, or O2,
    # if we got the reflected one. In any case, one of them should be
    # orthogonal.

    err1 = np.linalg.norm(np.dot(O1, O1.T)-np.eye(3), 'fro')
    err2 = np.linalg.norm(np.dot(O2, O2.T)-np.eye(3), 'fro')

    # In cany case, enforce the registering matrix O to be a rotation.
    if err1 < err2:
        u, _, vh = np.linalg.svd(O1)  # Use o1 as the registering matrix
        flag = 1
    else:
        u, _, vh = np.linalg.svd(O2)  # Use o2 as the registering matrix
        flag = 2
        print("detected J cojugation")
    
    O = (np.dot(u, vh)).T
    rots_alligned = np.zeros_like(rots)
    for i, rot in enumerate(rots):
        g_s = np.linalg.matrix_power(g, sign_g_Ri[i])
        rots_alligned[i] = np.linalg.multi_dot([g_s.T, O, rot])

    if flag == 2:
        rots_alligned = np.array([J_conjugate(rot) for rot in rots_alligned])

    diff = np.array([np.linalg.norm(rot-rot_gt, 'fro') for rot, rot_gt in zip(rots_alligned, rots_gt)])
    mse = np.sum(diff**2)/n_images
    return mse, rots_alligned, sign_g_Ri


def check_degrees_error(rots, n_symm, n_theta, rots_gt):

    assert len(rots) == len(rots_gt)
    n_images = len(rots)
    d_theta = 2*np.pi/n_theta
    g = generate_g(n_symm)

    g_s = np.array([np.linalg.matrix_power(g, s) for s in range(n_symm)])

    err_in_degrees = np.zeros(n_theta*n_images)
    # mean_err_per_rot = np.zeros(n_images)

    for k, rot_gt in enumerate(rots_gt):
        rays_gt = np.array([np.cos(j*d_theta)*rot_gt[:, 0] + np.sin(j*d_theta)*rot_gt[:, 1] for j in range(n_theta)])

        c_err_in_degrees = np.zeros((n_symm, n_theta))
        for s in range(n_symm):
            rot_k = np.dot(g_s[s], rots[k])
            rays_rot_k = np.array([np.cos(j*d_theta)*rot_k[:, 0] + np.sin(j*d_theta)*rot_k[:, 1] for j in range(n_theta)])

            cos_angle = np.sum(rays_gt * rays_rot_k, axis=1)
            cos_angle[cos_angle > 1] = 1
            cos_angle[cos_angle < -1] = -1
            err = np.arccos(cos_angle)
            c_err_in_degrees[s] = err*180/np.pi
        opt_s = np.argmin(np.mean(c_err_in_degrees, axis=1))
        err_in_degrees[k*n_theta: (k+1)*n_theta] = c_err_in_degrees[opt_s]

    return err_in_degrees


def J_conjugate(rots):
    J = np.diag([1, 1, -1])
    if rots.ndim == 2:
        return np.linalg.multi_dot([J, rots, J])
    else:
        return np.array([np.linalg.multi_dot([J, rot, J]) for rot in rots])


def test_check_rotations_error(n_images, n_symm, iters=1):
    if np.all(np.array([do_test_check_rotations_error(n_images, n_symm) for _ in np.arange(iters)])):
        print("all tests passed")
    else:
        print("tests failed")


def foo(n_images, n_symm):

    g = generate_g(n_symm)
    rots_gt = generate_rots(n_images)
    print("test 6: rots[i] = g^{s_{i}}*rots_gt[i]")
    rots = rots_gt.copy()
    s_is = np.random.choice(n_symm, n_images, replace=True)
    rots = [np.dot(np.linalg.matrix_power(g, si), rot) for si, rot in zip(s_is, rots)]
    mse, rots_aligned = check_rotations_error(rots, n_symm, rots_gt)
    check_degrees_error(rots_aligned, n_symm, 360, rots_gt)
    print("mse=" + str(mse))
    return rots_gt, rots, s_is


def do_test_check_rotations_error(n_images, n_symm):
    
    mses = np.zeros(8)
    
    g = generate_g(n_symm)
    rots_gt = generate_rots(n_images)

    print("test 1: rots==rots_gt")
    rots = rots_gt.copy()
    mses[0],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[0]))

    print("test 2: O*rots==rots_gt")
    rots = rots_gt.copy()
    O = generate_rots(n_images=1)
    rots = np.array([np.dot(O, rot) for rot in rots])
    mses[1],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[1]))

    print("test 3: rots==J*rots_gt*J")
    rots = rots_gt.copy()
    rots = J_conjugate(rots)
    mses[2],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[2]))

    print("test 4: rots==J*O*rots_gt*J")
    rots = rots_gt.copy()
    O = generate_rots(n_images=1)
    rots = np.array([np.dot(O, rot) for rot in rots])
    rots = J_conjugate(rots)
    mses[3],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[3]))

    print("test 5: rots[i] = g*rots_gt[i]")
    rots = rots_gt.copy()
    rots = [np.dot(g, rot) for rot in rots]
    mses[4],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[4]))

    print("test 6: rots[i] = g^{s_{i}}*rots_gt[i]")
    rots = rots_gt.copy()
    s_is = np.random.choice(n_symm, n_images, replace=True)
    rots = [np.dot(np.linalg.matrix_power(g, si), rot) for si, rot in zip(s_is, rots)]
    mses[5],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[5]))
    
    print("test 7: rots[i] = J*g^{s_{i}}*rots_gt[i]*J")
    rots = rots_gt.copy()
    s_is = np.random.choice(n_symm, n_images, replace=True)
    rots = np.array([np.dot(np.linalg.matrix_power(g, si), rot) for si, rot in zip(s_is, rots)])
    rots = J_conjugate(rots)
    mses[6],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[6]))

    print("test 8: rots[i] = J*O*g^{s_{i}}*rots_gt[i]*J")
    rots = rots_gt.copy()
    s_is = np.random.choice(n_symm, n_images, replace=True)
    rots = np.array([np.dot(np.linalg.matrix_power(g, si), rot) for si, rot in zip(s_is, rots)])
    O = generate_rots(n_images=1)
    rots = np.array([np.dot(O, rot) for rot in rots])
    rots = J_conjugate(rots)
    mses[7],*_ = check_rotations_error(rots, n_symm, rots_gt)
    print("mse=" + str(mses[7]))

    return np.all(mses < 1e-25)


def ang_to_orth(ang1, ang2, ang3):
    sa = np.sin(ang1)
    ca = np.cos(ang1)
    sb = np.sin(ang2)
    cb = np.cos(ang2)
    sc = np.sin(ang3)
    cc = np.cos(ang3)
    n = len(ang1)
    orthm = np.zeros((n, 3, 3))
    orthm[:, 0, 0] = cc*ca-sc*cb*sa
    orthm[:, 0, 1] = -cc*sa-sc*cb*ca
    orthm[:, 0, 2] = sc*sb
    orthm[:, 1, 0] = sc*ca+cc*cb*sa
    orthm[:, 1, 1] = -sa*sc+cc*cb*ca
    orthm[:, 1, 2] = -cc*sb
    orthm[:, 2, 0] = sb*sa
    orthm[:, 2, 1] = sb*ca
    orthm[:, 2, 2] = cb
    
    return orthm


def scl_detection_rate(n_symm, sclmatrix, rots_gt, n_theta, angle_tol_err_deg=5):

    sclmatrix_gt = find_scl_gt(n_symm, n_theta, rots_gt)
    n_images = len(rots_gt)
    assert sclmatrix.ndim == 2
    assert sclmatrix.shape == (n_images, 2)
    angle_tol_err = angle_tol_err_deg / 180 * np.pi  # how much angular error tol we allow

    sclmatrix_diff1 = sclmatrix_gt - sclmatrix
    sclmatrix_diff2 = sclmatrix_gt - np.flip(sclmatrix, 1)  # we need not tell the difference between the two scls
    sclmatrix_diff1_angle = sclmatrix_diff1*2*np.pi/n_theta
    sclmatrix_diff2_angle = sclmatrix_diff2*2*np.pi/n_theta

    # cosine is invariant to 2pi, and abs is invariant to +-pi due to J-conjugation
    # take the mean deviation wrt th two lines in each image
    sclmatrix_diff1_angle_mean = np.mean(np.arccos(abs(np.cos(sclmatrix_diff1_angle))), axis=1)
    sclmatrix_diff2_angle_mean = np.mean(np.arccos(abs(np.cos(sclmatrix_diff2_angle))), axis=1)

    sclmatrix_diff_angle_mean = np.vstack((sclmatrix_diff1_angle_mean, sclmatrix_diff2_angle_mean))
    scl_idx = np.argmin(sclmatrix_diff_angle_mean, axis=0)
    min_mean_angle_diff = scl_idx.choose(sclmatrix_diff_angle_mean)
    detec_rate = np.count_nonzero(min_mean_angle_diff < angle_tol_err)/n_images
    hist_hand, _ = np.histogram(scl_idx, np.arange(3))
    print("hist local handedness=" + str(hist_hand))
    print("self common lines detection rate=%.2f%%" % (detec_rate*100))
    return detec_rate


def cl_detection_rate_single(n_symm, clmatrix, rots_gt, n_theta, angle_tol_err_deg=5):

    clmatrices_gt = find_cl_gt(n_symm, n_theta, rots_gt)

    n_images = len(rots_gt)
    assert clmatrix.ndim == 2
    assert clmatrix.shape == (n_images, n_images)

    clmatrix_correct = np.zeros_like(clmatrix)
    angle_tol_err = angle_tol_err_deg / 180 * np.pi  # how much angular error tol we allow

    clmatrix_diff = np.array([clmatrix_gt - clmatrix for clmatrix_gt in clmatrices_gt])
    clmatrix_diff_angle = clmatrix_diff*2*np.pi/n_theta
    # take absolute cosine because of handedness there might be + 180 independent diff
    # for each image which at this stage hasn't been taken care yet.
    n_correct = 0
    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    hand_idx = np.zeros(m_choose_2)
    cl_idx = np.zeros(m_choose_2)
    k = 0
    for i in range(n_images):
        for j in range(i+1, n_images):
            diffs_cij = clmatrix_diff_angle[:, i, j]
            diffs_cji = clmatrix_diff_angle[:, j, i]
            diffs0 = np.arccos((np.cos(diffs_cij))) + np.arccos((np.cos(diffs_cji)))
            diffs1 = np.arccos((np.cos(diffs_cij + np.pi))) + np.arccos((np.cos(diffs_cji + np.pi)))
            min_idx0 = np.argmin(diffs0)
            min_idx1 = np.argmin(diffs1)
            min_diffs0 = diffs0[min_idx0]
            min_diffs1 = diffs1[min_idx1]

            if min_diffs0 < min_diffs1:
                hand_idx[k] = 0
                cl_idx[k] = min_idx0
            else:
                hand_idx[k] = 1
                cl_idx[k] = min_idx1
            if min(min_diffs0, min_diffs1) < 2*angle_tol_err:
                n_correct += 1
                clmatrix_correct[i, j] = 1
                clmatrix_correct[j, i] = 1
            k += 1
    hist_hand, _ = np.histogram(hand_idx, np.arange(3))
    print("hist cls handedness=" + str(hist_hand))
    hist_cl, _ = np.histogram(cl_idx, np.arange(n_symm+1))
    print("hist cls=" + str(hist_cl))
    detec_rate = n_correct / m_choose_2
    print("common lines detection rate=%.2f%%" % (detec_rate * 100))
    return detec_rate, clmatrix_correct


def test_find_cl_gt(n_symm, n_theta=360, n_images=100, rots_gt=None):
    assert n_symm > 1, "n_symm must be greater than 1"

    if not rots_gt:
        rots_gt = generate_rots(n_images)

    clmatrix_gt = find_cl_gt(n_symm, n_theta, rots_gt)
    assert clmatrix_gt.shape[0] == n_symm and clmatrix_gt.shape[1] == n_images and clmatrix_gt.shape[2] == n_images
    assert np.max(clmatrix_gt) <= n_theta and np.min(clmatrix_gt) >= 0

    # sclmatrix_gt = find_scl_gt(n_symm, n_theta, rots_gt)
    # assert sclmatrix_gt.shape[0] == n_images and sclmatrix_gt.shape[1] == 2
    # print(sclmatrix_gt)


def estimate_relative_rots_gt(n_symm, n_theta, rots_gt):
    n_images = len(rots_gt)
    Rijs = np.zeros((scipy.special.comb(n_images, 2).astype(int), 3, 3))
    g = generate_g(n_symm)
    c = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            Ri = rots_gt[i]
            Rj = rots_gt[j]
            s_ij = np.random.randint(n_symm)
            g_s_ij = np.linalg.matrix_power(g, s_ij)
            Rij = np.linalg.multi_dot([Ri.T, g_s_ij, Rj])
            if np.random.random() > 0.5:
                Rij = J_conjugate(Rij)
            Rijs[c] = Rij
            c = c + 1
    return Rijs


def estimate_all_relative_rots_gt(n_symm, rots_gt):
    n_images = len(rots_gt)
    Rijs = np.zeros((scipy.special.comb(n_images, 2).astype(int), n_symm, 3, 3))
    g = generate_g(n_symm)
    gs_s = np.array([np.linalg.matrix_power(g, s) for s in range(n_symm)])
    c = 0
    for i in range(n_images):
        for j in range(i + 1, n_images):
            Ri = rots_gt[i]
            Rj = rots_gt[j]
            Rijs[c] = np.array([np.linalg.multi_dot([Ri.T, gs, Rj]) for gs in gs_s])
            c += 1
    return Rijs


# def find_single_cl_gt(n_symm, n_theta, rots_gt):
#     print('estimating common-lines using GT')
#     clmatrix_gt = find_cl_gt(n_symm, n_theta, rots_gt, is_simulate_J=True)
#     select_id = np.random.randint(n_symm, size=clmatrix_gt.shape[1:])
#     select_id = (select_id + select_id.T)//2
#     return select_id.choose(clmatrix_gt)


def find_cl_gt(n_symm, n_theta, rots_gt, is_simulate_J=False, single_cl=False):

    n_images = len(rots_gt)
    g = generate_g(n_symm)
    # precalc all g^s
    gs = np.array([np.linalg.matrix_power(g, s) for s in range(n_symm)])
    clmatrix_gt = np.zeros((n_symm, n_images, n_images))
    for i in range(n_images):
        for j in range(i+1, n_images):
            Ri = rots_gt[i]
            Rj = rots_gt[j]
            if is_simulate_J and np.random.rand() > 0.5:  # TODO: simulate J per s not per i,j
                Ri, Rj = J_conjugate(Ri), J_conjugate(Rj)
            for s in range(n_symm):
                U = np.dot(np.dot(Ri.T, gs[s]), Rj)
                c1 = np.array([-U[1, 2], U[0, 2]])
                c2 = np.array([U[2, 1], -U[2, 0]])
                clmatrix_gt[s, i, j] = clAngles2Ind__(c1[np.newaxis, :], n_theta)
                clmatrix_gt[s, j, i] = clAngles2Ind__(c2[np.newaxis, :], n_theta)
    if single_cl:
        select_id = np.random.randint(n_symm, size=clmatrix_gt.shape[1:])
        # an ackward yet quick way to make sure that either
        # both cij and cji are j-conjugated or both are not
        select_id = (select_id + select_id.T) // 2
        clmatrix_gt = select_id.choose(clmatrix_gt)
    return clmatrix_gt.astype(int)


def detection_rate_self_relative_rots(Riis, n_symm, rots_gt):
    assert len(rots_gt) == len(Riis)
    n_images = len(rots_gt)

    g = generate_g(n_symm)

    min_idx = np.zeros(n_images, dtype=int)
    errs = np.zeros(n_images)
    for i, rot_gt in enumerate(rots_gt):
        Rii_gt = np.dot(np.dot(rot_gt.T, g), rot_gt)
        Rii = Riis[i]

        diff0 = np.linalg.norm(Rii - Rii_gt, 'fro')
        diff1 = np.linalg.norm(Rii.T - Rii_gt, 'fro')
        diff2 = np.linalg.norm(J_conjugate(Rii) - Rii_gt, 'fro')
        diff3 = np.linalg.norm(J_conjugate(Rii.T) - Rii_gt, 'fro')
        diffs = [diff0, diff1, diff2, diff3]
        min_idx[i] = np.argmin(diffs)
        errs[i] = diffs[min_idx[i]]

    mse = np.mean(errs ** 2)
    hist, _ = np.histogram(min_idx, np.arange(5))
    print("MSE of Riis=%10f, n_symm = %d" % (mse, n_symm))
    print("hist=" + str(hist))
    return mse, hist


def detection_rate_relative_rots(Rijs, n_symm, rots_gt):
    n_images = len(rots_gt)
    assert scipy.special.comb(n_images, 2) == len(Rijs)
    n_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    g = generate_g(n_symm)

    gs = np.zeros((n_symm, 3, 3,))
    for s in np.arange(n_symm):
        gs[s] = np.linalg.matrix_power(g, s)

    min_idx = np.zeros(n_choose_2, dtype=int)
    errs = np.zeros(n_choose_2)
    c = 0
    diffs = np.zeros(n_symm)
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            Rij = Rijs[c]
            Ri_gt = rots_gt[i]
            Rj_gt = rots_gt[j]
            for s in np.arange(n_symm):
                Rij_s_gt = np.linalg.multi_dot([Ri_gt.T, gs[s], Rj_gt])
                diffs[s] = np.min([np.linalg.norm(Rij - Rij_s_gt, 'fro'),
                                   np.linalg.norm(J_conjugate(Rij) - Rij_s_gt, 'fro')])
            min_idx[c] = np.argmin(diffs)
            errs[c] = diffs[min_idx[c]]
            c = c + 1
    mse = np.mean(errs ** 2)
    hist, _ = np.histogram(min_idx, np.arange(n_symm + 1))
    print("MSE of Rijs=%.10f, n_symm = %d" % (mse, n_symm))
    print("hist=" + str(hist))
    return mse, hist


def find_scl_gt(n_symm, n_theta, rots_gt, is_simulate_J=False, is_simulate_transpose=False):
    assert n_symm == 3 or n_symm == 4, "supports only C3 or C4"
    print('estimating self common-lines using GT')
    n_images = len(rots_gt)

    sclmatrix_gt = np.zeros((n_images, 2))
    g = generate_g(n_symm)
    g_pow_n_one = np.linalg.matrix_power(g, n_symm - 1)

    for i in np.arange(n_images):
        Ri = rots_gt[i]

        if is_simulate_J and np.random.rand() > 0.5:
            Ri = J_conjugate(Ri)

        U1 = np.dot(np.dot(Ri.T, g), Ri)
        U2 = np.dot(np.dot(Ri.T, g_pow_n_one), Ri)
        if is_simulate_transpose and np.random.rand() > 0.5:
            U1, U2 = U1.T, U2.T
        c1 = np.array([-U1[1, 2], U1[0, 2]])
        c2 = np.array([-U2[1, 2], U2[0, 2]])

        sclmatrix_gt[i, 0] = clAngles2Ind__(c1[np.newaxis, ], n_theta)
        sclmatrix_gt[i, 1] = clAngles2Ind__(c2[np.newaxis, ], n_theta)

    return sclmatrix_gt.astype(int)


def clAngles2Ind(clAngles, n_theta):
    theta = math.atan2(clAngles[1], clAngles[0])

    theta = np.mod(theta, 2*np.pi)  # Shift from [-pi,pi] to [0,2*pi).
    idx = np.mod(np.round(theta/(2*np.pi)*n_theta), n_theta).astype(int)  # linear scale from [0,2*pi) to [0,n_theta).

    return idx


def clAngles2Ind__(clAngles, n_theta):
    thetas = np.arctan2(clAngles[:, 1], clAngles[:, 0])
    thetas = np.mod(thetas, 2*np.pi)  # Shift from [-pi,pi] to [0,2*pi).
    return np.mod(np.round(thetas/(2*np.pi)*n_theta), n_theta).astype(int)  # linear scale from [0,2*pi) to [0,n_theta).


def detection_rate_gammas(gammas, n_symm, rots_gt, angle_deg_tol_err=10):
    assert len(gammas) == len(rots_gt)
    n_images = len(gammas)
    gammas_gt = np.zeros_like(gammas)
    g = generate_g(n_symm)
    angle_tol_err = angle_deg_tol_err / 180 * np.pi
    for i, rot_gt in enumerate(rots_gt):
        Ri_3 = rot_gt[:, 2]
        gammas_gt[i] = np.arccos(np.dot(Ri_3, np.dot(g, Ri_3)))

    n_correct_idxs = np.count_nonzero(np.abs(gammas_gt - gammas) < angle_tol_err)
    return n_correct_idxs / n_images * 100
    # TODO: add offending rots


def calc_shift_phases(n_r, max_shift, shift_step):
    n_shifts = int(np.ceil(2*max_shift/shift_step + 1))
    shift_phases = np.zeros((n_shifts, n_r), 'complex128')
    shifts = np.array([-max_shift + i*shift_step for i in range(n_shifts)], dtype='int')
    rs = np.arange(n_r)  # r_s should correspond to the real (underlying) frequency
    for i in range(n_shifts):
        shift = shifts[i]
        shift_phases[i] = np.exp(-2*np.pi*1j*rs*shift/(2*n_r-1))  # TODO: 2*n_r+1 or 2*n_r-1  ?

    return shift_phases


def detection_rate_viis(viis, n_symm, rots_gt):
    assert len(rots_gt) == len(viis)

    n_images = len(rots_gt)
    viis_gt = np.array([np.outer(rot_gt[2], rot_gt[2]) for rot_gt in rots_gt])

    min_idx = np.zeros(n_images, dtype=int)
    errs = np.zeros(n_images)
    diffs = np.zeros(2)
    for i, (vii_gt, vii) in enumerate(zip(viis_gt, viis)):
        diffs[0] = np.linalg.norm(vii - vii_gt, 'fro')
        diffs[1] = np.linalg.norm(J_conjugate(vii) - vii_gt, 'fro')
        min_idx[i] = np.argmin(diffs)
        errs[i] = diffs[min_idx[i]]
    mse = np.mean(errs ** 2)
    hist, _ = np.histogram(min_idx, np.arange(3))
    print("MSE of viis=%f, n_symm = %d" % (mse, n_symm))
    print("hist=" + str(hist))
    return mse, hist


def detection_rate_vijs(vijs, n_symm, rots_gt):
    n_images = len(rots_gt)
    assert scipy.special.comb(n_images, 2) == len(vijs)
    n_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    vijs_gt = np.zeros((n_choose_2, 3, 3))

    c = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            vi = rots_gt[i, 2]
            vj = rots_gt[j, 2]
            vijs_gt[c] = np.outer(vi, vj)
            c = c + 1

    min_idx = np.zeros(n_choose_2, dtype=int)
    errs = np.zeros(n_choose_2)
    diffs = np.zeros(2)
    for i, (vij_gt, vij) in enumerate(zip(vijs_gt, vijs)):
        diffs[0] = np.linalg.norm(vij - vij_gt, 'fro')
        diffs[1] = np.linalg.norm(J_conjugate(vij) - vij_gt, 'fro')
        min_idx[i] = np.argmin(diffs)
        errs[i] = diffs[min_idx[i]]
    mse = np.mean(errs ** 2)
    hist, _ = np.histogram(min_idx, np.arange(3))
    print("MSE of vijs=%f, n_symm = %d" % (mse, n_symm))
    print("hist=" + str(hist))
    return mse, hist



if __name__ == "__main__":
    n_symm = 3

    file_Name = "testfile"
    fileObject = open(file_Name, 'rb')
    # load the object from the file into var b
    rots_gt_arr, rots_arr, si_s = pickle.load(fileObject)

    fileObject.close()

    rots_gt = rots_gt_arr[0]
    rots = rots_arr[0]
    sis = si_s[0].astype(int)

    # g = generate_g(n_symm)
    # rots = [np.dot(np.linalg.matrix_power(g, n_symm-si), rot) for si, rot in zip(sis, rots)]

    mse, rots_aligned, sign_g_Ri = check_rotations_error(rots, n_symm, rots_gt)
    check_degrees_error(rots_aligned, n_symm, 360, rots_gt)
    print("mse=" + str(mse))

    # n_images = 100
    # # test_complete_third_row_to_rot()
    # rots_gt_arr = np.zeros((5,n_images,3,3))
    # rots_arr = np.zeros((5,n_images,3,3))
    # si_s = np.zeros((5, n_images))
    # for i in range(5):
    #     rots_gt_arr[i], rots_arr[i], si_s[i]  = foo(n_images, n_symm)
    #
    # file_Name = "testfile"
    #
    # # open the file for writing
    # fileObject = open(file_Name, 'wb')
    #
    # # this writes the object a to the
    # # file named 'testfile'
    # pickle.dump((rots_gt_arr, rots_arr, si_s), fileObject)
    #
    # # here we close the fileObject
    # fileObject.close()

    # test_check_rotations_error(n_images, n_symm, iters=3)
