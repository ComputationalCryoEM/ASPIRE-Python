## -*- coding: utf-8 -*-
#"""
#Created on Sun Jan 13 17:20:33 2019
#
#@author: Gabi
#"""
#
#import numpy as np
#import scipy.linalg
#import utils as utils
#
#J = np.diag([1,1,-1])
#seed = 12345 # to reproduce results
#
#	 
#def g_sync(rots,n_symm,rots_gt):
#    """ Every calculated rotation might be a the rotated version of g^s_{i}, 
#    where s_{i} \in [n] of the ground truth rotatation. This method synchronizes 
#    all rotations so that only a possibly single global rotation should be applied 
#    to all rotation. """
#    assert len(rots) == len(rots_gt), "#rots is not equal to #rots_gt"
#    n_images = len(rots)
#
#
##    a rotation of 360/n_symm degrees about the z-axis
#    cos_a = np.cos(2*np.pi/n_symm)
#    sin_a = np.sin(2*np.pi/n_symm)
#    g = np.array([[cos_a, -sin_a, 0],
#                  [sin_a, cos_a, 0],
#                  [    0,     0, 1]]) 
# 
#    A_g = np.zeros((n_images,n_images),dtype=complex)
#    sign_g_Ri = np.zeros(n_symm)
#    
#    for i in np.arange(n_images):
#        for j in np.arange(i+1,n_images):
#            Ri = rots[i]
#            Rj = rots[j]
#            Rij = np.dot(Ri.T,Rj)
#            
#            Ri_gt = rots_gt[i]
#            Rj_gt = rots_gt[j]
#            
#            
#            diffs = np.zeros(n_symm)
#            
#            for s in np.arange(n_symm):
#                Rij_gt = np.dot(Ri_gt.T,np.dot(np.linalg.matrix_power(g,s),Rj_gt))
#                diffs[s] = min([np.linalg.norm(Rij-Rij_gt,'fro'), 
#                                np.linalg.norm(Rij-J_conjugate(Rij_gt),'fro')])
#            
#            ind = np.argmin(diffs)
##            print(i,j,ind)
#            A_g[i,j] = np.exp(1j*2*np.pi/n_symm*ind)
#    
#    # A_g(k,l) is exp(-j(-theta_k+theta_j)) so we use transpose and not
#    # conjugate-transpose to obtain lower triangular entries
#    A_g = A_g + A_g.T
#    # Diagonal elements correspond to exp(-i*0) so put 1. 
#    # This is important only for verification purposes that spectrum is (K,0,0,0...,0)
#    A_g = A_g + np.eye(n_images) 
#
#    # calc the top 5 eigs
#    eig_vals,eig_vecs = scipy.linalg.eigh(A_g,eigvals=(n_images-3,n_images-1))
#    evect1 = eig_vecs[:,-1] 
#    
#    print("g_sync top 5 eigenvalues are " + str(eig_vals))
#    
#    angles = np.exp(1j*2*np.pi/n_symm*np.arange(n_symm))
#    sign_g_Ri = np.zeros(n_images)
#    
#    for ii in np.arange(n_images):
#        zi = evect1[ii]
#        zi = zi/np.abs(zi) # rescale so it lies on unit circle
#        # Since a ccw and a cw closest are just as good, 
#        # we take the absolute value of the angle
#        angleDists = np.abs(np.angle(zi/angles))
#        ind = np.argmin(angleDists)
#        sign_g_Ri[ii] = ind
#    
##    return np.zeros(n_images).astype(int)
##    print("sign_g_Ri"+ str(sign_g_Ri.astype(int)))
#    return sign_g_Ri.astype(int)
#
#
#def check_rotations_error(rots,n_symm,rots_gt):
#    """ Our estimate for each rotation matrix Ri may be g^{s}Ri for s in [n_symm] 
#        independently of other rotation matrices. As such, for error analysis,
#        we perform a g-synchronization. """
#    assert len(rots) == len(rots_gt), "#rots is not equal to #rots_gt"
#    n_images = len(rots)
#    
#    #    a rotation of 360/n_symm degrees about the z-axis
#    cos_a = np.cos(2*np.pi/n_symm)
#    sin_a = np.sin(2*np.pi/n_symm)
#    g = np.array([[cos_a, -sin_a, 0],
#                  [sin_a, cos_a, 0],
#                  [    0,     0, 1]]) 
#
#    sign_g_Ri = g_sync(rots,n_symm,rots_gt)
##    sign_g_Ri = s_is+5
#    
#    rots_stack =  np.zeros((3*n_images,3))
#    rots_gt1_stack = np.zeros((3*n_images,3))
#    rots_gt2_stack = np.zeros((3*n_images,3))
#    for i, (rot,rot_gt) in enumerate(zip(rots,rots_gt)):
#        rots_stack[3*i:3*i+3] = rots[i].T
#        rots_gt_i = np.dot(np.linalg.matrix_power(g,sign_g_Ri[i]),rots_gt[i])
#        rots_gt1_stack[3*i:3*i+3] = rots_gt_i.T
#        rots_gt2_stack[3*i:3*i+3] = J_conjugate(rots_gt_i).T    
#    
#    # Compute the two possible orthogonal matrices which register the
#    # estimated rotations to the true ones.
#    O1 = np.dot(rots_stack.T,rots_gt1_stack)/n_images
#    O2 = np.dot(rots_stack.T,rots_gt2_stack)/n_images
#    
#    # We are registering one set of rotations (the estimated ones) to
#    # another set of rotations (the true ones). Thus, the transformation
#    # matrix between the two sets of rotations should be orthogonal. This
#    # matrix is either O1 if we recover the non-reflected solution, or O2,
#    # if we got the reflected one. In any case, one of them should be
#    # orthogonal.
#
#    err1 = np.linalg.norm(np.dot(O1,O1.T)-np.eye(3),'fro')
#    err2 = np.linalg.norm(np.dot(O2,O2.T)-np.eye(3),'fro')
#
#    
#    # In cany case, enforce the registering matrix O to be a rotation.
#    if err1 < err2:
#        u,_,vh = np.linalg.svd(O1) # Use o1 as the registering matrix
#        flag = 1
#    else:
#        u,_,vh = np.linalg.svd(O2) # Use o2 as the registering matrix
#        flag = 2
#
##    print("flag="+str(flag))
#    
#    O =  np.transpose(np.dot(u,vh))
#    rots_alligned = np.zeros_like(rots)
#    
#    for i, (rot,rot_gt) in enumerate(zip(rots,rots_gt)):
##        rot = np.dot(O, rot)
#        # TODO: add g**sign_g_Ri[i] 
#        rot = np.dot(np.transpose(np.linalg.matrix_power(g,sign_g_Ri[i])),np.dot(O, rot))
#        if flag == 2:
#            rot = J_conjugate(rot)
#        rots_alligned[i] = rot
#        
#    diff = np.array([np.linalg.norm(rots_alligned[i]-rots_gt[i],'fro') \
#                     for i in range(len(rots))])
##    diff = np.array([np.linalg.norm(rot-rots_gt,'fro') \
##                     for rot,rot_gt in zip(rots_alligned,rots_gt)])
#    mse = np.sum(diff**2)/n_images
#    return mse 
##    rots_alligned, sign_g_Ri
#    
#def J_conjugate(rots):
#    if rots.ndim == 2:
#        return np.dot(J,np.dot(rots,J))
#    rots_out = np.zeros_like(rots)
#    for i,rot in enumerate(rots):
#        rots_out[i] = np.dot(J,np.dot(rot,J))
#    return rots_out    
##
##        
##    return np.array([np.dot(J,np.dot(rot,J)) for rot in rots])
#
#def test_check_rotations_error():
#    rots_gt = utils.generate_rots(n_images)
#    #%%
#    print("test 1: rots==rots_gt")
#    rots = rots_gt.copy()
#    print("mse=" + str(check_rotations_error(rots,n_symm,rots_gt)))
#    #%%
#    print("test 2: rots==J*rots_gt*J")
#    rots = rots_gt.copy()
#    rots = J_conjugate(rots)
#    print("mse=" + str(check_rotations_error(rots,n_symm,rots_gt)))
#    
#    print("test 3: rots[i] = g^{s_{i}}*rots_gt[i]")
#    rots = rots_gt.copy()
#    s_is = np.random.choice(n_symm,n_images,replace=True)
##    print("s_is=" + str(s_is))
#    rots = [np.dot(np.linalg.matrix_power(g,si),rot) for si,rot in zip(s_is,rots)]
#    print("mse=" + str(check_rotations_error(rots,n_symm,rots_gt)))
#    
#    print("test 4: rots[i] = g^{s_{i}}*J*rots_gt[i]*J")
#    rots = rots_gt.copy()
#    rots = J_conjugate(rots)
#    s_is = np.random.choice(n_symm,n_images,replace=True)
##    print("s_is=" + str(s_is))
#    rots = [np.dot(np.linalg.matrix_power(g,si),rot) for si,rot in zip(s_is,rots)]
#    print("mse=" + str(check_rotations_error(rots,n_symm,rots_gt)))
#
#
#if __name__ == "__main__":
#    test_check_rotations_error()