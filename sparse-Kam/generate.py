import pywt
import numpy as np

from scipy.stats import ortho_group
from aspire.basis import *
from aspire.utils.fft import centered_fft3, centered_ifft3
from random import randint
from copy import deepcopy
from scipy.io import loadmat

import pickle


def block_orth(L):
    U = np.zeros( ((L+1)**2, (L+1)**2), dtype = np.complex128)
    for l in range(L+1):
        if l == 0:
            U[0,0] = 2*(randint(0,1)-0.5)
        else:
            U[(l**2):((l+1)**2),(l**2):((l+1)**2)] = ortho_group.rvs(dim=(2*l+1))

    return U
   
 
    
def gen_matrices_3D_dictionary_bessel(M,L,speed='fast'):

    #V = emd_0409()
    #V = phantom3d()
    V = emd_25892()
    V = np.float64(V)

    py_basis = gen_bas((M,M,M),L,speed,0)
    A = cart2besselcoeff(py_basis,V,speed)

    return A,py_basis,V


def gen_bas(s,l,speed='fast',gen=0):

    if speed == 'fast':
        if gen == 1:
            py_basis = FFBBasis3D(s, ell_max = l, dtype=np.float64)
            py_basis.expand_mat()

            np.save("precomputed_matrices/piv_fast_saved_py_basis_"+str(l)+str(s[0]),py_basis.piv,allow_pickle=False)
            np.save("precomputed_matrices/lu1_fast_saved_py_basis_"+str(l)+str(s[0]),py_basis.lu[0:2500,:],allow_pickle=False)
            np.save("precomputed_matrices/lu2_fast_saved_py_basis_"+str(l)+str(s[0]),py_basis.lu[2500::,:],allow_pickle=False)

            # file_save = open("precomputed_matrices/3fast_saved_py_basis_"+str(l)+str(s[0]), "wb")
            # pickle.dump(py_basis, file_save)
            # file_save.close()
        else:
            py_basis = FFBBasis3D(s, ell_max = l, dtype=np.float64)
            piv=np.load("precomputed_matrices/piv_fast_saved_py_basis_"+str(l)+str(s[0])+".npy",allow_pickle=False)
            lu1=np.load("precomputed_matrices/lu1_fast_saved_py_basis_"+str(l)+str(s[0])+".npy",allow_pickle=False)
            lu2=np.load("precomputed_matrices/lu2_fast_saved_py_basis_"+str(l)+str(s[0])+".npy",allow_pickle=False)
            lu = np.vstack((lu1,lu2))
            py_basis.piv = piv
            py_basis.lu = lu
            # file_load = open("precomputed_matrices/3fast_saved_py_basis_"+str(l)+str(s[0]), "rb")
            # py_basis = pickle.load(file_load)
            # file_load.close()
    elif speed == 'slow':
        if gen == 1:
            py_basis = FBBasis3D(s, ell_max = l, dtype=np.float64)
            py_basis.expand_mat()
            file_save = open("precomputed_matrices/saved_py_basis_"+str(l)+str(s[0]), "wb")
            pickle.dump(py_basis, file_save)
            file_save.close()
        else:
            file_load = open("precomputed_matrices/saved_py_basis_"+str(l), "rb")
            py_basis = pickle.load(file_load)
            file_load.close()

    return py_basis



def clean_coeff(A):
    s = A.shape
    L = int(np.sqrt(s[1]))-1
    Aclean = deepcopy(A)
    for l in range(L+1):
        if l%2 == 0:
            Aclean[:,(l**2):((l+1)**2)] = np.real(Aclean[:,(l**2):((l+1)**2)])
        else:
            Aclean[:,(l**2):((l+1)**2)] = 1J*np.imag(Aclean[:,(l**2):((l+1)**2)])

    return Aclean

def cart2besselcoeff(py_basis,V,speed='fast'):
##    ### For FBBasis
    if speed == 'slow':
        V = centered_fft3(V)
        c = py_basis.expand_direct_lu(V)
        A = c2A(py_basis,c)
        A = clean_coeff(A)

    #    ### For FFBBasis
    if speed == 'fast':
        c = py_basis.expand_direct_lu(V)
        A = c2A(py_basis,c)

    return A
    
def besselcoeff2cart(py_basis,A,speed='fast'):

    # ### For FBBasis
    if speed == 'slow':
        c = A2c(py_basis,A)
        V = py_basis.evaluate(c)
        V = centered_ifft3(V)

    ### For FFBBasis
    if speed == 'fast':
        c = A2c(py_basis,A)
        V = py_basis.evaluate(c)

    return V



def phantom3d():
    # load it in from a pickle-file
    fload = 'phantom64data.mat'
    data=loadmat(fload)
    return data['V']
   
def emd_0409():
    # load it in from a pickle-file
    fload = 'emd_0409.mat'
    data=loadmat(fload)
    return data['V']

def emd_25892():
    # load it in from a pickle-file
    fload = 'emd_25892.mat'
    data=loadmat(fload)
    return data['V']


## Go from c to A
def c2A(bas,c):
    ind = 0
    ind_ang = 0
    ind_radial = 0

    global_k_max = max(bas.k_max)
    A = np.zeros(
        shape=(global_k_max, (bas.ell_max+1)**2), dtype=c.dtype
    )
    
    for ell in range(0, bas.ell_max + 1):
        k_max = bas.k_max[ell]
        leftover = global_k_max - k_max
        idx_radial = ind_radial + np.arange(0, k_max)

        for _ in range(-ell, ell + 1):
            idx = ind + np.arange(0, len(idx_radial))
            #for each l,m these are the indices, so stack them for each l. For varying m, pad columns with zeros so they all have the same length
            A[:,ind_ang:ind_ang+1] = np.vstack((np.array([c[idx]]).T, np.zeros((leftover,1)))) 
            ind += len(idx)
            ind_ang += 1

        ind_radial += len(idx_radial)
    return A


## Go from A to c
def A2c(bas,A):
    ind_ang = 0
    c_un = np.empty(0)
    for ell in range(0, bas.ell_max +1):
        for _ in range(-ell, ell + 1):
            c_un = np.concatenate((c_un,  A[0:bas.k_max[ell], ind_ang:ind_ang+1].flatten()))
            ind_ang += 1
    return c_un
    

def extract_A_l(A,l):
    return deepcopy(A[:,0:(l+1)**2 ])

def extract_U_l(U,l):
    return deepcopy(U[0:(l+1)**2,0:(l+1)**2 ])

def merge_U_l(U,U_l,l):
    tmp = deepcopy(U)
    tmp[0:(l+1)**2,0:(l+1)**2 ] = deepcopy(U_l)
    return tmp
    
def cart2waveletcoeff(X,wav='haar',n=6):
    res = pywt.wavedecn(X,wav,level=n)
    return res
    
    
def waveletcoeff2cart(S,wav='haar',n=6):
    res = pywt.waverecn(S,wav)
    return res

    
    
    
def best_sparse_col(M,k):

    M = np.real(M) #want only real wavelet coefficients
    s = M.shape
    I = np.argsort(abs(M))[-k:]
    B = np.zeros(M.shape,dtype=np.complex128)
    B[I] = deepcopy(M[I])
    return B
    


def best_sparse_cell(input_list, cut):
    l = len(input_list)
    S = np.zeros((3,l), dtype=int)
    v = np.empty((0,1))
    for ind in range(0,l):
        inner_list = input_list[ind]
        
        if ind == 0:
            s = inner_list.shape

            S[:,ind] = s
            v = np.array(inner_list.flatten())

        else:
            for key in inner_list:
                s = inner_list[key].shape
                S[:,ind] = s
                v = np.concatenate((v,inner_list[key].flatten()))
    v = best_sparse_col(v,cut)
    res = deepcopy(input_list)
    curr = 0
    for ind in range(0,l):
        inner_list = input_list[ind]
        
        if ind == 0:
            inds = int(np.prod(S[:,ind]))
            tmp = np.reshape( v[curr:curr+inds], S[:,ind])
            res[ind] = tmp
            curr = curr+inds
        else:
            inds = int(np.prod(S[:,ind]))
            for key in inner_list:
                tmp = np.reshape( v[curr:curr+inds], S[:,ind])
                res[ind][key] = tmp
                curr = curr+inds

    return res
    
    
def procrustes(M):
    Ut,_,Vt = np.linalg.svd(M,full_matrices=False)
    U = Ut@Vt

    return U
    
    
def procrustes_block(M):
    s = M.shape
    L = int(np.sqrt(s[1]))-1
    U = np.zeros( ((L+1)**2, (L+1)**2))

    for l in range(L+1):
        if l == 0:
            U[0,0] = np.real(M[0,0])/abs(M[0,0])
        else:
            U[l**2:(l+1)**2,l**2:(l+1)**2] = np.real(procrustes(M[l**2:(l+1)**2,l**2:(l+1)**2]))

    return U
    

def procrustes_block_l(M,l):
    U = np.zeros( ((l+1)**2, (l+1)**2))
    if l == 0:
        U = np.real(M[0,0])/abs(M[0,0])
    else:
        U = procrustes(M[l**2:(l+1)**2,l**2:(l+1)**2])

    return U
    
