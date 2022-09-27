import numpy as np
from generate import *
import copy



def run_RRR_bessel_wavelet(cut,A,py_basis,beta=0.5,wav='haar',lev=6,speed='fast',alg='RRR',**kwargs):
    
    print('Initializing RRR...')

    _,r = A.shape
    L = int(np.sqrt(r)-1)
    U = kwargs.get('U0', block_orth( L))

    max_restarts = kwargs.get('max_restarts', 1)
    maxiter = kwargs.get('maxiter', 10)

    carted = A@U
    carted = besselcoeff2cart(py_basis,carted,speed)

    tmpold = kwargs.get('Vest',copy.deepcopy(carted))

    S0 = kwargs.get('S0', cart2waveletcoeff(carted,wav=wav,n=lev))
    S = copy.deepcopy(S0)


    print('Running RRR...')

    for rest in range(max_restarts):
        errs = np.zeros((maxiter,1))


        for iter in range(maxiter):
            print("iteration: ",iter)

            tmp = copy.deepcopy(tmpold)

            tmp = cart2besselcoeff(py_basis,np.real(tmp),speed)
            U = procrustes_block(np.conj(A.T)@tmp)
            carted = A@U
            tmp = besselcoeff2cart(py_basis,carted,speed)

            tmp1 = copy.deepcopy(tmp)

            #####for RRR
            if alg == 'RRR':
                tmp = 2*tmp - tmpold
            #####

            S = cart2waveletcoeff(tmp,wav=wav,n=lev)
            tmp = waveletcoeff2cart(best_sparse_cell(S, cut), wav=wav,n=lev)

            if alg == 'RRR':
                tmp = tmpold + beta*(tmp - tmp1 )

            elif alg == 'alternating':
                tmp = tmp
  
            tmpold = copy.deepcopy(tmp)



    return U, tmp


