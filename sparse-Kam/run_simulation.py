import copy
from generate import *
from simulate import *
from scipy.io import savemat



## Setting parameters
L = 12
M = 64
speed = 'fast'
s = (M,M,M)
max_restarts = 1
maxiter = 50
alg = 'RRR'#'alternating'
K = 4000
beta = 0.5
wav = 'haar'
lev = 6
num_batch = 400

print('Initializing matrices...')
A,py_basis,V = gen_matrices_3D_dictionary_bessel(M,L,speed)

ground = besselcoeff2cart(py_basis,A,speed)
Aorig = copy.deepcopy(A)
Utrue = block_orth(L)
A = A@Utrue
Vnoiter = besselcoeff2cart(py_basis,A,speed)

U0=block_orth(L)
U = copy.deepcopy(U0)

for l in range(L,L+1):
    numiter = 0

    py_basis_l = gen_bas((M,M,M),l)
    ground = besselcoeff2cart(py_basis_l,extract_A_l(Aorig,l),speed)
    A_l = extract_A_l(A,l)
    U_l = extract_U_l(U,l)
    Vinit = besselcoeff2cart(py_basis_l,A_l@U_l,speed)
    Vinit_rand = besselcoeff2cart(py_basis_l,A_l@extract_U_l(U0,l),speed)
    py_basis_l = gen_bas((M,M,M),l)
    carted = A_l@U_l
    Vest = besselcoeff2cart(py_basis_l,carted)

    S = cart2waveletcoeff(Vest,wav='haar',n=6)


    for k in range(1,num_batch):
        U_l, Vest = run_RRR_bessel_wavelet(K,A_l,py_basis_l,U0=U_l,beta=beta,wav=wav,lev=lev,speed=speed,M=M,max_restarts=max_restarts,maxiter=maxiter,alg=alg)
        numiter += maxiter
        Vestsmooth = besselcoeff2cart(py_basis_l,A_l@U_l,speed)
        fsave = './simulation_results/data_0409_M='+str(M)+'_l='+str(l)+'_k='+str(k)+'.mat'
        d = {'Vest':Vest, 'Vestsmooth':Vestsmooth,'Vfull':V, 'Vspharm':ground, 'M':M, 'L':L, 'Utrue':Utrue,'Vinit':Vinit,'Vinit_rand':Vinit_rand,'numiter':numiter, 'K':K,'U_l':U_l}
        savemat(fsave,d)

    U = merge_U_l(U,U_l,l)
