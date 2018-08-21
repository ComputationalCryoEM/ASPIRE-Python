import os
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.special as sp
from abinitio.data_utils import *
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.linalg as scl
import scipy.optimize as optim
from pyfftw.interfaces import numpy_fft
import pyfftw
import mrcfile
import finufftpy

np.random.seed(1137)


def run():
    algo = 2
    projs = mat_to_npy('projs')
    cryo_abinitio_c1_worker(algo, projs)


def cryo_abinitio_c1_worker(alg, projs, outvol=None, outparams=None, showfigs=None, verbose=None, n_theta=360, n_r=0.5, max_shift=0.15, shift_step=1):
    num_projs = projs.shape[2]
    resolution = projs.shape[1]
    n_r *= resolution
    max_shift *= resolution

    if projs.shape[0] != projs.shape[1]:
        raise ValueError('input images must be squares')

    # why 0.45
    mask_radius = resolution * 0.45
    # mask_radius is ?.5
    if mask_radius * 2 == int(mask_radius * 2):
        mask_radius = int(np.ceil(mask_radius))
    # mask is not ?.5
    else:
        mask_radius = int(round(mask_radius))



run()
