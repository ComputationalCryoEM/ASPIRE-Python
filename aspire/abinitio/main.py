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
    projs = projs.transpose((2, 0, 1)).copy()
    vol = cryo_abinitio_c1_worker(algo, projs)


def cryo_abinitio_c1_worker(alg, projs, outvol=None, outparams=None, showfigs=None, verbose=None, n_theta=360, n_r=0.5, max_shift=0.15, shift_step=1):
    num_projs = projs.shape[0]
    resolution = projs.shape[1]
    n_r *= resolution
    max_shift *= resolution

    if projs.shape[1] != projs.shape[2]:
        raise ValueError('input images must be squares')

    # why 0.45
    mask_radius = resolution * 0.45
    # mask_radius is ?.5
    if mask_radius * 2 == int(mask_radius * 2):
        mask_radius = int(np.ceil(mask_radius))
    # mask is not ?.5
    else:
        mask_radius = int(round(mask_radius))

    # mask projections
    m = fuzzy_mask(resolution, 2, mask_radius, 2)
    projs *= m

    # compute polar fourier transform
    pf, _ = cryo_pft(projs, n_r, n_theta)

    return 0


def cryo_pft(p, n_r, n_theta):
    return 0, 0


def fuzzy_mask(n, dims, r0, rise_time, origin=None):
    if isinstance(n, int):
        n = np.array([n])

    if isinstance(r0, int):
        r0 = np.array([r0])

    center = (n + 1.0) / 2
    k = 1.782 / rise_time

    if dims == 1:
        if origin is None:
            origin = center
            origin = origin.astype('int')
        r = np.abs(np.arange(1 - origin[0], n - origin[0] + 1))

    elif dims == 2:
        if origin is None:
            origin = np.floor(n / 2) + 1
            origin = origin.astype('int')
        if len(n) == 1:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1]

        if len(r0) < 2:
            r = np.sqrt(np.square(x) + np.square(y))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]))

    elif dims == 3:
        if origin is None:
            origin = center
            origin = origin.astype('int')
        if len(n) == 1:
            x, y, z = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y, z = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1, 1 - origin[2]:n[2] - origin[2] + 1]

        if len(r0) < 3:
            r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]) + np.square(z * r0[0] / r0[2]))
    else:
        return 0  # raise error

    m = 0.5 * (1 - sp.erf(k * (r - r0[0])))
    return m


pass

run()
