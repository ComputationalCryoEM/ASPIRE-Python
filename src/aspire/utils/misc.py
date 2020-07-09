"""
Miscellaneous Utilities that have no better place (yet).
"""
import logging
import numpy as np
from numpy.linalg import qr, solve

from aspire.utils.matrix import mat_to_vec, vec_to_mat
from aspire.volume import Volume

logger = logging.getLogger(__name__)


def real_type(complextype):
    """
    Get Numpy real type from corresponding complex type

    :param complextype: Numpy complex type
    :return realtype: Numpy real type
    """
    complextype = np.dtype(complextype)
    realtype = None
    if complextype == np.complex64:
        realtype = np.float32
    elif complextype == np.complex128:
        realtype = np.float64
    else:
        logger.error('Corresponding real type is not defined.')
    return realtype


def complex_type(realtype):
    """
    Get Numpy complex type from corresponding real type

    :param realtype: Numpy real type
    :return complextype: Numpy complex type
    """
    realtype = np.dtype(realtype)
    complextype = None
    if realtype == np.float32:
        complextype = np.complex64
    elif realtype == np.float64:
        complextype = np.complex128
    else:
        logger.error('Corresponding complex type is not defined')
    return complextype


def src_wiener_coords(sim, mean_vol, eig_vols, lambdas=None, noise_var=0, batch_size=512):
    """
    Calculate coordinates using Wiener filter
    :param sim: A simulation object containing the images whose coordinates we want.
    :param mean_vol: The mean volume of the source as Volume instance.
    :param eig_vols: The eigenvolumes of the source as Volume instance.
    :param lambdas: The eigenvalues in a K-by-K diagonal matrix (default `eye(K)`).
    :param noise_var: The variance of the noise in the images (default 0).
    :param batch_size: The size of the batches in which to compute the coordinates (default 512).
    :return: A K-by-`src.n` array of coordinates corresponding to the Wiener filter coordinates of each image in sim.

    The coordinates are obtained by the formula
        alpha_s = eig_vols^T H_s ( y_s - P_s mean_vol ) ,

    where P_s is the forward image mapping and y_s is the sth image,
        H_s = Sigma * P_s^T ( P_s Sigma P_s^T + noise_var I )^(-1) ,

    and Sigma is the covariance matrix eig_vols * lambdas * eig_vols^T.
    Note that when noise_var is zero, this reduces to the projecting y_s onto the span of P_s eig_vols.

    # TODO: Find a better place for this functionality other than in utils
    """
    k = eig_vols.N
    if lambdas is None:
        lambdas = np.eye(k)

    coords = np.zeros((k, sim.n))
    covar_noise = noise_var * np.eye(k)

    for i in range(0, sim.n, batch_size):
        ims = sim.images(i, batch_size)
        batch_n = ims.shape[0]
        ims -= sim.vol_forward(mean_vol, i, batch_n)

        Qs, Rs = qr_vols_forward(sim, i, batch_n, eig_vols, k)

        Q_vecs = mat_to_vec(Qs)


        # ugh
        ims = np.swapaxes(ims.data, 1, 2)
        ims = np.swapaxes(ims, 0, 2)
        im_vecs = mat_to_vec(ims)

        for j in range(batch_n):
            im_coords = Q_vecs[:, :, j].T @ im_vecs[:, j]
            covar_im = (Rs[:, :, j] @ lambdas @ Rs[:, :, j].T) + covar_noise
            xx = solve(covar_im, im_coords)
            im_coords = lambdas @ Rs[:, :, j].T @ xx
            coords[:, i+j] = im_coords

    return coords


def qr_vols_forward(sim, s, n, vols, k):
    """
    TODO: Write docstring
    TODO: Find a better place for this!
    :param sim:
    :param s:
    :param n:
    :param vols:
    :param k:
    :return:
    """
    ims = np.zeros((k, n, sim.L, sim.L), dtype=vols.dtype)
    for ell in range(k):
        ims[ell, :, :, :] = sim.vol_forward(Volume(vols[ell]), s, n).asnumpy()

    #ugh
    #ims = np.swapaxes(ims, 0, 1)

    ims = np.swapaxes(ims, 1, 3)
    ims = np.swapaxes(ims, 0, 2)
    #print('ims shape', ims.shape)
    
    Q_vecs = np.zeros((sim.L**2, k, n), dtype=vols.dtype)
    Rs = np.zeros((k, k, n), dtype=vols.dtype)

    im_vecs = mat_to_vec(ims)
    for i in range(n):
        Q_vecs[:, :, i], Rs[:, :, i] = qr(im_vecs[:, :, i])
    Qs = vec_to_mat(Q_vecs)

    return Qs, Rs
