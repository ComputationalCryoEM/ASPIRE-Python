import numpy as np
from numpy.linalg import eigh

from aspire.utils.matrix import anorm
from aspire.utils.misc import all_pairs


def estimate_third_rows(vijs, viis):
    """
    Find the third row of each rotation matrix given a collection of matrices
    representing the outer products of the third rows from each rotation matrix.

    :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds the third rows
    outer product of the rotation matrices Ri and Rj.

    :param viis: An n_imgx3x3 array where the i'th 3x3 slice holds the outer product of
    the third row of Ri with itself.

    :param order: The underlying molecular symmetry.

    :return: vis, An n_imgx3 matrix whose i'th row is the third row of the rotation matrix Ri.
    """

    n_img = viis.shape[0]

    # Build matrix V whose (i,j)-th block of size 3x3 holds the outer product vij
    V = np.zeros((n_img, n_img, 3, 3), dtype=vijs.dtype)

    # All pairs (i,j) where i<j
    pairs = all_pairs(n_img)

    # Populate upper triangle of V with vijs and lower triangle with vjis, where vji = vij^T.
    for idx, (i, j) in enumerate(pairs):
        V[i, j] = vijs[idx]
        V[j, i] = vijs[idx].T

    # Populate diagonal of V with viis
    for i, vii in enumerate(viis):
        V[i, i] = vii

    # Permute axes and reshape to (3 * n_img, 3 * n_img).
    V = np.swapaxes(V, 1, 2).reshape(3 * n_img, 3 * n_img)

    # In a clean setting V is of rank 1 and its eigenvector is the concatenation
    # of the third rows of all rotation matrices.
    # In the noisy setting we use the eigenvector corresponding to the leading eigenvalue
    val, vec = eigh(V)
    lead_idx = np.argmax(val)
    lead_vec = vec[:, lead_idx]

    # We decompose the leading eigenvector and normalize to obtain the third rows, vis.
    vis = lead_vec.reshape((n_img, 3))
    vis /= anorm(vis, axes=(-1,))[:, np.newaxis]

    return vis
