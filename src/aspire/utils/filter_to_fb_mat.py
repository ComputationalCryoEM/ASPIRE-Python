import numpy as np

from aspire.basis import FFBBasis2D
from aspire.basis.basis_utils import lgwt
from aspire.operators import BlkDiagMatrix


def filter_to_fb_mat(h_fun, fbasis):
    """
    Convert a nonradial function in k space into a basis representation.

    :param h_fun: The function form in k space.
    :param fbasis: The basis object for expanding.

    :return: a BlkDiagMatrix instance representation using the
    `fbasis` expansion.
    """

    if not isinstance(fbasis, FFBBasis2D):
        raise NotImplementedError("Currently only fast FB method is supported")
    # Set same dimensions as basis object
    n_k = fbasis.n_r
    n_theta = fbasis.n_theta
    radial = fbasis.get_radial()

    # get 2D grid in polar coordinate
    k_vals, wts = lgwt(n_k, 0, 0.5, dtype=fbasis.dtype)
    k, theta = np.meshgrid(
        k_vals, np.arange(n_theta) * 2 * np.pi / (2 * n_theta), indexing="ij"
    )

    # Get function values in polar 2D grid and average out angle contribution
    omegax = k * np.cos(theta)
    omegay = k * np.sin(theta)
    omega = 2 * np.pi * np.vstack((omegax.flatten("C"), omegay.flatten("C")))
    h_vals2d = h_fun(omega).reshape(n_k, n_theta)
    h_vals = np.sum(h_vals2d, axis=1) / n_theta

    # Represent 1D function values in fbasis
    h_fb = BlkDiagMatrix.empty(2 * fbasis.ell_max + 1, dtype=fbasis.dtype)
    ind_ell = 0
    for ell in range(0, fbasis.ell_max + 1):
        k_max = fbasis.k_max[ell]
        rmat = 2 * k_vals.reshape(n_k, 1) * fbasis.r0[0:k_max, ell].T
        fb_vals = np.zeros_like(rmat)
        ind_radial = np.sum(fbasis.k_max[0:ell])
        fb_vals[:, 0:k_max] = radial[ind_radial : ind_radial + k_max].T

        h_fb_vals = fb_vals * h_vals.reshape(n_k, 1)
        h_fb_ell = fb_vals.T @ (
            h_fb_vals * k_vals.reshape(n_k, 1) * wts.reshape(n_k, 1)
        )
        h_fb[ind_ell] = h_fb_ell
        ind_ell += 1
        if ell > 0:
            h_fb[ind_ell] = h_fb[ind_ell - 1]
            ind_ell += 1

    return h_fb
