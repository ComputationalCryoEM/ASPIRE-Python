import numpy as np

from aspire.operators import BlkDiagMatrix


# TODO, Generalize
def filter_to_basis_mat(h_fun, basis):
    """
    Convert a nonradial function in k space into a basis representation.

    :param h_fun: The function form in k space.
    :param basis: The basis object for expanding.

    :return: a BlkDiagMatrix instance representation using the
        `basis` expansion.
    """

    # These form a circular dependence, import locally until time to clean up.
    from aspire.basis import SteerableBasis2D
    from aspire.basis.basis_utils import lgwt

    if not isinstance(basis, SteerableBasis2D):
        raise NotImplementedError(
            "Currently only subclasses of SteerableBasis2D supported."
        )
    # Set same dimensions as basis object
    n_k = basis.n_r
    n_theta = basis.n_theta
    radial = basis.get_radial()

    # get 2D grid in polar coordinate
    k_vals, wts = lgwt(n_k, 0, 0.5, dtype=basis.dtype)
    k, theta = np.meshgrid(
        k_vals, np.arange(n_theta) * 2 * np.pi / (2 * n_theta), indexing="ij"
    )

    # Get function values in polar 2D grid and average out angle contribution
    omegax = k * np.cos(theta)
    omegay = k * np.sin(theta)
    omega = 2 * np.pi * np.vstack((omegax.flatten("C"), omegay.flatten("C")))
    h_vals2d = h_fun(omega).reshape(n_k, n_theta).astype(basis.dtype)
    h_vals = np.sum(h_vals2d, axis=1) / n_theta

    # Represent 1D function values in basis
    h_basis = BlkDiagMatrix.empty(2 * basis.ell_max + 1, dtype=basis.dtype)
    ind_ell = 0
    for ell in range(0, basis.ell_max + 1):
        k_max = basis.k_max[ell]
        rmat = 2 * k_vals.reshape(n_k, 1) * basis.r0[ell][0:k_max].T
        basis_vals = np.zeros_like(rmat)
        ind_radial = np.sum(basis.k_max[0:ell])
        basis_vals[:, 0:k_max] = radial[ind_radial : ind_radial + k_max].T
        h_basis_vals = basis_vals * h_vals.reshape(n_k, 1)
        h_basis_ell = basis_vals.T @ (
            h_basis_vals * k_vals.reshape(n_k, 1) * wts.reshape(n_k, 1)
        )
        h_basis[ind_ell] = h_basis_ell
        ind_ell += 1
        if ell > 0:
            h_basis[ind_ell] = h_basis[ind_ell - 1]
            ind_ell += 1

    return h_basis
