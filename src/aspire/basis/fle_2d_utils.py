import numpy as np

from aspire.numeric import sparse, xp


def transform_complex_to_real(B, ells):
    """
    Transforms coefficients of the matrix B (see Eq. 3) from complex
        to real. B is the linear transformation that takes FLE coefficients
        to images.

    :param B: Complex matrix B.
    :param ells: List of ells (Bessel function orders) in this basis.
    :return: Transformed matrix.
    """
    num_basis_functions = B.shape[1]
    X = np.zeros(B.shape, dtype=np.float64)

    for i in range(num_basis_functions):
        ell = ells[i]
        if ell == 0:
            X[:, i] = np.real(B[:, i])
        # for each ell != 0, we can populate two entries of the matrix
        # by taking the complex conjugate of the ell with the opposite sign
        if ell < 0:
            s = (-1) ** np.abs(ell)
            x0 = (B[:, i] + s * B[:, i + 1]) / np.sqrt(2)
            x1 = (-B[:, i] + s * B[:, i + 1]) / (1j * np.sqrt(2))
            X[:, i] = np.real(x0)
            X[:, i + 1] = np.real(x1)

    return X


def precomp_transform_complex_to_real(ells):
    """
    Returns a sparse matrix that transforms coefficients into the complex
        representation of the basis to coefficients in the real
        representation of the basis. See Remark 1.1 of Marshall, Mickelin,
        and Singer.

    :param ells: The list of integer Bessel function orders.
    :return: Sparse complex to real transformation matrix.
    """
    count = len(ells)
    num_nonzero = np.sum(ells == 0) + 2 * np.sum(ells != 0)
    idx = np.zeros(num_nonzero, dtype=int)
    jdx = np.zeros(num_nonzero, dtype=int)
    vals = np.zeros(num_nonzero, dtype=np.complex128)

    k = 0
    for i in range(count):
        ell = ells[i]
        # ell = 0 is a special case (DC component)
        if ell == 0:
            vals[k] = 1
            idx[k] = i
            jdx[k] = i
            k = k + 1
        # Only branch the case ell < 0 and also update -ell
        # via complex conjugation
        if ell < 0:
            s = (-1) ** np.abs(ell)

            # positive ell
            vals[k] = 1 / np.sqrt(2)
            idx[k] = i
            jdx[k] = i
            k = k + 1

            # positive ell
            vals[k] = s / np.sqrt(2)
            idx[k] = i
            jdx[k] = i + 1
            k = k + 1

            # negative ell
            vals[k] = -1 / (1j * np.sqrt(2))
            idx[k] = i + 1
            jdx[k] = i
            k = k + 1

            # negative ell
            vals[k] = s / (1j * np.sqrt(2))
            idx[k] = i + 1
            jdx[k] = i + 1
            k = k + 1

    A = sparse.csr_matrix(
        (xp.asarray(vals), (xp.asarray(idx), xp.asarray(jdx))),
        shape=(count, count),
        dtype=np.complex128,
    )

    return A.conjugate()


def barycentric_interp_sparse(target_points, known_points, numsparse):
    """
    Returns the sparse matrices that perform barycentric interpolation to compute values
        of Betas at the points `target_points` at known points `known_points`, and the transpose
        of this operation. For each target point in `target_points`, only `numsparse` centered
        source points from `known_points` around the target point are used.

        Performed via the method described in

        "Barycentric Lagrange Interpolation", Jean-Paul Berrut and Lloyd Trefethen.
        SIAM Review 2004 46:3, 501-517
        https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

    :param target_points: The target set of points at which to evaluate the functions.
    :param known_points: The points at which the values of the functions are known.
    :param numsparse: Number of points used for interpolation around each target point.
    :return: The interpolation matrix and its transpose as a 2-tuple.
    """

    n = len(target_points)
    m = len(known_points)

    # Modify points by 2e-16 to avoid division by zero
    vals, x_ind, xs_ind = np.intersect1d(
        target_points, known_points, return_indices=True, assume_unique=True
    )
    target_points[x_ind] = target_points[x_ind] + 2e-16

    idx = np.zeros((n, numsparse))
    jdx = np.zeros((n, numsparse))
    vals = np.zeros((n, numsparse))
    xss = np.zeros((n, numsparse))
    denom = np.zeros((n, 1))
    temp = np.zeros((n, 1))
    ws = np.zeros((n, numsparse))
    xdiff = np.zeros(n)

    # loop over target points
    for i in range(n):
        # choose `numsparse` source points centered around each target point
        # in order to apply a sparse barycentric interpolation to this target
        # point
        k = np.searchsorted(target_points[i] < known_points, True)

        idp = np.arange(k - numsparse // 2, k + (numsparse + 1) // 2)
        if idp[0] < 0:
            idp = np.arange(numsparse)
        if idp[-1] >= m:
            idp = np.arange(m - numsparse, m)
        # xss stores the values from `known_points` used for interpolation on the i'th
        # target point
        xss[i, :] = known_points[idp]
        jdx[i, :] = idp
        idx[i, :] = i

    # Auxiliary vector for computing products of expressions of the form  xss[:,i] - xss[:,j]
    # in order not to include xss[:,i] - xss[:,i] = 0. Will be all ones except for the index i
    # not to include in the running product. this index is updated in the loop
    Iw = np.ones(numsparse, dtype=bool)
    ew = np.zeros((n, 1))
    xtw = np.zeros((n, numsparse - 1))

    Iw[0] = False
    const = np.zeros((n, 1))

    for _ in range(numsparse):
        ew = np.sum(-np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1)
        # normalization constant
        constw = np.exp(ew / numsparse)
        constw = constw.reshape(-1, 1)
        const += constw
    # this normalization constant prevents numerical issues and cancels out in the end
    # not included in return result
    const = const / numsparse

    for j in range(numsparse):
        Iw[j] = False
        # compute the denominator in Eq 3.2 of Berrut and Trefethen
        xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
        ws[:, j] = 1 / np.prod(xtw, axis=1)
        Iw[j] = True

    xdiff = xdiff.flatten()
    target_points = target_points.flatten()
    temp = temp.flatten()
    denom = denom.flatten()

    for j in range(numsparse):
        # xdiff[i] is the i'th target point minus the j'th source point for that target pt
        # see the denominator in Eq. 3.3 of Berrut and Trefethen
        xdiff = target_points - xss[:, j]
        temp = ws[:, j] / xdiff
        # vals[:,j] = (1/const)*w_j/(x[i] - xs[j]), with the notation in Eq. 3.3
        vals[:, j] = vals[:, j] + temp
        denom = denom + temp

    # Eq 4.2
    # note that const cancels in numerator and denominator
    vals = vals / denom.reshape(-1, 1)

    vals = xp.array(vals.flatten())
    idx = xp.array(idx.flatten())
    jdx = xp.array(jdx.flatten())
    # A is the linear operator mapping the function values from the fixed source
    # points to the fixed target points.
    # A(i,j) = \ell(x[i] ) w_j/(x[i] - xs[j]), with the notation in Eq. 3.3
    A = sparse.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=np.float64)
    A_T = sparse.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)

    return A, A_T
