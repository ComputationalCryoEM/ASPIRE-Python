import numpy as np
import scipy.sparse as sparse


def transform_complex_to_real(B_conj, ells):
    """
    Transforms coefficients of the matrix B (see Eq. 3) from complex
        to real. B is the linear transformation that takes FLE coefficients
        to images.

    :param B_conj: Complex conjugate of the matrix B.
    :param ells: List of ells (Bessel function orders) in this basis.
    :return: Transformed matrix.
    """
    num_basis_functions = B_conj.shape[1]
    X = np.zeros(B_conj.shape, dtype=np.float64)

    for i in range(num_basis_functions):
        ell = ells[i]
        if ell == 0:
            X[:, i] = np.real(B_conj[:, i])
        # for each ell != 0, we can populate two entries of the matrix
        # by taking the complex conjugate of the ell with the opposite sign
        if ell < 0:
            s = (-1) ** np.abs(ell)
            x0 = (B_conj[:, i] + s * B_conj[:, i + 1]) / np.sqrt(2)
            x1 = (-B_conj[:, i] + s * B_conj[:, i + 1]) / (1j * np.sqrt(2))
            X[:, i] = np.real(x0)
            X[:, i + 1] = np.real(x1)

    return X


def precomp_transform_complex_to_real(ells):

    count = len(ells)
    num_nonzero = np.sum(ells == 0) + 2 * np.sum(ells != 0)
    idx = np.zeros(num_nonzero, dtype=int)
    jdx = np.zeros(num_nonzero, dtype=int)
    vals = np.zeros(num_nonzero, dtype=np.complex128)

    k = 0
    for i in range(count):
        ell = ells[i]
        if ell == 0:
            vals[k] = 1
            idx[k] = i
            jdx[k] = i
            k = k + 1
        if ell < 0:
            s = (-1) ** np.abs(ell)

            vals[k] = 1 / np.sqrt(2)
            idx[k] = i
            jdx[k] = i
            k = k + 1

            vals[k] = s / np.sqrt(2)
            idx[k] = i
            jdx[k] = i + 1
            k = k + 1

            vals[k] = -1 / (1j * np.sqrt(2))
            idx[k] = i + 1
            jdx[k] = i
            k = k + 1

            vals[k] = s / (1j * np.sqrt(2))
            idx[k] = i + 1
            jdx[k] = i + 1
            k = k + 1

    A = sparse.csr_matrix((vals, (idx, jdx)), shape=(count, count), dtype=np.complex128)

    return A


def barycentric_interp_sparse(x, xs, ys, s):
    """
    Perform barycentric interpolation to compute values of Betas at the points
        `x`, based on their values (`ys`) at known points (`xs`).
        https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

    :param x: The target set of points at which to evaluate the functions.
    :param xs: The points at which the values of the functions are known.
    :param ys: The values of the functions at the points `xs`.
    :param s: Numsparse.
    :return: The interpolation matrix and its transpose as a 2-tuple.
    """

    n = len(x)
    m = len(xs)

    # Modify points by 2e-16 to avoid division by zero
    vals, x_ind, xs_ind = np.intersect1d(x, xs, return_indices=True, assume_unique=True)
    x[x_ind] = x[x_ind] + 2e-16

    idx = np.zeros((n, s))
    jdx = np.zeros((n, s))
    vals = np.zeros((n, s))
    xss = np.zeros((n, s))
    denom = np.zeros((n, 1))
    temp = np.zeros((n, 1))
    ws = np.zeros((n, s))
    xdiff = np.zeros(n)
    for i in range(n):

        # get a kind of blanced interval around our point
        k = np.searchsorted(x[i] < xs, True)

        idp = np.arange(k - s // 2, k + (s + 1) // 2)
        if idp[0] < 0:
            idp = np.arange(s)
        if idp[-1] >= m:
            idp = np.arange(m - s, m)
        xss[i, :] = xs[idp]
        jdx[i, :] = idp
        idx[i, :] = i

    Iw = np.ones(s, dtype=bool)
    ew = np.zeros((n, 1))
    xtw = np.zeros((n, s - 1))

    Iw[0] = False
    const = np.zeros((n, 1))
    for _ in range(s):
        ew = np.sum(-np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1)
        constw = np.exp(ew / s)
        constw = constw.reshape(-1, 1)
        const += constw
    const = const / s

    for j in range(s):
        Iw[j] = False
        xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
        ws[:, j] = 1 / np.prod(xtw, axis=1)
        Iw[j] = True

    xdiff = xdiff.flatten()
    x = x.flatten()
    temp = temp.flatten()
    denom = denom.flatten()
    for j in range(s):
        xdiff = x - xss[:, j]
        temp = ws[:, j] / xdiff
        vals[:, j] = vals[:, j] + temp
        denom = denom + temp
    vals = vals / denom.reshape(-1, 1)

    vals = vals.flatten()
    idx = idx.flatten()
    jdx = jdx.flatten()
    A = sparse.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=np.float64)
    A_T = sparse.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)

    return A, A_T
