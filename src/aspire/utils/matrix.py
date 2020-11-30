"""
Utilties for arrays/n-dimensional matrices.
"""

import numpy as np
from scipy.linalg import eigh

from aspire.utils import ensure
from aspire.utils.matlab_compat import m_reshape

SQRT2 = np.sqrt(2)
SQRT2_R = 1 / SQRT2


def unroll_dim(X, dim):
    # TODO: dim is still 1-indexed like in MATLAB to reduce headaches for now
    # TODO: unroll/roll are great candidates for a context manager since they're always used in conjunction.
    dim = dim - 1
    old_shape = X.shape
    new_shape = old_shape[:dim]

    new_shape += (-1,)

    Y = m_reshape(X, new_shape)

    removed_dims = old_shape[dim:]

    return Y, removed_dims


def roll_dim(X, dim):
    # TODO: dim is still 1-indexed like in MATLAB to reduce headaches for now
    old_shape = X.shape
    new_shape = old_shape[:-1] + dim
    Y = m_reshape(X, new_shape)
    return Y


def im_to_vec(im):
    """
    Roll up images into vectors
    :param im: An N-by-N-by-... array.
    :return: An N^2-by-... array.
    """
    shape = im.shape
    ensure(im.ndim >= 2, "Array should have at least 2 dimensions")
    ensure(shape[0] == shape[1], "Array should have first 2 dimensions identical")

    return m_reshape(im, (shape[0] ** 2,) + (shape[2:]))


def vol_to_vec(X):
    """
    Roll up volumes into vectors
    :param X: N-by-N-by-N-by-... array.
    :return: An N^3-by-... array.
    """
    shape = X.shape
    ensure(X.ndim >= 3, "Array should have at least 3 dimensions")
    ensure(
        shape[0] == shape[1] == shape[2],
        "Array should have first 3 dimensions identical",
    )

    return m_reshape(X, (shape[0] ** 3,) + (shape[3:]))


def vec_to_im(X):
    """
    Unroll vectors to images
    :param X: N^2-by-... array.
    :return: An N-by-N-by-... array.
    """
    shape = X.shape
    N = round(shape[0] ** (1 / 2))
    ensure(N ** 2 == shape[0], "First dimension of X must be square")

    return m_reshape(X, (N, N) + (shape[1:]))


def vec_to_vol(X):
    """
    Unroll vectors to volumes
    :param X: N^3-by-... array.
    :return: An N-by-N-by-N-by-... array.
    """
    shape = X.shape
    N = round(shape[0] ** (1 / 3))
    ensure(N ** 3 == shape[0], "First dimension of X must be cubic")

    return m_reshape(X, (N, N, N) + (shape[1:]))


def vecmat_to_volmat(X):
    """
    Roll up vector matrices into volume matrices
    :param X: A vector matrix of size L1^3-by-L2^3-by-...
    :return: A volume "matrix" of size L1-by-L1-by-L1-by-L2-by-L2-by-L2-by-...
    """
    # TODO: Use context manager?
    shape = X.shape
    ensure(X.ndim >= 2, "Array should have at least 2 dimensions")

    L1 = round(shape[0] ** (1 / 3))
    L2 = round(shape[1] ** (1 / 3))

    ensure(L1 ** 3 == shape[0], "First dimension of X must be cubic")
    ensure(L2 ** 3 == shape[1], "Second dimension of X must be cubic")

    return m_reshape(X, (L1, L1, L1, L2, L2, L2) + (shape[2:]))


def volmat_to_vecmat(X):
    """
    Unroll volume matrices to vector matrices
    :param X: A volume "matrix" of size L1-by-L1-by-L1-by-L2-by-L2-by-L2-by-...
    :return: A vector matrix of size L1^3-by-L2^3-by-...
    """
    # TODO: Use context manager?
    shape = X.shape
    ensure(X.ndim >= 6, "Array should have at least 6 dimensions")
    ensure(shape[0] == shape[1] == shape[2], "Dimensions 1-3 should be identical")
    ensure(shape[3] == shape[4] == shape[5], "Dimensions 4-6 should be identical")

    l1 = shape[0]
    l2 = shape[3]

    return m_reshape(X, (l1 ** 3, l2 ** 3) + (shape[6:]))


def mdim_mat_fun_conj(X, d1, d2, f):
    """
    Conjugate a multidimensional matrix using a linear mapping
    :param X: An N_1-by-...-by-N_d1-by-N_1...-by-N_d1-by-... array, with the first 2*d1 dimensions corresponding to
        matrices with columns and rows of dimension d1.
    :param d1: The dimension of the input matrix X
    :param d2: The dimension of the output matrix Y
    :param f: A function handle of a linear map that takes an array of size N_1-by-...-by-N_d1-by-... and returns an
        array of size M_1-by-...-by-M_d2-by-... .
    :return: An array of size M_1-by-...-by-M_d2-by-M_1-by-...-by-M_d2-by-... resulting from applying fun to the rows
        and columns of the multidimensional matrix X.

    TODO: Very complicated to wrap head around this one!
    """
    X, sz_roll = unroll_dim(X, 2 * d1 + 1)
    X = f(X)

    # Swap the first d2 axes block of X with the next d1 axes block
    X = np.moveaxis(X, list(range(d1 + d2)), list(range(d1, d1 + d2)) + list(range(d1)))

    X = np.conj(X)
    X = f(X)

    # Swap the first d2 axes block of X with the next d2 axes block
    X = np.moveaxis(X, list(range(2 * d2)), list(range(d2, 2 * d2)) + list(range(d2)))

    X = np.conj(X)
    X = roll_dim(X, sz_roll)

    return X


def symmat_to_vec_iso(mat):
    """
    Isometrically maps a symmetric matrix to a packed vector
    :param mat: An array of size N-by-N-by-... where the first two dimensions constitute symmetric or Hermitian
        matrices.
    :return: A vector of size N*(N+1)/2-by-... consisting of the lower triangular part of each matrix, reweighted so
        that the Frobenius inner product is mapped to the Euclidean inner product.
    """
    mat, sz_roll = unroll_dim(mat, 3)
    N = mat.shape[0]
    mat = mat_to_vec(mat)
    mat[np.arange(0, N ** 2, N + 1)] *= SQRT2_R
    mat *= SQRT2
    mat = vec_to_mat(mat)
    mat = roll_dim(mat, sz_roll)
    vec = symmat_to_vec(mat)

    return vec


def vec_to_symmat_iso(vec):
    """
    Isometrically map packed vector to symmetric matrix
    :param vec: A vector of size N*(N+1)/2-by-... describing a symmetric (or Hermitian) matrix.
    :return: An array of size N-by-N-by-... which indexes symmetric/Hermitian matrices that occupy the first two
        dimensions. The lower triangular parts of these matrices consists of the corresponding vectors in vec,
        reweighted so that the Euclidean inner product maps to the Frobenius inner product.
    """
    mat = vec_to_symmat(vec)
    mat, sz_roll = unroll_dim(mat, 3)
    N = mat.shape[0]
    mat = mat_to_vec(mat)
    mat[np.arange(0, N ** 2, N + 1)] *= SQRT2
    mat *= SQRT2_R
    mat = vec_to_mat(mat)
    mat = roll_dim(mat, sz_roll)
    return mat


def symmat_to_vec(mat):
    """
    Packs a symmetric matrix into a lower triangular vector
    :param mat: An array of size N-by-N-by-... where the first two dimensions constitute symmetric or
        Hermitian matrices.
    :return: A vector of size N*(N+1)/2-by-... consisting of the lower triangular part of each matrix.

    Note that a lot of acrobatics happening here (swapaxes/triu instead of tril etc.) are so that we can get
    column-major ordering of elements (to get behavior consistent with MATLAB), since masking in numpy only returns
    data in row-major order.
    """
    N = mat.shape[0]
    ensure(mat.shape[1] == N, "Matrix must be square")

    mat, sz_roll = unroll_dim(mat, 3)
    triu_indices = np.triu_indices(N)
    vec = mat.swapaxes(0, 1)[triu_indices]
    vec = roll_dim(vec, sz_roll)

    return vec


def vec_to_symmat(vec):
    """
    Convert packed lower triangular vector to symmetric matrix
    :param vec: A vector of size N*(N+1)/2-by-... describing a symmetric (or Hermitian) matrix.
    :return: An array of size N-by-N-by-... which indexes symmetric/Hermitian matrices that occupy the first two
        dimensions. The lower triangular parts of these matrices consists of the corresponding vectors in vec.
    """
    # TODO: Handle complex values in vec
    if np.iscomplex(vec).any():
        raise NotImplementedError("Coming soon")

    # M represents N(N+1)/2
    M = vec.shape[0]
    N = int(round(np.sqrt(2 * M + 0.25) - 0.5))
    ensure(
        (M == 0.5 * N * (N + 1)) and N != 0,
        "Vector must be of size N*(N+1)/2 for some N>0.",
    )

    vec, sz_roll = unroll_dim(vec, 2)
    index_matrix = np.empty((N, N))
    i_upper = np.triu_indices_from(index_matrix)
    index_matrix[i_upper] = np.arange(
        M
    )  # Incrementally populate upper triangle in row major order
    index_matrix.T[i_upper] = index_matrix[i_upper]  # Copy to lower triangle

    mat = vec[index_matrix.flatten("F").astype("int")]
    mat = m_reshape(mat, (N, N) + mat.shape[1:])
    mat = roll_dim(mat, sz_roll)

    return mat


def mat_to_vec(mat, is_symmat=False):
    """
    Converts a matrix into vectorized form
    :param mat: An array of size N-by-N-by-... containing the matrices to be vectorized.
    :param is_symmat: Specifies whether the matrices are symmetric/Hermitian, in which case they are stored in packed
        form using symmat_to_vec (default False).
    :return: The vectorized form of the matrices, with dimension N^2-by-... or N*(N+1)/2-by-... depending on the value
        of is_symmat.
    """
    if not is_symmat:
        sz = mat.shape
        N = sz[0]
        ensure(sz[1] == N, "Matrix must be square")
        return m_reshape(mat, (N ** 2,) + sz[2:])
    else:
        return symmat_to_vec(mat)


def vec_to_mat(vec, is_symmat=False):
    """
    Converts a vectorized matrix into a matrix
    :param vec: The vectorized representations. If the matrix is non-symmetric, this array has the dimensions
        N^2-by-..., but if the matrix is symmetric, the dimensions are N*(N+1)/2-by-... .
    :param is_symmat: True if the vectors represent symmetric matrices (default False)
    :return: The array of size N-by-N-by-... representing the matrices.
    """
    if not is_symmat:
        sz = vec.shape
        N = int(round(np.sqrt(sz[0])))
        ensure(sz[0] == N ** 2, "Vector must represent square matrix.")
        return m_reshape(vec, (N, N) + sz[1:])
    else:
        return vec_to_symmat(vec)


def make_symmat(A):
    """
    Symmetrize a matrix
    :param A: A matrix.
    :return: The Hermitian matrix (A+A')/2.
    """
    return 0.5 * (A + A.T)


def anorm(x, axes=None):
    """
    Calculate array norm along given axes
    :param x: An array of arbitrary size and shape.
    :param axes: The axis along which to compute the norm. If None, the norm is calculated along all axes.
    :return: The Euclidean (l^2) norm of x along specified axes.
    """
    if axes is None:
        norm = np.linalg.norm(x)
    else:
        axes = tuple(axes)  # Unrolls any generators, like `range`.
        norm = np.sqrt(ainner(x, x, axes=axes))
    return norm


def acorr(x, y, axes=None):
    """
    Calculate array correlation along given axes
    :param x: An array of arbitrary shape
    :param y: An array of same shape as x
    :param axes: The axis along which to compute the correlation. If None, the correlation is calculated along all axes.
    :return: The correlation of x along specified axes.
    """
    ensure(x.shape == y.shape, "The shapes of the inputs have to match")

    if axes is None:
        axes = range(x.ndim)
    return ainner(x, y, axes) / (anorm(x, axes) * anorm(y, axes))


def ainner(x, y, axes=None):
    """
    Calculate array inner product along given axes
    :param x: An array of arbitrary shape
    :param y: An array of same shape as x
    :param axes: The axis along which to compute the inner product. If None, the product is calculated along all axes.
    :return:
    """
    ensure(x.shape == y.shape, "The shapes of the inputs have to match")

    if axes is not None:
        axes = tuple(axes)  # Unrolls any generators, like `range`.

    return np.sum(x * y, axis=axes)


def eigs(A, k):
    """
    Multidimensional partial eigendecomposition
    :param A: An array of size `sig_sz`-by-`sig_sz`, where `sig_sz` is a size containing d dimensions.
        The array represents a matrix with d indices for its rows and columns.
    :param k: The number of eigenvalues and eigenvectors to calculate (default 6).
    :return: A 2-tuple of values
        V: An array of eigenvectors of size `sig_sz`-by-k.
        D: A matrix of size k-by-k containing the corresponding eigenvalues in the diagonals.
    """
    sig_sz = A.shape[: int(A.ndim / 2)]
    sig_len = np.prod(sig_sz)
    A = m_reshape(A, (sig_len, sig_len))

    dtype = A.dtype
    w, v = eigh(A.astype(np.float64), eigvals=(sig_len - 1 - k + 1, sig_len - 1))

    # Arrange in descending order (flip column order in eigenvector matrix) and typecast to proper type
    w = w[::-1].astype(dtype)
    v = np.fliplr(v)

    v = m_reshape(v, sig_sz + (k,)).astype(dtype)

    return v, np.diag(w)
