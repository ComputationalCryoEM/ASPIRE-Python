"""
Utilities for arrays/n-dimensional matrices.
"""

import numpy as np
from scipy.linalg import eigh

SQRT2 = np.sqrt(2)
SQRT2_R = 1 / SQRT2


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
    """

    X = f(X)

    # Swap the last d2 axes with the first d1 axes
    dims1 = [X.ndim - d - 1 for d in range(d1 + d2)]
    dims2 = [X.ndim - d - 1 for d in list(range(d1, d1 + d2)) + list(range(d1))]
    X = np.moveaxis(X, dims1, dims2)

    X = np.conj(X)
    X = f(X)

    # Swap the next d2 axes block
    dims1 = [X.ndim - d - 1 for d in range(2 * d2)]
    dims2 = [X.ndim - d - 1 for d in list(range(d2, 2 * d2)) + list(range(d2))]
    X = np.moveaxis(X, dims1, dims2)

    X = np.conj(X)

    return X


def symmat_to_vec_iso(mat):
    """
    Isometrically maps a symmetric matrix to a packed vector

    :param mat: An array of size ...-by-N-by-N where the last two dimensions constitute symmetric or Hermitian
        matrices.
    :return: A vector of size ...-by-N*(N+1)/2 consisting of the lower triangular part of each matrix, reweighted so
        that the Frobenius inner product is mapped to the Euclidean inner product.
    """
    sz = mat.shape[:-2]
    N = mat.shape[-1]
    vec = mat.reshape(*sz, -1)
    vec[..., np.arange(0, N**2, N + 1)] *= SQRT2_R
    vec *= SQRT2
    mat = vec.reshape(*sz, N, N)
    vec = symmat_to_vec(mat)

    return vec


def vec_to_symmat_iso(vec):
    """
    Isometrically map packed vector to symmetric matrix

    :param vec: A vector of size ...-by-N*(N+1)/2 describing a symmetric (or Hermitian) matrix.
    :return: An array of size ...-by-N-by-N which indexes symmetric/Hermitian matrices that occupy the first two
        dimensions. The lower triangular parts of these matrices consists of the corresponding vectors in vec,
        reweighted so that the Euclidean inner product maps to the Frobenius inner product.
    """

    mat = vec_to_symmat(vec)
    N = mat.shape[-1]
    mat = mat_to_vec(mat)
    mat[..., np.arange(0, N**2, N + 1)] *= SQRT2
    mat *= SQRT2_R
    mat = vec_to_mat(mat)
    return mat


def symmat_to_vec(mat):
    """
    Packs a symmetric matrix into a upper triangular vector

    :param mat: An array of size ...-by-N-by-N where the first two dimensions constitute symmetric or
        Hermitian matrices.
    :return: A vector of size ...-by-N*(N+1)/2 consisting of the upper triangular part of each matrix.
    """

    N = mat.shape[-1]
    assert mat.shape[-2] == N, "Matrix must be square"

    sz = mat.shape[:-2]
    tri_indices = np.triu_indices(N)
    # Python 3.11 this can change to mat[..., *tri_indices]
    #   See PEP 646, variadics
    #   https://peps.python.org/pep-0646/#multiple-unpackings-in-a-tuple-not-allowed
    vec = mat[..., tri_indices[0], tri_indices[1]].reshape(*sz, N * (N + 1) // 2)

    return vec


def vec_to_symmat(vec):
    """
    Convert packed upper triangular vector to symmetric matrix

    :param vec: A vector of size ...-by-N*(N+1)/2 describing a symmetric (or Hermitian) matrix.
    :return: An array of size ...-by-N-by-N which indexes symmetric/Hermitian matrices that occupy the first two
        dimensions. The upper triangular parts of these matrices consists of the corresponding vectors in vec.
    """
    # TODO: Handle complex values in vec
    if np.iscomplex(vec).any():
        raise NotImplementedError("Coming soon")

    M = vec.shape[-1]
    N = int(round(np.sqrt(2 * M + 0.25) - 0.5))
    assert (
        M == 0.5 * N * (N + 1)
    ) and N != 0, "Vector must be of size N*(N+1)/2 for some N>0."

    sz = vec.shape[:-1]
    index_matrix = np.empty((N, N))
    i_upper = np.triu_indices_from(index_matrix)
    index_matrix[i_upper] = np.arange(
        M
    )  # Incrementally populate upper triangle in row major order
    index_matrix.T[i_upper] = index_matrix[i_upper]  # Copy to lower triangle

    mat = vec[..., index_matrix.astype("int")]
    mat = mat.reshape(*sz, N, N)

    return mat


def mat_to_vec(mat, is_symmat=False):
    """
    Converts a matrix into vectorized form

    :param mat: An array of size ...-by-N-by-N-by containing the matrices to be vectorized.
    :param is_symmat: Specifies whether the matrices are symmetric/Hermitian, in which case they are stored in packed
        form using symmat_to_vec (default False).
    :return: The vectorized form of the matrices, with dimension ...-by-N^2 or ...-by-N*(N+1)/2 depending on the value
        of is_symmat.
    """
    if not is_symmat:
        sz = mat.shape
        N = sz[-1]
        assert sz[-2] == N, "Matrix must be square"
        return mat.reshape(*sz[:-2], N**2)
    else:
        return symmat_to_vec(mat)


def vec_to_mat(vec, is_symmat=False):
    """
    Converts a vectorized matrix into a matrix

    :param vec: The vectorized representations. If the matrix is non-symmetric, this array has the dimensions
        ...-by-N^2, but if the matrix is symmetric, the dimensions are ...-by-N*(N+1)/2 .
    :param is_symmat: True if the vectors represent symmetric matrices (default False)
    :return: The array of size ...-by-N-by-N representing the matrices.
    """
    if not is_symmat:
        sz = vec.shape
        N = int(round(np.sqrt(sz[-1])))
        assert sz[-1] == N**2, "Vector must represent square matrix."
        return vec.reshape(*sz[:-1], N, N)
    else:
        return vec_to_symmat(vec)


def make_symmat(A):
    """
    Symmetrize a matrix

    :param A: A matrix.
    :return: The Hermitian matrix (A+A')/2.
    """
    return 0.5 * (A + A.T)


def make_psd(A):
    """
    Make a matrix positive semi-definite

    This is the simplest way by setting negative eigenvalues to zero.

    :param A: A matrix.
    :return: The positive semi-definite matrix
    """
    B = make_symmat(A)
    W, V = eigh(B)
    W[W < 0.0] = 0.0

    return V @ np.diag(W) @ np.transpose(V)


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
    assert x.shape == y.shape, "The shapes of the inputs have to match"

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
    assert x.shape == y.shape, "The shapes of the inputs have to match"

    if axes is not None:
        axes = tuple(axes)  # Unrolls any generators, like `range`.

    return np.sum(x * np.conj(y), axis=axes)


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
    A = A.reshape((sig_len, sig_len))

    dtype = A.dtype
    w, v = eigh(
        A.astype(np.float64), subset_by_index=(sig_len - 1 - k + 1, sig_len - 1)
    )

    # Arrange in descending order (flip column order in eigenvector matrix) and typecast to proper type
    w = w[::-1].astype(dtype)
    v = np.fliplr(v)

    v = v.reshape(sig_sz + (k,)).astype(dtype)
    return v, np.diag(w)


def best_rank1_approximation(A):
    """
    Computes the best rank-1 approximation of A.

    :param A: A 2D array or a 3D array where the first axis is the stack axis.
    :return: rank-1 ndarray of same size.
    """
    og_shape = A.shape

    if A.ndim == 2:
        A = A[np.newaxis]
    assert A.ndim == 3, "Array must be 2D or 3D representing a stack."

    U, S, V = np.linalg.svd(A)
    S_rank1 = np.zeros_like(A)
    S_rank1[:, 0, 0] = S[:, 0]

    return (U @ S_rank1 @ V).reshape(og_shape)


def nearest_rotations(A, allow_reflection=False):
    """
    Uses the SVD method to compute the set of nearest rotations to the set A of noisy rotations.

    Note when `allow_reflection` is `True`, results may contain reflections.

    :param A: A 2D array or a 3D array where the first axis is the stack axis.
    :param allow_reflection: Optionally allow reflections (disables correction).
    :return: ndarray of rotations of equal size to A.
    """
    og_shape = A.shape
    dtype = A.dtype

    if A.ndim == 2:
        A = A[np.newaxis]
    if A.ndim != 3 or not A.shape[1] == A.shape[2] == 3:
        raise ValueError(
            f"Array must be of shape (3, 3) or (n, 3, 3). Found shape {A.shape}."
        )

    # For the singular value decomposition A = U @ S @ VT,
    # we compute the nearest rotation matrices R = U @ VT.
    U, _, VT = np.linalg.svd(A)

    if not allow_reflection:
        # If det(U)*det(V) = -1,
        #   we want to find a pure rotation R that is closest to the
        #   preserving the 2D projection induced by the  orthogonal transform A.
        #
        #   This can be done by reflecting about the origin,
        #   then rotating around projection axis.
        #
        #     R = (U @ diag([-1,-1,-1]) @ VT) @ r_projection
        #
        # This is accomplished by the single application of d to elements of VT.
        #
        #     R = (U @ diag([-1,-1,-1]) @ VT) @ diag([-1,-1,1])
        #     R = (U * -1 @ VT) * [-1,-1,1]
        #     R =  (U @ VT) * (-1 * [-1,-1,1])
        #     R =  U @ (VT * [1,1,-1])
        d = np.array([1, 1, -1], dtype=dtype)
        neg_det_idx = np.linalg.det(U) * np.linalg.det(VT) < 0
        VT[neg_det_idx] = VT[neg_det_idx] * d

    rots = U @ VT

    return rots.reshape(og_shape)


def fix_signs(u):
    """
    Negates columns so the sign of the largest element in the column is positive.

    For complex values sign is taken as norm(x)/x, zero columns unchanged.

    Typically this is used for making eigenvectors deterministically signed.

    :param u: matrix as numpy array
    :return: matrix as numpy array
    """

    # Locate the largest element in each column
    # Internally np.absolute performs `norm` for complex values.
    index_array = np.argmax(np.absolute(u), axis=0)

    # Create array of sign corrections
    signs = np.take_along_axis(u, np.expand_dims(index_array, axis=0), axis=0).squeeze()
    _abs = np.absolute(signs)
    signs = np.divide(_abs, signs, where=_abs != 0)

    # Now we only care about the sign +1/-1.
    #  The following corrects for any numerical division noise,
    #  and also remaps 0 to +1.
    signs = np.sign(signs.real * 2 + 1)

    # Apply signs elementwise to matrix
    return u * signs
