"""
Define operation functions of block diagonal matrix needed by Cov2D analysis.
"""
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
from scipy.linalg import block_diag
from scipy.special import jv

from aspire.utils.cell import Cell2D
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.basis.basis_utils import lgwt


def blk_diag_zeros(blk_partition, dtype=None):
    """
    Build a block diagonal zero matrix

    :param blk_partition: The matrix block partition of `blk_diag` in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
    :param dtype: The data type to set precision of diagonal matrix block.
    :return: A block diagonal matrix consisting of `K` zero blocks.
    """

    if dtype is None:
        dtype = 'double'
    blk_diag = []
    for blk_sz in blk_partition:
        blk_diag.append(np.zeros(blk_sz, dtype=dtype))
    return blk_diag


def blk_diag_eye(blk_partition, dtype=None):
    """
    Build a block diagonal eye matrix

    :param blk_partition: The matrix block partition of `blk_diag` in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
    :param dtype: The data type to set the pricision of diagonal matrix block.
    :return: A block diagonal matrix consisting of `K` eye (identity) blocks.
    """
    if dtype is None:
        dtype = 'double'
    blk_diag = []
    for blk_sz in blk_partition:
        mat_temp = np.zeros(blk_sz, dtype=dtype)
        for i in range(mat_temp.shape[0]):
            mat_temp[i, i] = 1.0
        blk_diag.append(mat_temp)
    return blk_diag


def blk_diag_partition(blk_diag):
    """
    Create a partition of block diagonal matrix

    :param blk_diag: A block diagonal matrix in the form of a list. Each
        element corresponds to a diagonal block.
    :return: The matrix block partition of `blk_diag` in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
    """
    blk_partition = []
    for mat in blk_diag:
        blk_partition.append(np.shape(mat))
    return blk_partition


def blk_diag_add(blk_diag_a, blk_diag_b):
    """
    Define the element addition of block diagonal matrix

    :param blk_diag_a: The block diagonal A matrix
    :param blk_diag_b: The block diagonal B matrix
    :return: A block diagonal matrix with elements equal to corresponding
        sums from A and B elements.
    """
    if len(blk_diag_a) != len(blk_diag_b):
        raise RuntimeError('Dimensions of two block diagonal matrices are not equal!')
    elif np.allclose([np.size(mat_a) for mat_a in blk_diag_a], [np.size(mat_b) for mat_b in blk_diag_b]):
        blk_diag_c = []
        for i in range(0, len(blk_diag_a)):
            mat_c = blk_diag_a[i] + blk_diag_b[i]
            blk_diag_c.append(mat_c)
    else:
        raise RuntimeError('Elements of two block diagonal matrices have not the same size!')
    return blk_diag_c


def blk_diag_minus(blk_diag_a, blk_diag_b):
    """
    Define the element subtraction of block diagonal matrix

    :param blk_diag_a: The block diagonal A matrix
    :param blk_diag_b: The block diagonal B matrix
    :return: A block diagonal matrix with elements equal to corresponding
        subtraction of A from B elements.
    """
    if len(blk_diag_a) != len(blk_diag_b):
        raise RuntimeError('Dimensions of two block diagonal matrices are not equal!')
    elif np.allclose([np.size(mat_a) for mat_a in blk_diag_a], [np.size(mat_b) for mat_b in blk_diag_b]):
        blk_diag_c = []
        for i in range(0, len(blk_diag_a)):
            mat_c = blk_diag_a[i] - blk_diag_b[i]
            blk_diag_c.append(mat_c)
    else:
        raise RuntimeError('Elements of two block diagonal matrices have not the same size!')
    return blk_diag_c


def blk_diag_apply(blk_diag, x):
    """
    Define the apply option of a block diagonal matrix with a matrix of coefficient vectors

    :param blk_diag: The block diagonal matrix
    :param x: The coefficient matrix with each column is a coefficient vector
    :return: A matrix with new coefficient vectors
    """
    cols = np.array([np.size(mat, 1) for mat in blk_diag])

    if np.sum(cols) != np.size(x, 0):
        raise RuntimeError('Sizes of matrix `blk_diag` and `x` are not compatible.')
    rows = np.array([np.size(x, 1), ])
    cellarray = Cell2D(cols, rows, dtype=x.dtype)
    x_cell = cellarray.mat2cell(x, cols, rows)
    y = []
    for i in range(0, len(blk_diag)):
        mat = blk_diag[i] @ x_cell[i]
        y.append(mat)
    y = np.concatenate(y, axis=0)
    return y


def blk_diag_mult(blk_diag_a, blk_diag_b):
    """
    Define the matrix multiplication of two block diagonal matrices

    :param blk_diag_a: The block diagonal A matrix
    :param blk_diag_b: The block diagonal B matrix
    :return: A block diagonal matrix with elements resulted from
        the matrix multiplication of A from B block diagonal matrices.
    """

    if blk_diag_isnumeric(blk_diag_a):
        blk_diag_c = blk_diag_mult(blk_diag_b, blk_diag_a)
        return blk_diag_c

    if blk_diag_isnumeric(blk_diag_b):
        blk_diag_c = []
        for i in range(0, len(blk_diag_a)):
            mat = blk_diag_b * blk_diag_a[i]
            blk_diag_c.append(mat)
        return blk_diag_c

    if len(blk_diag_a) != len(blk_diag_b):
        raise RuntimeError('Dimensions of two block diagonal matrices are not equal!')
    elif np.allclose([np.size(mat_a, 1) for mat_a in blk_diag_a], [np.size(mat_b, 0) for mat_b in blk_diag_b]):
        blk_diag_c = []
        for i in range(0, len(blk_diag_a)):
            mat_c = blk_diag_a[i] @ blk_diag_b[i]
            blk_diag_c.append(mat_c)
        return blk_diag_c
    else:
        raise RuntimeError('Block diagonal matrix sizes are incompatible.!')


def blk_diag_norm(blk_diag):
    """
    Compute the 2-norm of a block diagonal matrix

    :param blk_diag: The block diagonal matrix
    :return: The 2-norm of the block diagonal matrix
    """
    maxvalues = []
    for i in range(0, len(blk_diag)):
        maxvalues.append(norm(blk_diag[i],2))
    max_results = max(maxvalues)
    return max_results


def blk_diag_solve(blk_diag, y):
    """
    Solve a linear system involving a block diagonal matrix

    :param blk_diag: The block diagonal matrix
    :param y: The right-hand side in the linear system.  May be a matrix
        consisting of  coefficient vectors, in which case each column is
        solved for separately.
    :return: The result of solving the linear system formed by the matrix.
    """
    rows = np.array([np.size(mat_a, 0) for mat_a in blk_diag])
    if sum(rows) != np.size(y, 0):
        raise RuntimeError('Sizes of matrix `blk_diag` and `y` are not compatible.')
    cols = np.array([np.size(y, 1)])
    cellarray = Cell2D(rows, cols, dtype=y.dtype)
    y = cellarray.mat2cell(y, rows, cols)
    x = []
    for i in range(0,len(blk_diag)):
        x.append(solve(blk_diag[i], y[i]))
    x = np.concatenate(x, axis=0)
    return x


def blk_diag_to_mat(blk_diag):
    """
    Convert list representation of block diagonal matrix into full matrix

    :param blk_diag: The block diagonal matrix

    :return: The block diagonal matrix including the zero elements of
        non-diagonal blocks.
    """
    mat = block_diag(blk_diag)
    return mat


def mat_to_blk_diag(mat, blk_partition):
    """
    Convert full block diagonal matrix into list representation
    :param mat; The full block diagonal matrix including the zero elements of
        non-diagonal blocks.
    :param blk_partition: The matrix block partition of `blk_diag` in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.

    :return: The block diagonal matrix in the list representation of diagonal blocks
    """
    rows = blk_partition[:, 0]
    cols = blk_partition[:, 1]
    cellarray = Cell2D(rows, cols, dtype=mat.dtype)
    blk_diag = cellarray.mat2blk_diag(mat, rows, cols)
    return blk_diag


def blk_diag_transpose(blk_diag):
    """
    Get the transpose matrix of a block diagonal matrix

    :param blk_diag: The block diagonal matrix
    :return: The corresponding transpose form of a block diagonal matrix
    """

    blk_diag_t = []
    for i in range(0, len(blk_diag)):
        blk_diag_t.append(blk_diag[i].T)
    return blk_diag_t


def blk_diag_compatible(blk_diag_a, blk_diag_b):
    """
    Check whether two block diagonal matrices are compatible or not

    :param blk_diag_a: The block diagonal A matrix
    :param blk_diag_b: The block diagonal B matrix
    :return: `True` value if all of shapes in two list representations of
        block diagonal matrices are the same, otherwise return `False`.
    """
    if len(blk_diag_a) != len(blk_diag_b):
        return False
    elif np.allclose([np.size(mat_a, 1) for mat_a in blk_diag_a],
                     [np.size(mat_b, 0) for mat_b in blk_diag_b]):
        return True
    else:
        return False


def blk_diag_isnumeric(x):
    """
    Check a block diag matrix is numeric or not
    """
    try:
        return 0 == x*0
    except:
        return False


def filter_to_fb_mat(h_fun, fbasis):
    """
    Convert a nonradial function in k space into a basis representation

    :param h_fun: The function form in k space
    :param fbasis: The basis object for expanding
    :return: a matrix representation using the `fbasis` expansion
    """
    if not isinstance(fbasis, FFBBasis2D):
            raise NotImplementedError('Currently only fast FB method is supported')
    # Set same dimensions as basis object
    n_k = int(np.ceil(4 * fbasis.rcut * fbasis.kcut))
    n_theta = np.ceil(16 * fbasis.kcut * fbasis.rcut)
    n_theta = int((n_theta + np.mod(n_theta, 2)) / 2)
    # get 2D grid in polar coordinate
    k_vals, wts = lgwt(n_k, 0, 0.5)
    k, theta = np.meshgrid(k_vals, np.arange(n_theta) * 2 * np.pi / (2 * n_theta), indexing='ij')
    # Get function values in polar 2D grid and average out angle contribution
    omegax = k*np.cos(theta)
    omegay = k*np.sin(theta)
    omega = 2 * np.pi * np.vstack((omegax.flatten('C'), omegay.flatten('C')))
    h_vals2d = h_fun(omega).reshape(n_k, n_theta)
    h_vals = np.sum(h_vals2d, axis=1)/n_theta

    # Represent 1D function values in fbasis
    h_fb = []
    ind = 0
    for ell in range(0, fbasis.ell_max+1):
        k_max = fbasis.k_max[ell]
        rmat = 2*k_vals.reshape(n_k, 1)*fbasis.r0[0:k_max, ell].T
        fb_vals = np.zeros_like(rmat)
        for ik in range(0, k_max):
            fb_vals[:, ik] = jv(ell, rmat[:, ik])
        fb_nrms = 1/np.sqrt(2)*abs(jv(ell+1, fbasis.r0[0:k_max, ell].T))/2
        fb_vals = fb_vals/fb_nrms
        h_fb_vals = fb_vals*h_vals.reshape(n_k, 1)
        h_fb_ell = fb_vals.T @ (h_fb_vals*k_vals.reshape(n_k, 1)*wts.reshape(n_k, 1))
        h_fb.append(h_fb_ell)
        ind = ind+1
        if ell > 0:
            h_fb.append(h_fb[ind-1])
            ind = ind+1

    return h_fb
    # return fun2fb_mat(k_vals, wts, h_vals, fbasis)


def fun2fb_mat(k_vals, wts, h_vals, fbasis):
    """
    Convert 1D array of function values in k space into a basis representation

    :param k_vals: The k values
    :param wts: The weights for each k value
    :param h_vals: The function values in k space
    :param fbasis: The basis object for expanding
    :return: a matrix representation using the `fbasis` expansion
    """
    # Set same dimensions as basis object
    n_k = k_vals.size

    h_fb = []
    ind = 0
    for ell in range(0, fbasis.ell_max+1):
        k_max = fbasis.k_max[ell]
        rmat = 2*k_vals.reshape(n_k, 1)*fbasis.r0[0:k_max, ell].T
        fb_vals = np.zeros_like(rmat)
        for ik in range(0, k_max):
            fb_vals[:, ik] = jv(ell, rmat[:, ik])
        fb_nrms = 1/np.sqrt(2)*abs(jv(ell+1, fbasis.r0[0:k_max, ell].T))/2
        fb_vals = fb_vals/fb_nrms
        h_fb_vals = fb_vals*h_vals.reshape(n_k, 1)
        h_fb_ell = fb_vals.T @ (h_fb_vals*k_vals.reshape(n_k, 1)*wts.reshape(n_k, 1))
        h_fb.append(h_fb_ell)
        ind = ind+1
        if ell > 0:
            h_fb.append(h_fb[ind-1])
            ind = ind+1

    return h_fb
