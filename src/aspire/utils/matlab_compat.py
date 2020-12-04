"""
Functions for compatibility with MATLAB behavior.
At some point when the package is full validated against MatLab, the 'order' arguments in the functions here
can be changed to 'C', and subsequently, this package deprecated altogether (i.e. the reshape/flatten methods used
directly by the caller).
"""

import numpy as np
import scipy.sparse as sparse


def m_reshape(x, new_shape):
    # This is a somewhat round-about way of saying:
    #   return x.reshape(new_shape, order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    # Note that flattening is required before reshaping, because
    if isinstance(new_shape, tuple):
        return m_flatten(x).reshape(new_shape[::-1]).T
    else:
        return x


def m_flatten(x):
    # This is a somewhat round-about way of saying:
    #   return x.flatten(order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    return x.T.flatten()


def stable_eigsh(*args, **kwargs):
    """
    A Wrapper function to fix sign problem of eigen-vectors

    There is an ambiguous sign problem for the eigenvectors from
    scipy.sparse.linalg.eigsh function. We need to rescale the
    eigenvectors and make them consistent for repeated runs.

    :param *args: Positional arguments
    :param **kwargs: Keyword arguments
    :return: Eigenvalues and eigenvectors
    """

    d, v = sparse.linalg.eigsh(*args, **kwargs)
    # Find component index of maximum absolute value of each eigenvector
    ind_max = np.argmax(np.absolute(v), axis=0)
    # Rescale eigenvector based on sign from the component with the
    # maximum absolute value
    signs = np.array([np.sign(v[ind_max[k], k]) for k in range(len(d))])

    return d, v * signs
