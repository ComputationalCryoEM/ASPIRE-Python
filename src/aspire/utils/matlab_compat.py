"""
Functions for compatibility with MATLAB behavior.
At some point when the package is full validated against MatLab, the 'order' arguments in the functions here
can be changed to 'C', and subsequently, this package deprecated altogether (i.e. the reshape/flatten methods used
directly by the caller).
"""

import numpy as np
import scipy.sparse as sparse


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
