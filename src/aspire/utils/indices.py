"""
Utility functions for common index arithmetic.
"""

import numpy as np
import random

def random_tri_indices(nsamples, n, k=0, m=None,
                       triu_or_tril='triu', seed=None):
    """
    Returns random nsamples of indices from a trianglular matrix.

    Indices are returned as a list of 2-tuples.

    :param nsamples: Number of samples to return.
    :param n: The number of rows of the matrix.
    :param k: Optional, diagonal offset; defaults 0.
    :param m: Optional, number of cols of matrix; defaults=n.
    :param triu_or_tril: Optional, `triu` or `tril` indices; defaults `triu`.
    :param seed: Optional, specify rng seed, note resets random state.

    :return: Random sample of indices as list of 2-tuples.
    """

    if triu_or_tril == 'triu':
        ind_f = np.triu_indices
    elif triu_or_tril == 'tril':
        ind_f = np.tril_indices
    else:
        raise RuntimeError('triu_or_tril must be one of `triu` or `tril`.')

    if seed is not None:
        random.seed(seed)

    tri_population = np.dstack(ind_f(n, k, m))[0].tolist()
    return random.sample(tri_population, nsamples)
