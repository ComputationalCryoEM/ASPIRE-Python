"""
This module contains helper functions to help repeating common functions with ease.
"""


def f_flatten(mat):
    """ Accept a matrix and return a flattened vector.

        This func mimics MATLAB's behavior:
        mat = mat(:)

    """

    return mat.flatten('F')
