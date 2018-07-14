"""
This module contains helper functions to help repeating common functions with ease.
"""
from preprocessor.exceptions import DimensionsIncompatible


class TupleCompare(object):
    """ Helper class to compare between all members of tuples. """

    @classmethod
    def validate_same_length(cls, a, b):
        if len(a) != len(b):
            raise DimensionsIncompatible("Can't compare tuples of different length")

    @classmethod
    def gt(cls, a, b, eq=False):
        cls.validate_same_length(a, b)
        if eq is True:
            return all([i >= j for i, j in zip(a, b)])
        else:
            return all([i > j for i, j in zip(a, b)])

    @classmethod
    def lt(cls, a, b, eq=False):
        cls.validate_same_length(a, b)
        if eq is True:
            return all([i <= j for i, j in zip(a, b)])
        else:
            return all([i < j for i, j in zip(a, b)])

    @classmethod
    def eq(cls, a, b):
        cls.validate_same_length(a, b)
        return all([i == j for i, j in zip(a, b)])


def f_flatten(mat):
    """ Accept a matrix and return a flattened vector.

        This func mimics MATLAB's behavior:
        mat = mat(:)

    """

    return mat.flatten('F')
