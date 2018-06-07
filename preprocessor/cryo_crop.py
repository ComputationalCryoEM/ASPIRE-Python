# converted from MATLAB func "cryo_crop.m"
import math
import numpy


def cryo_crop_stack(mat, n, is_stack=False, fill_value=0):
    """
        Reduce the size of the 1d array, square or cube m by cropping (or
        increase the size by padding with fill_value, by default zero) to a final
        size of [n, n] or [n, n, n]. This is the analogue of down-sample, but
        it doesn't change magnification.

        If m is 2-dimensional and n is a vector, m is cropped to n=[nx ny].

        The function handles odd and even-sized arrays correctly  The center of
        an odd array is taken to be at (n+1)/2, and an even array is n/2+1.

        If the flag isstack = 1 then a 3D array m is treated as a stack of 2D
        images, and each image is cropped to n x n.

        For 2D images, the input image doesn't have to be square.
        The result is double if fillval is double; by default the result is
        single.​
    """

    num_dimensions = len(mat.shape)
    if num_dimensions == 2 and 1 in mat.shape:
        num_dimensions = 1

    if num_dimensions == 1:
        mat = numpy.reshape(mat, [mat.size, 1])  # force a column vector
        ns = math.floor(mat.size / 2) - math.floor(n / 2)  # shift term for scaling down
        if ns >= 0:  # cropping
            result_mat = mat[ns: ns + n]

        else:  # padding
            result_mat = fill_value * numpy.ones([n, 1])
            result_mat[-ns: mat.size - ns] = mat

    elif num_dimensions == 2:
        nx, ny = mat.shape
        nsx = math.floor(nx / 2) - math.floor(n / 2)  # shift term for scaling down
        nsy = math.floor(ny / 2) - math.floor(n / 2)

        if nsx >= 0:  # cropping
            result_mat = mat[nsx: nsx + n, nsy: nsy + n]

        else:  # padding
            result_mat = fill_value * numpy.ones([n, n])
            result_mat[-nsx: nx - nsx, -nsy: ny - nsy] = mat

    elif num_dimensions == 3:  # m is 3D
        raise Exception(f'not implemented yet!')

    else:
        raise Exception(f"Can't crop! number of dimensions is too big! ({num_dimensions}).")

    return result_mat
