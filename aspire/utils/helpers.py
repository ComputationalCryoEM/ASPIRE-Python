import os

import numpy as np

from aspire.class_avrages.helpers import image_grid


def cart2rad(n):
    """ Compute the radii corresponding to the points of a cartesian grid of size NxN points
        XXX This is a name for this function. """

    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def get_file_type(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    return os.path.splitext(file_path)[1]

