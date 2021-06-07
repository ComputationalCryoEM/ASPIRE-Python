import logging
from unittest import TestCase

import numpy as np
from numpy import mgrid, sin, cos, all, asarray, ceil, log2
from pytest import raises

import warnings

from aspire.operators import wemd_norm

logger = logging.getLogger(__name__)


def _smoothed_disk_image(x, y, radius, width, height):
    (Y, X) = mgrid[:height,:width]
    ratio = ((X-x)**2 + (Y-y)**2)/(radius**2)
    return  2.0 - 2/(1 + np.exp(-ratio)) # Sigmoid funciton


def _is_monotone(seq):
    arr = asarray(seq)
    assert arr.ndim == 1
    return all(arr[1:] >= arr[:-1])


class WEMDTestCase(TestCase):
    '''Test that the WEMD distance between smoothed disks of various radii, angles and distances is monotone in the Euclidean distance of their centers.
    Note that this monotonicity isn't strictly required by the theory, but holds empirically.'''
    def test_wemd_norm(self):
        WIDTH = 64
        HEIGHT = 64
        CENTER_X = WIDTH/2
        CENTER_Y = HEIGHT/2
        LEVEL = int(ceil(log2(max(WIDTH, HEIGHT)))) + 1 # A rule of thumb for the wavelet level is to take the log2 of the side length.
        WAVELET = 'coif3'
        warnings.filterwarnings("ignore", message="Level value of .* is too high: all coefficients will experience boundary effects.")
        for radius in [1, 2, 3, 4, 5, 6, 7]: # Just some random radii
            for angle in [0.0, 0.4755, 0.6538, 1.9818, 3.0991, 4.4689, 4.9859, 5.5752]: # Just some random angles
                disks = [_smoothed_disk_image(CENTER_X + int(k*cos(angle)), CENTER_Y + int(k*sin(angle)), radius, WIDTH, HEIGHT) for k in range(0,16,2)]
                wemd_distances_along_ray = [wemd_norm(disks[0] - disk, WAVELET, LEVEL) for disk in disks]
                print(wemd_distances_along_ray)
                self.assertTrue(_is_monotone(wemd_distances_along_ray))

