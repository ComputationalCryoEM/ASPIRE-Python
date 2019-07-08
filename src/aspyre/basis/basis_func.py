"""
Define functions for Fourier-Bessel (2D), Spherical Fourier-Bessel (3D) objects.
"""

import sys

import logging
import numpy as np
from numpy import pi, log, exp, diff
from scipy.special import jv, lpmv

from aspyre.utils import ensure
from aspyre.utils.math import grid_2d, grid_3d


logger = logging.getLogger(__name__)


def check_besselj_zeros(nu, z):
    dz = np.diff(z)
    ddz = np.diff(dz)

    result = True
    result = result and all(np.isreal(z))
    result = result and z[0] > 0
    result = result and all(dz > 3)

    if nu >= 0.5:
        result = result and all(ddz < 16 * np.spacing(z[1:-1]))
    else:
        result = result and all(ddz > -16 * np.spacing(z[1:-1]))

    return result


def besselj_newton(nu, z0, max_iter=10):
    z = z0

    # Factor worse than machine precision
    c = 8

    for i in range(max_iter):
        # Calculate values and derivatives at z
        f = jv(nu, z)
        fp = jv(nu - 1, z) - nu * f / z

        # Update zeros
        dz = - f / fp
        z = z + dz

        # Check for convergence
        if all(np.abs(dz) < c * np.spacing(z)):
            break

        # If we're not converging yet, start relaxing convergence criterion
        if i >= 6:
            c *= 2

    return z


def sph_bessel(ell, r):
    """
    Compute spherical Bessel function values.

    :param ell: The order of the spherical Bessel function.
    :param r: The coordinates where the function is to be evaluated.
    :return: The value of j_ell at r.
    """
    scalar = np.isscalar(r)
    len_r = 1 if scalar else len(r)

    j = np.zeros(len_r)
    j[r == 0] = 1 if ell == 0 else 0

    r_mask = r != 0
    j[r_mask] = np.sqrt(pi / (2 * r[r_mask])) * jv(ell + 0.5, r[r_mask])

    return np.asscalar(j) if scalar else j


def norm_assoc_legendre(j, m, x):
    """
    Evaluate the normalized associated Legendre polynomial

    :param j: The order of the associated Legendre polynomial, must satisfy |m| < j.
    :param m: The degree of the associated Legendre polynomial, must satisfy |m| < j.
    :param x: An array of values between -1 and +1 on which to evaluate.
    :return: The normalized associated Legendre polynomial evaluated at corresponding x.

    """

    if m < 0:
        m = -m
        y = (-1)**m * norm_assoc_legendre(j, m, x)
    else:
        y = lpmv(m, j, x)
        # Beware of using just np.prod in the denominator here
        # Unless we use float64, values in the denominator > 13! will be incorrect
        try:
            y = np.sqrt((2 * j + 1) / (2*np.prod(range(j - m + 1, j + m + 1), dtype='float64'))) * y
        except RuntimeWarning:
            print('debug')
    return y


def real_sph_harmonic(j, m, theta, phi):
    """
    Evaluate a real spherical harmonic

    :param j: The order of the spherical harmonic. These must satisfy |m| < j.
    :param m: The degree of the spherical harmonic. These must satisfy |m| < j.
    :param theta: The spherical coordinates of the points at which we want to evaluate the real spherical harmonic.
        `theta` is the latitude between 0 and pi
    :param phi: The spherical coordinates of the points at which we want to evaluate the real spherical harmonic.
        `phi` is the longitude, between 0 and 2*pi
    :return: The real spherical harmonics evaluated at the points (theta, phi).
    """
    abs_m = abs(m)
    y = lpmv(abs_m, j, np.cos(theta))

    # Beware of using just np.prod in the denominator here
    # Unless we use float64, values in the denominator > 13! will be incorrect
    try:
        y = np.sqrt((2 * j + 1) / (4 * pi) / np.prod(range(j - abs_m + 1, j + abs_m + 1), dtype='float64')) * y
    except RuntimeWarning:
        print('debug')

    if m < 0:
        y = np.sqrt(2) * np.sin(abs_m * phi) * y
    elif m > 0:
        y = np.sqrt(2) * np.cos(abs_m * phi) * y

    return y


def besselj_zeros(nu, k):
    ensure(k >= 3, 'k must be >= 3')
    ensure(0 <= nu <= 1e7, 'nu must be between 0 and 1e7')

    z = np.zeros(k)

    # Guess first zeros using powers of nu
    c0 = np.array([
        [0.1701, -0.6563, 1.0355, 1.8558],
        [0.1608, -1.0189, 3.1348, 3.2447],
        [-0.2005, -1.2542, 5.7249, 4.3817]
    ])
    z0 = nu + c0 @ ((nu + 1) ** np.array([[-1, -2 / 3, -1 / 3, 1 / 3]]).T)

    # refine guesses
    z[:3] = besselj_newton(nu, z0).squeeze()

    n = 3
    j = 2
    err_tol = 5e-3

    # Estimate further zeros iteratively using spacing of last three zeros so far
    while n < k:
        j = min(j, k - n)

        # Use last 3 zeros to predict spacing for next j zeros
        r = diff(z[n - 3:n]) - pi
        if (r[0] * r[1]) > 0 and (r[0] / r[1]) > 1:
            p = log(r[0] / r[1]) / log(1 - 1 / (n - 1))
            t = np.array(np.arange(1, j + 1), ndmin=2).T / (n - 1)
            dz = pi + r[1] * exp(p * log(1 + t))
        else:
            dz = pi * np.ones((j, 1))

        # Guess and refine
        z0 = z[n - 1] + np.cumsum(dz)
        z[n: n + j] = besselj_newton(nu, z0)

        # Check to see that the sequence of zeros makes sense
        ensure(check_besselj_zeros(nu, z[n - 2: n + j]),
               "Unable to properly estimate Bessel function zeros.")

        # Check how far off we are
        err = (z[n: n + j] - z0) / np.diff(z[n - 1: n + j])

        n = n + j
        if max(abs(err)) < err_tol:
            # Predictions were close enough, double no. of zeros
            j *= 2
        else:
            # Some predictions were off, set to double the no. of good predictions
            j = 2 * (np.where(abs(err) >= err_tol)[0][0] + 1)

    return z


def num_besselj_zeros(ell, r):
    k = 4
    r0 = besselj_zeros(ell, k)
    while all(r0 < r):
        k *= 2
        r0 = besselj_zeros(ell, k)
    r0 = r0[r0 < r]
    return len(r0), r0


def unique_coords_nd(N, ndim):
    """
    Generate unique polar coordinates from 2D or 3D rectangular coordinates.
    :param N: length size of a square or cube.
    :param ndim: number of dimension, 2 or 3.
    :return: The unique polar coordinates in 2D or 3D
    """
    ensure(ndim in (2, 3), 'Only two- or three-dimensional basis functions are supported.')
    ensure(N > 0, 'Number of grid points should be greater than 0.')

    if ndim == 2:
        grid = grid_2d(N)
        mask = grid['r'] <= 1

        # Minor differences in r/theta/phi values are unimportant for the purpose
        # of this function, so round off before proceeding

        # TODO: numpy boolean indexing will return a 1d array (like MATLAB)
        # However, it always searches in row-major order, unlike MATLAB (column-major),
        # with no options to change the search order. The results we'll be getting back are thus not comparable.
        # We transpose the appropriate ndarrays before applying the mask to obtain the same behavior as MATLAB.
        r = grid['r'].T[mask].round(5)
        phi = grid['phi'].T[mask].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(phi, return_inverse=True)

    else:
        grid = grid_3d(N)
        mask = grid['r'] <= 1

        # In Numpy, elements in the indexed array are always iterated and returned in row-major (C-style) order.
        # To emulate a behavior where iteration happens in Fortran order, we swap axes 0 and 2 of both the array
        # being indexed (r/theta/phi), as well as the mask itself.
        # TODO: This is only for the purpose of getting the same behavior as MATLAB while porting the code, and is
        # likely not needed in the final version.

        # Minor differences in r/theta/phi values are unimportant for the purpose of this function,
        # so we round off before proceeding.

        mask_ = np.swapaxes(mask, 0, 2)
        r = np.swapaxes(grid['r'], 0, 2)[mask_].round(5)
        theta = np.swapaxes(grid['theta'], 0, 2)[mask_].round(5)
        phi = np.swapaxes(grid['phi'], 0, 2)[mask_].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(np.vstack([theta, phi]), axis=1, return_inverse=True)

    return {
        'r_unique': r_unique,
        'ang_unique': ang_unique,
        'r_idx': r_idx,
        'ang_idx': ang_idx,
        'mask': mask
    }


def lgwt(ndeg, a, b):
    """
    Compute Legendre-Gauss quadrature

    Generates the Legendre-Gauss nodes and weights on an interval
    [a, b] with truncation order of ndeg for computing definite integrals
    using Legendre-Gauss quadrature. Computes
    Suppose you have a continuous function f(x) which is defined on [a, b]
    which you can evaluate at any x in [a, b]. Simply evaluate it at all of
    the values contained in the x vector to obtain a vector f, then compute
    the definite integral using sum(f.*w);

    The nodes are sorted in ascending order.

    Modified from the MATLAB lgwt version by Greg von Winckel - 02/25/2004

    :param ndeg: truncation order, that is, the number of nodes.
    :param a, b: The endpoints of the interval over which the quadrature is defined.
    :return x, w: The quadrature nodes and weights.
    """
    N = ndeg - 1
    N1 = N + 1
    N2 = N + 2

    xu = np.linspace(-1, 1, N1).T

    # Initial guess.
    y = np.cos((2 * np.arange(N + 1).T + 1) * pi / (2 * N + 2)) + (0.27 / N1) * np.sin(pi * xu * N / N2)

    # Legendre-Gauss Vandermonde matrix.
    L = np.zeros((N1, N2))

    # Derivative of the matrix.
    Lp = np.zeros_like(y)

    # Compute the zeros of the (N+1)th Legendre polynomial using the recursion
    # relation and the Newton-Raphson method. Iterate until new points are
    # uniformly within epsilon of old points.
    y0 = 2
    while np.max(abs(y - y0)) > sys.float_info.epsilon:
        L[:, 0] = 1
        L[:, 1] = y

        for k in range(1, N1):
            L[:, k + 1] = ((2 * (k + 1) - 1) * y * L[:, k] - k * L[:, k - 1]) / (k + 1)

        Lp = N2 * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y ** 2)

        y0 = y
        y = y0 - L[:, N2 - 1] / Lp

    # Linear map from [-1, 1] to [a, b].
    x = (a * (1 - y) + b * (1 + y)) / 2

    # Compute the weights.
    w = (b - a) / ((1 - y ** 2) * Lp ** 2) * (N2 / N1) ** 2

    # Sort x in ascending order.
    ids = np.argsort(x)
    w = w[ids]
    x = x[ids]

    return x, w
