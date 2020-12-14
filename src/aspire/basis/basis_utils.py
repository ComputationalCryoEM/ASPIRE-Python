"""
Define related utility functions for Fourier–Bessel (2D), Spherical Fourier–Bessel (3D) and
prolate spheroidal wave function (PSWF) objects.
"""

import logging

import numpy as np
from numpy import diff, exp, log, pi
from numpy.polynomial.legendre import leggauss
from scipy.special import jn, jv, lpmv, sph_harm

from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d, grid_3d

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
        dz = -f / fp
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

    if scalar:
        j = j.item()

    return j


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
        y = (-1) ** m * norm_assoc_legendre(j, m, x)
    else:
        y = lpmv(m, j, x)
        # Beware of using just np.prod in the denominator here
        # Unless we use float64, values in the denominator > 13! will be incorrect
        try:
            y = (
                np.sqrt(
                    (2 * j + 1)
                    / (2 * np.prod(range(j - m + 1, j + m + 1), dtype=np.float64))
                )
                * y
            )
        except RuntimeWarning:
            logger.error("debug")
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

    y = sph_harm(abs_m, j, phi, theta)
    if m < 0:
        y = np.sqrt(2) * np.imag(y)
    elif m > 0:
        y = np.sqrt(2) * np.real(y)
    else:
        y = np.real(y)

    return y


def besselj_zeros(nu, k):
    ensure(k >= 3, "k must be >= 3")
    ensure(0 <= nu <= 1e7, "nu must be between 0 and 1e7")

    z = np.zeros(k)

    # Guess first zeros using powers of nu
    c0 = np.array(
        [
            [0.1701, -0.6563, 1.0355, 1.8558],
            [0.1608, -1.0189, 3.1348, 3.2447],
            [-0.2005, -1.2542, 5.7249, 4.3817],
        ]
    )
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
        r = diff(z[n - 3 : n]) - pi
        if (r[0] * r[1]) > 0 and (r[0] / r[1]) > 1:
            p = log(r[0] / r[1]) / log(1 - 1 / (n - 1))
            t = np.array(np.arange(1, j + 1), ndmin=2).T / (n - 1)
            dz = pi + r[1] * exp(p * log(1 + t))
        else:
            dz = pi * np.ones((j, 1))

        # Guess and refine
        z0 = z[n - 1] + np.cumsum(dz)
        z[n : n + j] = besselj_newton(nu, z0)

        # Check to see that the sequence of zeros makes sense
        ensure(
            check_besselj_zeros(nu, z[n - 2 : n + j]),
            "Unable to properly estimate Bessel function zeros.",
        )

        # Check how far off we are
        err = (z[n : n + j] - z0) / np.diff(z[n - 1 : n + j])

        n = n + j
        if max(abs(err)) < err_tol:
            # Predictions were close enough, double number of zeros
            j *= 2
        else:
            # Some predictions were off, set to double the number of good predictions
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


def unique_coords_nd(N, ndim, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate unique polar coordinates from 2D or 3D rectangular coordinates.
    :param N: length size of a square or cube.
    :param ndim: number of dimension, 2 or 3.
    :param shifted: shifted half pixel or not for odd N.
    :param normalized: normalize the grid or not.
    :return: The unique polar coordinates in 2D or 3D
    """
    ensure(
        ndim in (2, 3), "Only two- or three-dimensional basis functions are supported."
    )
    ensure(N > 0, "Number of grid points should be greater than 0.")

    if ndim == 2:
        grid = grid_2d(N, shifted=shifted, normalized=normalized, dtype=dtype)
        mask = grid["r"] <= 1

        # Minor differences in r/theta/phi values are unimportant for the purpose
        # of this function, so round off before proceeding

        # TODO: numpy boolean indexing will return a 1d array (like MATLAB)
        # However, it always searches in row-major order, unlike MATLAB (column-major),
        # with no options to change the search order. The results we'll be getting back are thus not comparable.
        # We transpose the appropriate ndarrays before applying the mask to obtain the same behavior as MATLAB.
        r = grid["r"].T[mask].round(5)
        phi = grid["phi"].T[mask].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(phi, return_inverse=True)

    else:
        grid = grid_3d(N, shifted=shifted, normalized=normalized, dtype=dtype)
        mask = grid["r"] <= 1

        # In Numpy, elements in the indexed array are always iterated and returned in row-major (C-style) order.
        # To emulate a behavior where iteration happens in Fortran order, we swap axes 0 and 2 of both the array
        # being indexed (r/theta/phi), as well as the mask itself.
        # TODO: This is only for the purpose of getting the same behavior as MATLAB while porting the code, and is
        # likely not needed in the final version.

        # Minor differences in r/theta/phi values are unimportant for the purpose of this function,
        # so we round off before proceeding.

        mask_ = np.swapaxes(mask, 0, 2)
        r = np.swapaxes(grid["r"], 0, 2)[mask_].round(5)
        theta = np.swapaxes(grid["theta"], 0, 2)[mask_].round(5)
        phi = np.swapaxes(grid["phi"], 0, 2)[mask_].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(
            np.vstack([theta, phi]), axis=1, return_inverse=True
        )

    return {
        "r_unique": r_unique,
        "ang_unique": ang_unique,
        "r_idx": r_idx,
        "ang_idx": ang_idx,
        "mask": mask,
    }


def lgwt(ndeg, a, b, dtype=np.float32):
    """
    Compute Legendre-Gauss quadrature

    Generates the Legendre-Gauss nodes and weights on an interval
    [a, b] with truncation order of ndeg for computing definite integrals
    using Legendre-Gauss quadrature.
    Suppose you have a continuous function f(x) which is defined on [a, b]
    which you can evaluate at any x in [a, b]. Simply evaluate it at all of
    the values contained in the x vector to obtain a vector f, then compute
    the definite integral using sum(f.*w);

    This is a 2rapper for numpy.polynomial leggauss which outputs only in the
    range of (-1, 1).

    :param ndeg: truncation order, that is, the number of nodes.
    :param a, b: The endpoints of the interval over which the quadrature is defined.
    :return x, w: The quadrature nodes and weights.
    """

    x, w = leggauss(ndeg)
    scale_factor = (b - a) / 2
    shift = (a + b) / 2
    x = scale_factor * x + shift
    w = scale_factor * w

    return x.astype(dtype), w.astype(dtype)


def d_decay_approx_fun(a, b, c, d):
    return np.square(c) / (16 * (np.square(d) + d * (2 * b + a + 1)) - np.square(c))


def p_n(n, alpha, beta, x):
    """
    The first n jacobi polynomial of x as defined in Yoel's PSWF paper, eq (2), page 6

    :param n: int, > 0
        Number of polynomials to compute
    :param alpha: float, > -1
    :param beta: float, > -1
    :param x: (m,) ndarray
    :return: v: (m, n + 1) ndarray
        v[:, i] = P^{(alpha, beta)}_n(x) as defined in the paper
    """
    m = len(x)
    if n < 0:
        return np.array([])
    if n == 0:
        return np.ones(m)
    x = x.reshape((m,))
    v = np.zeros((m, n + 1))
    alpha_p_beta = alpha + beta
    alpha_m_beta = alpha - beta
    v[:, 0] = 1
    v[:, 1] = (1 + 0.5 * alpha_p_beta) * x + 0.5 * alpha_m_beta

    for i in range(2, n + 1):
        c1 = 2 * i * (i + alpha_p_beta) * (2 * i + alpha_p_beta - 2)
        c2 = (
            (2 * i + alpha_p_beta)
            * (2 * i + alpha_p_beta - 1)
            * (2 * i + alpha_p_beta - 2)
        )
        c3 = (2 * i + alpha_p_beta - 1) * alpha_p_beta * alpha_m_beta
        c4 = -2 * (i - 1 + alpha) * (i - 1 + beta) * (2 * i + alpha_p_beta)
        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1
    return v


def t_x_mat(x, n, j, approx_length):
    a = np.power(x, n + 0.5)
    b = np.sqrt(2 * (2 * j + n + 1))
    c = p_n(approx_length - 1, n, 0, 1 - 2 * np.square(x))
    return np.einsum("i,j,ij->ij", a, b, c)


def t_x_mat_dot(x, n, j, approx_length):
    #  x need to be a matrix instead of a vector in t_x_mat
    return np.power(x, n + 0.5).dot(np.sqrt(2 * (2 * j + n + 1))) * p_n(
        approx_length - 1, n, 0, 1 - 2 * np.square(x)
    )


def t_x_derivative_mat(t1, t2, x, big_n, range_array, approx_length):
    return -2 * (big_n + range_array + 1) * np.outer(
        np.power(x, big_n + 1.5), t2
    ) * np.column_stack(
        (np.zeros(len(x)), p_n(approx_length - 2, big_n + 1, 1, t1))
    ) + (
        big_n + 0.5
    ) * np.outer(
        np.power(x, big_n - 0.5), t2
    ) * p_n(
        approx_length - 1, big_n, 0, t1
    )


def t_radial_part_mat(x, n, j, m):
    a = np.power(x, n)
    b = np.sqrt(2 * (2 * j + n + 1))
    c = p_n(m - 1, n, 0, 1 - 2 * np.square(x))
    return np.einsum("i,j,ij->ij", a, b, c)


def k_operator(nu, x):
    return jn(nu, x) * np.sqrt(x)
