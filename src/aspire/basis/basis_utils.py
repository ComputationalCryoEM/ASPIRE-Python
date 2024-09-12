"""
Define related utility functions for Fourier–Bessel (2D), Spherical Fourier–Bessel (3D) and
prolate spheroidal wave function (PSWF) objects.
"""

import logging

import numpy as np
from numpy import diff, exp, log, pi
from numpy.polynomial.legendre import leggauss
from scipy.special import jn, jv

from aspire.utils import grid_2d, grid_3d

logger = logging.getLogger(__name__)


def check_besselj_zeros(nu, z):
    """
    Sanity-check a sequence of estimated zeros of the Bessel function with order `nu`.

    :param nu: The real number order of the Bessel function.
    :param z: (Array-like) A sequence of postulated zeros.
    :return result: True or False.
    """
    # Compute first and second order differences of the sequence of zeros
    dz = np.diff(z)
    ddz = np.diff(dz)

    # Check criteria for acceptable zeros
    result = True
    # Real roots
    result = result and all(np.isreal(z))
    # All roots should be > 0, check first of increasing sequence
    result = result and z[0] > 0
    # Spacing between zeros is greater than 3
    result = result and all(dz > 3)

    # Second order differences should be zero or just barely increasing to
    # within 16x machine precision.
    if nu >= 0.5:
        result = result and all(ddz < 16 * np.spacing(z[1:-1]))
    # For nu < 0.5 the spacing will be slightly decreasing, so flip the sign
    else:
        result = result and all(ddz > -16 * np.spacing(z[1:-1]))

    return result


def besselj_newton(nu, z0, max_iter=10):
    """
    Uses the Newton-Raphson method to compute the zero(s) of the
    Bessel function with order `nu` with initial guess(es) `z0`.

    :param nu: The real number order of the Bessel function.
    :param z0: (Array-like) The initial guess(es) for the root-finding algorithm.
    :param max_iter: Maximum number of iterations for Newton-Raphson
        (default: 10).
    :return z: (Array-like) The estimated root(s).
    """
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

    # For negative m, flip sign and use the symmetry identity.
    # In the rest, we assume that m is non-negative.
    if m < 0:
        m = -m
        px = (-1) ** m * norm_assoc_legendre(j, m, x)
        px *= (-1) ** m
        return px

    # Initialize the recurrence at (m, m) and (m, m+1).
    p0 = (
        (-1) ** m
        * np.sqrt(
            (2 * m + 1)
            / 2
            * np.prod(np.arange(2 * m - 1, 0, -2) / np.arange(2 * m, 0, -2))
        )
        * (1 - x * x) ** (m / 2)
    )

    p1 = x * np.sqrt(2 * m + 3) * p0

    # If these are the desired indices, return these initial values.
    if j == m:
        px = p0
    elif j == m + 1:
        px = p1
    else:
        # Fixing m, work our way up from (m, m+1) to (m, j).
        for n in range(m + 1, j):
            px = np.sqrt((2 * n + 3) / ((n + 1 + m) * (n + 1 - m))) * (
                np.sqrt(2 * n + 1) * x * p1
                - np.sqrt((n + m) * (n - m) / (2 * n - 1)) * p0
            )
            p0 = p1
            p1 = px

    return px


def sph_harm(j, m, theta, phi):
    """
    Compute spherical harmonics.

    Note call signature convention may be different from other packages.

    :param m: Order |m| <= j
    :param j: Harmonic degree, j>=0
    :param theta: latitude coordinate [0, pi]
    :param phi: longitude coordinate [0, 2*pi]
    :return: Complex array of evaluated spherical harmonics.
    """

    # Compute sph_harm for positive `abs(m)`
    y = (
        norm_assoc_legendre(j, abs(m), np.cos(theta))
        * np.exp(1j * abs(m) * phi)
        * np.sqrt(0.5 / np.pi)
    )

    # Use identity for negative `m`
    if m < 0:
        y = (-1) ** (m % 2) * np.conj(y)

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

    # Note the calling convention here may not match other `sph_harm` packages
    y = sph_harm(j, abs_m, theta, phi)

    if m < 0:
        y = np.sqrt(2) * np.imag(y)
    elif m > 0:
        y = np.sqrt(2) * np.real(y)
    else:
        y = np.real(y)

    return y


def besselj_zeros(nu, k):
    """
    Finds the first `k` zeros of the Bessel function of order `nu`, i.e. J_nu.
    Adapted from "zerobess.m" by Jonas Lundgren <splinefit@gmail.com>

    :param nu: The real number order of the Bessel function (must be positive and <1e7).
    :param k: The number of zeros to return (must be >= 3).
    :return z: A 1D NumPy array of the first `k` zeros.
    """
    assert k >= 3, "k must be >= 3"
    assert 0 <= nu <= 1e7, "nu must be between 0 and 1e7"

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
        assert check_besselj_zeros(
            nu, z[n - 2 : n + j]
        ), "Unable to properly estimate Bessel function zeros."

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


def all_besselj_zeros(ell, r):
    """
    Compute the zeros of the order `ell` Bessel function which are less than `r`.

    :param ell: The real number order of the Bessel function.
    :param r: The upper bound for zeros returned.
    :return n, r0: The number of zeros and the zeros themselves
        as a NumPy array.
    """
    k = 4
    # get the first 4 zeros
    r0 = besselj_zeros(ell, k)
    while all(r0 < r):
        # increase the number of zeros sought
        # until one of the zeros is greater than `r`
        k *= 2
        r0 = besselj_zeros(ell, k)
    r0 = r0[r0 < r]
    # return the number of zeros and the zeros themselves
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
    assert ndim in (
        2,
        3,
    ), "Only two- or three-dimensional basis functions are supported."
    assert N > 0, "Number of grid points should be greater than 0."

    if ndim == 2:
        grid = grid_2d(
            N, shifted=shifted, normalized=normalized, indexing="yx", dtype=dtype
        )
        mask = grid["r"] <= 1

        # Minor differences in r/theta/phi values are unimportant for the purpose
        # of this function, so round off before proceeding

        r = grid["r"][mask].round(5)
        phi = grid["phi"][mask].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(phi, return_inverse=True)

    else:
        grid = grid_3d(
            N, shifted=shifted, normalized=normalized, indexing="zyx", dtype=dtype
        )
        mask = grid["r"] <= 1

        # Minor differences in r/theta/phi values are unimportant for the purpose of this function,
        # so we round off before proceeding.

        r = grid["r"][mask].round(5)
        theta = grid["theta"][mask].round(5)
        phi = grid["phi"][mask].round(5)

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
