import numpy as np

from aspire.numeric import fft, xp


def _pre_rotate(theta):
    """
    Given angle `theta` (radians) return nearest rotation of 90
    degrees required to place angle within [-45,45) degrees and residual
    rotation (radians).
    """

    # todo
    theta = np.rad2deg(theta)

    theta = np.mod(theta, 360)

    # 0 < 45
    rot90 = 0
    residual = theta

    if theta >= 45 and theta < 135:
        rot90 = 1
        residual = theta - 90
    elif theta >= 135 and theta < 225:
        rot90 = 2
        residual = theta - 180
    elif theta >= 215 and theta < 315:
        rot90 = 3
        residual = theta - 270
    elif theta >= 315 and theta < 360:
        rot90 = 0
        residual = theta - 360

    return np.deg2rad(residual), rot90


def _shift_center(n):
    """
    Given `n` pixels return center pixel and shift amount, 0 or 1/2.
    """
    if n % 2 == 0:
        c = n // 2  # center
        s = 1 / 2  # shift
    else:
        c = n // 2
        s = 0

    return c, s


def _pre_compute(theta, nx, ny):
    """
    Retuns M = (Mx, My, rot90)

    :param theta: angle in radians
    """
    theta, mult90 = _pre_rotate(theta)

    theta = -theta  # Yaroslavsky rotated CW

    cy, sy = _shift_center(ny)
    cx, sx = _shift_center(nx)

    # Floating point epsilon
    eps = np.finfo(np.float64).eps

    # Precompute Y interpolation tables
    My = np.zeros((nx, ny), dtype=np.complex128)
    r = np.arange(cy + 1, dtype=int)
    u = (1 - np.cos(theta)) / np.sin(theta + eps)
    alpha1 = 2 * np.pi * 1j * r / ny

    linds = np.arange(ny - 1, cy, -1, dtype=int)
    rinds = np.arange(1, cy - 2 * sy + 1, dtype=int)
    # This can be broadcast, but leaving loop since would be close to CUDA...
    for x in range(nx):
        Ux = u * (x - cx + sx + 2)
        My[x, r] = np.exp(alpha1 * Ux)
        My[x, linds] = My[x, rinds].conj()

    # Precompute X interpolation tables
    Mx = np.zeros((ny, nx), dtype=np.complex128)
    r = np.arange(cx + 1, dtype=int)
    u = -np.sin(theta)
    alpha2 = 2 * np.pi * 1j * r / nx

    linds = np.arange(nx - 1, cx, -1, dtype=int)
    rinds = np.arange(1, cx - 2 * sx + 1, dtype=int)
    # This can be broadcast, but leaving loop since would be close to CUDA...
    for y in range(ny):
        Uy = u * (y - cy + sy + 2)
        Mx[y, r] = np.exp(alpha2 * Uy)
        Mx[y, linds] = Mx[y, rinds].conj()

    # After building, transpose to (nx, ny).
    Mx = Mx.T

    return Mx, My, mult90


def _rot90(img):
    return xp.flipud(img.T)


def _rot180(img):
    return xp.flipud(xp.fliplr(img))


def _rot270(img):
    return xp.fliplr(img.T)


def faasrotate(images, theta, M=None):
    """
    Rotate `images` array by `theta` radians ccw.

    :param images: (n , px, px) array of image data
    :param theta: rotation angle in radians
    :param M: optional precomputed shearing table
    :return: (n, px, px) array fo rotated image data
    """

    # Make a stack of 1
    if images.ndim == 2:
        images = images[None, :, :]

    n, px0, px1 = images.shape
    assert px0 == px1, "Currently only implemented for square images."

    if M is None:
        M = _pre_compute(theta, px0, px1)
    Mx, My, Mrot90 = M

    Mx, My = xp.asarray(Mx), xp.asarray(My)

    result = xp.empty((n, px0, px1), dtype=np.float64)
    for i in range(n):

        img = xp.asarray(images[i])

        # Pre rotate by multiples of 90
        if Mrot90 == 1:
            img = _rot90(img)
        elif Mrot90 == 2:
            img = _rot180(img)
        elif Mrot90 == 3:
            img = _rot270(img)

        # Shear 1
        img_k = fft.fft(img, axis=-1)
        img_k = img_k * My
        result[i] = fft.ifft(img_k, axis=-1).real

        # Shear 2
        img_k = fft.fft(result[i], axis=-2)
        img_k = img_k * Mx
        result[i] = fft.ifft(img_k, axis=-2).real

        # Shear 3
        img_k = fft.fft(result[i], axis=-1)
        img_k = img_k * My
        result[i] = fft.ifft(img_k, axis=-1).real

    return xp.asnumpy(result)
