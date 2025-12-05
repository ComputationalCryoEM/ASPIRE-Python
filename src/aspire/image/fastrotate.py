import numpy as np

from aspire.numeric import fft, xp


def _pre_rotate(theta):
    """
    Given `theta` radians return nearest rotation of pi/2
    required to place angle within [-pi/4,pi/4) and the residual
    rotation in radians.

    :param theta: Rotation in radians
    :returns:
        - Residual angle in radians
        - Number of pi/2 rotations
    """

    theta = np.mod(theta, 2 * np.pi)

    # 0 < pi/4
    rots = 0
    residual = theta

    if theta >= np.pi / 4 and theta < 3 * np.pi / 4:
        rots = 1
        residual = theta - np.pi / 2
    elif theta >= 3 * np.pi / 4 and theta < 5 * np.pi / 4:
        rots = 2
        residual = theta - np.pi
    elif theta >= 5 * np.pi / 4 and theta < 7 * np.pi / 4:
        rots = 3
        residual = theta - 3 * np.pi / 2
    elif theta >= 7 * np.pi / 4 and theta < 2 * np.pi:
        rots = 0
        residual = theta - 2 * np.pi

    return residual, rots


def _shift_center(n):
    """
    Given `n` pixels return center pixel and shift amount, 0 or 1/2.

    :param n: Number of pixels
    :returns:
        - center pixel
        - shift amount
    """
    if n % 2 == 0:
        c = n // 2  # center
        s = 1 / 2  # shift
    else:
        c = n // 2
        s = 0

    return c, s


def compute_fastrotate_interp_tables(theta, nx, ny):
    """
    Retuns iterpolation tables as tuple M = (Mx, My, rots).

    :param theta: angle in radians
    :param nx: Number pixels first axis
    :param ny: Number pixels second axis
    """
    theta, mult90 = _pre_rotate(theta)

    # Reverse rotation, Yaroslavsky rotated CW
    theta = -theta

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

    Ux = u * (np.arange(nx) - cx + sx + 2)
    My[:, r] = np.exp(alpha1[None, :] * Ux[:, None])
    My[:, linds] = My[:, rinds].conj()

    # Precompute X interpolation tables
    Mx = np.zeros((ny, nx), dtype=np.complex128)
    r = np.arange(cx + 1, dtype=int)
    u = -np.sin(theta)
    alpha2 = 2 * np.pi * 1j * r / nx

    linds = np.arange(nx - 1, cx, -1, dtype=int)
    rinds = np.arange(1, cx - 2 * sx + 1, dtype=int)

    Uy = u * (np.arange(ny) - cy + sy + 2)
    Mx[:, r] = np.exp(alpha2[None, :] * Uy[:, None])
    Mx[:, linds] = Mx[:, rinds].conj()

    # After building, transpose to (nx, ny).
    Mx = Mx.T

    return Mx, My, mult90


# The following helper utilities are written to work with
# `img` data of dimension 2 or more where the data is expected to be
# in the (-2,-1) dimensions with any other dims as stack axes.
def _rot90(img):
    """Rotate image array by 90 degrees."""
    # stack broadcast of flipud(img.T)
    return xp.flip(xp.swapaxes(img, -1, -2), axis=-2)


def _rot180(img):
    """Rotate image array by 180 degrees."""
    # stack broadcast of flipud(fliplr)
    return xp.flip(img, axis=(-1, -2))


def _rot270(img):
    """Rotate image array by 90 degrees."""
    # stack broadcast of fliplr(img.T)
    return xp.flip(xp.swapaxes(img, -1, -2), axis=-1)


def fastrotate(images, theta, M=None):
    """
    Rotate `images` array by `theta` radians ccw using shearing algorithm.

    Note that this algorithm may have artifacts near the rotation boundary
    and will have artifacts outside the rotation boundary.
    Users can avoid these by zero padding the input image then
    cropping the rotated image and/or masking.

    For reference and notes:
        `https://github.com/PrincetonUniversity/aspire/blob/760a43b35453e55ff2d9354339e9ffa109a25371/common/fastrotate/fastrotate.m`

    :param images: (n , px, px) array of image data
    :param theta: rotation angle in radians
    :param M: optional precomputed shearing table
    :return: (n, px, px) array of rotated image data
    """

    # Make a stack of 1
    if images.ndim == 2:
        images = images[None, :, :]

    n, px0, px1 = images.shape
    assert px0 == px1, "Currently only implemented for square images."

    if M is None:
        M = compute_fastrotate_interp_tables(theta, px0, px1)
    Mx, My, Mrots = M

    Mx, My = xp.asarray(Mx, dtype=images.dtype), xp.asarray(My, dtype=images.dtype)

    # Store if `images` data was provide on host (np.darray)
    _host = isinstance(images, np.ndarray)

    # If needed copy image array to device
    images = xp.asarray(images)

    # Pre rotate by multiples of 90 (pi/2)
    if Mrots == 1:
        images = _rot90(images)
    elif Mrots == 2:
        images = _rot180(images)
    elif Mrots == 3:
        images = _rot270(images)

    # Shear 1
    img_k = fft.fft(images, axis=-1)
    img_k = img_k * My
    images = fft.ifft(img_k, axis=-1).real

    # Shear 2
    img_k = fft.fft(images, axis=-2)
    img_k = img_k * Mx
    images = fft.ifft(img_k, axis=-2).real

    # Shear 3
    img_k = fft.fft(images, axis=-1)
    img_k = img_k * My
    images = fft.ifft(img_k, axis=-1).real

    # Return to host if needed
    if _host:
        images = xp.asnumpy(images)

    return images
