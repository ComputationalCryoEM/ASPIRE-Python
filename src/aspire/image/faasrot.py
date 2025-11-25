import numpy as np

from aspire.numeric import xp


def _pre_rotate(theta):
    """
    Given angle `theta` (degrees) return nearest rotation of 90
    degrees required to place angle within [-45,45) and residual
    rotation in degrees.
    """

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

    return residual, rot90


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
    """
    theta, mult90 = _pre_rotate(theta)

    theta = np.pi * theta / 180
    theta = -theta  # Yaroslavsky rotated CW

    cy, sy = _shift_center(ny)
    cx, sx = _shift_center(nx)

    # Floating point epsilon
    eps = np.finfo(np.float64).eps

    # Precompute Y interpolation tables
    My = np.zeros((nx, ny), dtype=np.complex128)
    r = np.arange(cy + 1, dtype=int)
    u = (1 - np.cos(theta)) / np.sin(theta + eps)
    # print("u", u)
    alpha1 = 2 * np.pi * 1j * r / ny

    # print("alpha1", alpha1)

    linds = np.arange(ny - 1, cy, -1, dtype=int)
    # print('aaa', ny-1, cy, -1)
    rinds = np.arange(1, cy - 2 * sy + 1, dtype=int)
    # print(linds,rinds)
    # This can be broadcast, but leaving loop since would be close to CUDA...
    for x in range(nx):
        Ux = u * (x - cx + sx + 2)
        # print("Ux",Ux)
        My[x, r] = np.exp(alpha1 * Ux)
        My[x, linds] = np.conj(My[x, rinds])

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
        Mx[y, linds] = np.conj(Mx[y, rinds])

    # After building, transpose to (nx, ny).
    Mx = Mx.T

    return Mx, My, mult90


def _rot90(img):
    return np.flipud(img.T)


def _rot180(img):
    return np.flipud(np.fliplr(img))


def _rot270(img):
    return np.fliplr(img.T)


def faastrotate(images, theta, M=None):

    # Make a stack of 1
    if images.ndim == 2:
        images = images[None, :, :]

    n, px0, px1 = images.shape
    assert px0 == px1, "Currently only implemented for square images."

    if M is None:
        M = _pre_compute(theta, px0, px1)
    Mx, My, Mrot90 = M

    result = np.empty((n, px0, px1), dtype=np.float64)

    for i in range(n):

        img = images[i]

        # Pre rotate by multiples of 90
        if Mrot90 == 1:
            img = _rot90(img)
        elif Mrot90 == 2:
            img = _rot180(img)
        elif Mrot90 == 3:
            img = _rot270(img)

        # Shear 1
        img_k = np.fft.fft(img, axis=-1)
        # okay print("\nfft1(img_k):\n", img_k,"\n")
        print("\nMy:\n", My, "\n")
        img_k = img_k * My
        print("\nmult (img_k):\n", img_k, "\n")  # okay

        # for _i in range(16):
        #     #print(f'A[{_i}].x = {img_k.flatten()[_i].real};')
        #     #print(f'A[{_i}].y = {img_k.flatten()[_i].imag};')
        #     print(f'FA[{_i}] = {img_k.flatten()[_i]};')

        # breakpoint()
        result[i] = np.real(np.fft.ifft(img_k, axis=-1))
        print("\nstage1\n", result[i] * 4, "\n")

        # Shear 2
        img_k = np.fft.fft(result[i], axis=0)
        img_k = img_k * Mx
        result[i] = np.real(np.fft.ifft(img_k, axis=0))

        print("\nstage2\n", result * 4 * 4)

        # Shear 3
        img_k = np.fft.fft(result[i], axis=-1)
        img_k = img_k * My
        result[i] = np.real(np.fft.ifft(img_k, axis=-1))

        print("\nstage3\n", result * 4 * 4 * 4, "\n")

    return result
