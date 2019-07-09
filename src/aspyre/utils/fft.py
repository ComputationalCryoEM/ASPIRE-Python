"""
FFT/IFFT utilities
"""

from scipy.fftpack import ifftshift, ifft, ifft2, fftshift, fft, fft2

def centered_ifft1(x):
    """
    Calculate a centered, one-dimensional inverse FFT
    :param x: The one-dimensional signal to be transformed.
        The inverse FFT is only applied along the first dimension.
    :return: The centered inverse Fourier transform of x.
    """
    x = ifftshift(x, 0)
    x = ifft(x, axes=0)
    x = fftshift(x, 0)
    return x


def centered_fft1(x):
    x = ifftshift(x, 0)
    x = fft(x, axes=0)
    x = fftshift(x, 0)
    return x

def centered_ifft2(x):
    """
    Calculate a centered, two-dimensional inverse FFT
    :param x: The two-dimensional signal to be transformed.
        The inverse FFT is only applied along the first two dimensions.
    :return: The centered inverse Fourier transform of x.
    """
    x = ifftshift(ifftshift(x, 0), 1)
    x = ifft2(x, axes=(0, 1))
    x = fftshift(fftshift(x, 0), 1)
    return x


def centered_fft2(x):
    x = ifftshift(ifftshift(x, 0), 1)
    x = fft2(x, axes=(0, 1))
    x = fftshift(fftshift(x, 0), 1)
    return x

def centered_ifft3(x):
    """
    Calculate a centered, three-dimensional inverse FFT
    :param x: The three-dimensional signal to be transformed.
        The inverse FFT is only applied along the first three dimensions.
    :return: The centered inverse Fourier transform of x.
    """
    x = ifftshift(ifftshift(ifftshift(x, 0), 1), 2)
    x = ifftn(x, axes=(0, 1, 2))
    x = fftshift(fftshift(fftshift(x, 0), 1), 2)
    return x


def centered_fft3(x):
    x = ifftshift(ifftshift(ifftshift(x, 0), 1), 2)
    x = fftn(x, axes=(0, 1, 2))
    x = fftshift(fftshift(fftshift(x, 0), 1), 2)
    return x


def mdim_ifftshift(x, dims=None):
    """
    Multi-dimensional FFT unshift
    :param x: The array to be unshifted.
    :param dims: An array of dimension indices along which the unshift should occur.
        If None, the unshift is performed along all dimensions.
    :return: The x array unshifted along the desired dimensions.
    """
    if dims is None:
        dims = range(0, x.ndim)
    for dim in dims:
        x = ifftshift(x, dim)
    return x


def mdim_fftshift(x, dims=None):
    """
    Multi-dimensional FFT shift

    :param x: The array to be shifted.
    :param dims: An array of dimension indices along which the shift should occur.
        If None, the shift is performed along all dimensions.
    :return: The x array shifted along the desired dimensions.
    """
    if dims is None:
        dims = range(0, x.ndim)
    for dim in dims:
        x = fftshift(x, dim)
    return x
