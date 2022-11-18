"""
FFT/IFFT utilities
"""

from aspire.numeric import fft

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
        x = fft.ifftshift(x, dim)
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
        x = fft.fftshift(x, dim)
    return x
