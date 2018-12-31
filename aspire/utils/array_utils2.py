import numpy as np
from numpy.fft import fftshift, ifftshift
from pyfftw.interfaces import numpy_fft


def crop2(x, out_shape):  # [30, 20, 50]
    out_shape = np.array(out_shape)
    x_shape = np.array(x.shape)
    out_shape[np.where(out_shape < 0)] = x_shape[np.where(out_shape < 0)]
    odds = (x_shape - out_shape) % 2 == 1
    low = (x_shape - out_shape + odds) // 2
    high = low + out_shape

    if x.ndim is 1:
        return x[low[0]:high[0]]
    elif x.ndim is 2:
        return x[low[0]:high[0], low[1]:high[1]]
    elif x.ndim is 3:
        return x[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    elif x.ndim is 4:
        return x[low[0]:high[0], low[1]:high[1], low[2]:high[2], low[3]:high[3]]
    else:
        raise Exception('Unsupported num of dimensions: ' + x.ndim)


def downsample2(x, out_shape):

    out_shape = np.array(out_shape)
    x_shape = np.array(x.shape)
    out_shape[np.where(out_shape < 0)] = x_shape[np.where(out_shape < 0)]
    axes = [a for a in np.arange(len(x_shape)) if out_shape[a] < x_shape[a]]

    fx = crop2(fftshift(numpy_fft.fftn(x, axes=axes), axes=axes), out_shape)
    out = numpy_fft.ifftn(ifftshift(fx, axes=axes), axes=axes) * (np.prod(out_shape) / np.prod(x_shape))

    if np.isreal(x).any():
        out = np.real(out)

    fx = ifftshift(fx)
    return [out, fx]
