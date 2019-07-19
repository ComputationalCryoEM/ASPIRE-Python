import numpy as np

from scipy.interpolate import RegularGridInterpolator
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_1d, grid_2d, grid_3d
from aspire.utils.fft import centered_fft1, centered_ifft1, centered_fft2, centered_ifft2, centered_fft3, centered_ifft3


def downsample(insamples, szout):
    """
    Blur and downsample 1D to 3D objects such as, curves, images or volumes
    :param insamples: Set of objects to be downsampled in the form of an array, the last dimension
                    is the number of objects.
    :param szout: The desired resolution of for output objects.
    :return: An array consists of the blurred and downsampled objects.
    """

    ensure(insamples.ndim-1 == szout.ndim, 'The number of downsampling dimensions is not the same as that of objects.')

    L_in = insamples.shape[0]
    L_out = szout.shape[0]
    ndata = insamples.shape(-1)
    outdims = szout
    outdims.push_back(ndata)
    outsamples = np.zeros((outdims)).astype(insamples.dtype)

    if insamples.ndim == 2:
        # one dimension object
        grid_in = grid_1d(L_in)
        grid_out = grid_1d(L_out)
        # x values corresponding to 'grid'. This is what scipy interpolator needs to function.
        x = np.ceil(np.arange(-L_in/2, L_in/2)) / (L_in/2)
        mask = (np.abs(grid_in['x']) < L_out/L_in)
        insamples_fft = np.real(centered_ifft1(centered_fft1(insamples) * np.expand_dims(mask, 1)))
        for idata in range(ndata):
            interpolator = RegularGridInterpolator(
                (x,),
                insamples_fft[:, idata],
                bounds_error=False,
                fill_value=0
            )
            outsamples[:, :, idata] = interpolator(np.dstack([grid_out['x']]))

    elif insamples.ndim == 3 :
        grid_in = grid_2d(L_in)
        grid_out = grid_2d(L_out)
        # x, y values corresponding to 'grid'. This is what scipy interpolator needs to function.
        x = y = np.ceil(np.arange(-L_in/2, L_in/2)) / (L_in/2)
        mask = (np.abs(grid_in['x']) < L_out/L_in) & (np.abs(grid_in['y']) < L_out/L_in)
        insamples_fft = np.real(centered_ifft2(centered_fft2(insamples) * np.expand_dims(mask, 2)))
        for idata in range(ndata):
            interpolator = RegularGridInterpolator(
                (x, y),
                insamples_fft[:, :, idata],
                bounds_error=False,
                fill_value=0
            )
            outsamples[:, :, idata] = interpolator(np.dstack([grid_out['x'], grid_out['y']]))

    elif insamples.ndim == 4:
        grid_in = grid_3d(L_in)
        grid_out = grid_3d(L_out)
        # x, y, z values corresponding to 'grid'. This is what scipy interpolator needs to function.
        x = y = z = np.ceil(np.arange(-L_in/2, L_in/2)) / (L_in/2)
        mask = (np.abs(grid_in['x']) < L_out/L_in) & (np.abs(grid_in['y']) < L_out/L_in) & (np.abs(grid_in['z']) < L_out/L_in)
        insamples_fft = np.real(centered_ifft3(centered_fft3(insamples) * np.expand_dims(mask, 3)))
        for idata in range(ndata):
            interpolator = RegularGridInterpolator(
                (x, y, z),
                insamples_fft[:, :, :, idata],
                bounds_error=False,
                fill_value=0
            )
            outsamples[:, :, :, idata] = interpolator(np.dstack([grid_out['x'], grid_out['y'], grid_out['z']]))

    return outsamples





