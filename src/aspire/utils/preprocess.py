import logging
import math
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_1d, grid_2d, grid_3d
from scipy.fftpack import ifftshift, ifft, ifft2, fftshift, fft, fft2, ifftn, fftn
from aspire.utils.fft import centered_fft1, centered_ifft1, centered_fft2, centered_ifft2, centered_fft3, centered_ifft3
from aspire.nfft import Plan
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


def downsample_centered(insamples, szout):
    """
    Blur and downsample 1D to 3D objects such as, curves, images or volumes

    :param insamples: Set of objects to be downsampled in the form of an array, the last dimension
                    is the number of objects.
    :param szout: The desired resolution of for output objects.
    :return: An array consists of the blurred and downsampled objects.
    """

    ensure(insamples.ndim-1 == np.size(szout), 'The number of downsampling dimensions is not the same as that of objects.')

    L_in = insamples.shape[0]
    L_out = szout[0]
    ndata = insamples.shape[-1]
    outdims = np.r_[szout, ndata]
    outsamples = np.zeros(outdims, dtype=insamples.dtype)

    if insamples.ndim == 2:
        # stack of one dimension objects
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
            outsamples[:, idata] = interpolator(np.dstack([grid_out['x']]))

    elif insamples.ndim == 3:
        # stack of two dimension objects
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
        # stack of three dimension objects
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
            outsamples[:, :, :, idata] = interpolator(np.stack((grid_out['x'], grid_out['y'], grid_out['z']), axis=-1))

    else:
        raise RuntimeError('Number of dimensions > 3 for input objects.')

    return outsamples


def crop_pad(mat, n, fill_value=None):
    """
    Reduce the size of a vector, square or cube 'mat' by cropping (or
    increase the size by padding with fill_value, by default zero) to a final
    size of n, (n x n), or (n x n x n) respectively. This is the analogue of downsample, but
    it doesn't change magnification.

    If mat is 2-dimensional and n is a vector, m is cropped to n=[mat_x mat_y].

    The function handles odd and even-sized arrays correctly. The center of
    an odd array is taken to be at (n+1)/2, and an even array is n/2+1.

    If flag is_stack is set to True, then a 3D array 'mat' is treated as a stack of 2D
    images, and each image is cropped to (n x n).

    For 2D images, the input image doesn't have to be square.

    * The original MATLAB function supported cropping to non-square matrices.
      As real-world uses will always crop to square (n, n), we don't support it with Python.


    :param mat: Vector, 2D array, stack of 2D arrays or a 3D array
    :param n: Size of desired cropped vector, side of 2D array or side of 3D array
    :param fill_value: Padding value. Defaults to 0.

    :return: Cropped or padded mat to size of n, (n x n) or (n x n x n)

    """

    num_dimensions = len(mat.shape)

    if num_dimensions not in [1, 2, 3]:
        raise RuntimeError("cropping/padding failed! number of dimensions is too big!"
                           f" ({num_dimensions} while max is 3).")

    if num_dimensions == 2 and 1 in mat.shape:
        num_dimensions = 1

    if fill_value is None:
            fill_value = 0.0

    if num_dimensions == 1:  # mat is a vector for 1D object
        mat = np.reshape(mat, [mat.size, 1])  # force a column vector
        ns = math.floor(mat.size / 2) - math.floor(n / 2)  # shift term for scaling down
        if ns >= 0:  # cropping
            return mat[ns: ns + n].astype('float32')

        else:  # padding
            result_mat = fill_value * np.ones([n, 1], dtype=complex)
            result_mat[-ns: mat.size - ns] = mat
            return result_mat.astype('float32')

    elif num_dimensions == 2:  # mat is 2D image
        mat_x, mat_y = mat.shape
        # start_x = math.floor(mat_x / 2) - math.floor(n / 2)
        start_x = mat_x / 2 - n / 2       # shift term for scaling down
        # start_y = math.floor(mat_y / 2) - math.floor(n / 2)
        start_y = mat_y / 2 - n / 2       # shift term for scaling down

        if start_x >= 0 and start_y >= 0:  # cropping
            start_x, start_y = math.floor(start_x), math.floor(start_y)
            return mat[start_x: start_x + int(n), start_y: start_y + int(n)].astype('float32')

        elif start_x < 0 and start_y < 0:  # padding
            start_x, start_y = math.floor(start_x), math.floor(start_y)
            result_mat = fill_value * np.ones([n, n], dtype=complex)
            result_mat[-start_x: mat_x - start_x, -start_y: mat_y - start_y] = mat
            return result_mat.astype('float32')

        else:
            raise RuntimeError("Can't crop and pad simultaneously!")

    else:  # mat is 3D object

        from_shape = np.array(mat.shape)
        to_shape = np.array((n, n, n))

        ns = np.floor(from_shape / 2) - np.floor(to_shape / 2)
        ns, to_shape = ns.astype(int), to_shape.astype(int)  # can't slice later with float

        if np.all(ns >= 0):   # crop
            return mat[ns[0]: ns[0]+to_shape[0],
                       ns[1]: ns[1]+to_shape[1],
                       ns[2]: ns[2]+to_shape[2]]

        elif np.all(ns <= 0): # pad
            result_mat = fill_value * np.ones([n, n, n], dtype=complex)
            result_mat[-ns[0]: from_shape[0] - ns[0],
                       -ns[1]: from_shape[2] - ns[1],
                       -ns[2]: from_shape[2] - ns[2]] = mat

            return result_mat.astype('float32')

        else:
            raise RuntimeError("Can't crop and pad simultaneously!")


def downsample(insamples, szout, mask=None):
    """
    Blur and downsample 1D to 3D objects such as, curves, images or volumes

    The function handles odd and even-sized arrays correctly. The center of
    an odd array is taken to be at (n+1)/2, and an even array is n/2+1.
    :param insamples: Set of objects to be downsampled in the form of an array, the last dimension
                    is the number of objects.
    :param szout: The desired resolution of for output objects.
    :return: An array consists of the blurred and downsampled objects.
    """

    ensure(insamples.ndim-1 == np.size(szout),
           'The number of downsampling dimensions is not the same as that of objects.')

    L_in = insamples.shape[0]
    L_out = szout[0]
    ndata = insamples.shape[-1]
    outdims = np.r_[szout, ndata]

    outsamples = np.zeros(outdims, dtype=insamples.dtype)

    if mask is None:
        mask = 1.0

    if insamples.ndim == 2:
        # stack of one dimension objects

        for idata in range(ndata):
            insamples_fft = crop_pad(fftshift(fft(insamples[:, idata])), L_out)*mask
            outsamples[:, idata] = np.real(ifft(ifftshift(insamples_fft))*(L_out / L_in))

    elif insamples.ndim == 3:
        # stack of two dimension objects
        for idata in range(ndata):
            insamples_fft = crop_pad(fftshift(fft2(insamples[:, :, idata])), L_out)*mask
            outsamples[:, :, idata] = np.real(ifft2(ifftshift(insamples_fft)) * (L_out**2/L_in**2))

    elif insamples.ndim == 4:
        # stack of three dimension objects
        for idata in range(ndata):
            insamples_fft = crop_pad(fftshift(fftn(insamples[:, :, :, idata])), L_out)*mask
            outsamples[:, :, :, idata] = np.real(ifftn(ifftshift(insamples_fft)) * (L_out**3/L_in**3))

    else:
        raise RuntimeError('Number of dimensions > 3 for input objects.')

    return outsamples


def vol2img(volume, rots, L=None, dtype=None):
    """
    Generate 2D images from the input volume and rotation angles

    The function handles odd and even-sized arrays correctly. The center of
    an odd array is taken to be at (n+1)/2, and an even array is n/2+1.
    :param volume: A 3D volume objects.
    :param rots: A n-by-3-by-3 array of rotation angles.
    :param L: The output size of 2D images.
    :return: An array consists of 2D images.
    """

    if L is None:
        L = np.size(volume, 0)
    if dtype is None:
        dtype = volume.dtype

    lv = np.size(volume, 0)
    if L > lv+1:
        # For compatibility with gen_projections, allow one pixel aliasing.
        # More precisely, it should be N>nv, however, by using nv+1 the
        # results match those of gen_projections.
        if np.mod(L-lv, 2)==1:
            raise RuntimeError('Upsampling from odd to even sizes or vice versa is '
                               'currently not supported')
        dL = np.floor((L-lv)/2)
        fv = centered_fft3(volume)
        padded_volume = np.zeros((L, L, L), dtype=dtype)
        padded_volume[dL+1:dL+lv+1, dL+1:dL+lv+1, dL+1:dL+lv+1] = fv
        volume = centered_ifft3(padded_volume)
        ensure(np.norm(np.imag(volume[:]))/np.norm(volume[:]) < 1.0e-5,
               "The image part of volume is related large (>1.0e-5).")
        #  The new volume size
        lv = L

    grid2d = grid_2d(lv, shifted=True, normalized=False)

    num_pts = lv**2
    num_rots = rots.shape[0]
    pts = np.pi * np.vstack([grid2d['x'].flatten('F'), grid2d['y'].flatten('F'), np.zeros(num_pts)])

    pts_rot = np.zeros((3, num_pts, num_rots))

    for i in range(num_rots):
        pts_rot[:, :, i] = rots[i, :, :].T @ pts

    pts_rot = m_reshape(pts_rot, (3, lv**2*num_rots))

    pts_rot = -2*pts_rot/lv

    im_f = Plan(volume.shape, -pts_rot).transform(volume)

    im_f = m_reshape(im_f, (lv, lv, -1))

    if lv % 2 == 0:
        pts_rot = m_reshape(pts_rot, (3, lv, lv, num_rots))
        im_f = im_f * np.exp(1j*np.sum(pts_rot, 0)/2)
        im_f = im_f * np.expand_dims(np.exp(2*np.pi*1j*(grid2d['x'] +grid2d['y']-1)/(2*lv)), 2)

    im = centered_ifft2(im_f)
    if lv % 2 == 0:
        im = im * m_reshape(np.exp(2*np.pi*1j*(grid2d['x'] +grid2d['y'])/(2*lv)), (lv, lv, 1))

    return np.real(im)


# TODO - out of core version
def cryo_crop(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


def cryo_downsample(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: downsampled x
    """
    dtype_in = x.dtype
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    fourier_dims = np.array([i for i, s in enumerate(out_shape) if 0 < s < in_shape[i]])
    size_in = np.prod(in_shape[fourier_dims])
    size_out = np.prod(out_shape[fourier_dims])

    fx = cryo_crop(np.fft.fftshift(np.fft.fft2(x, axes=fourier_dims), axes=fourier_dims), out_shape)
    out = ifft2(np.fft.ifftshift(fx, axes=fourier_dims), axes=fourier_dims) * (size_out / size_in)
    return out.astype(dtype_in)


def downsample_preprocess(stack, n, mask=None, stack_in_fourier=False):
    """
    A specific downsample version for preprocess for optimized code. In the likely case where preprocess does not use
    crop, the phaseflip_star_file return the images in fourier and downsample_preprocess receives it in fourier.
    These two functions use full sized images, so the fourier transformation is slow. This function also works with C
    aligned images instead of F aligned.
    Args:
        stack: ndarray (N, L, L)
        n: size to downsample to, n<L
        mask: ndarray (L, L)
        stack_in_fourier: Bool (True or False), if true, stack is assumed to be in fourier.

    Returns:
        downsampled_images: ndarray (N, n, n), downsampled images.

    """
    size_in = np.square(stack.shape[1])
    size_out = np.square(n)
    mask = 1 if mask is None else mask
    num_images = stack.shape[0]
    downsampled_images = np.zeros((num_images, n, n), dtype='float32')
    images_batches = np.array_split(np.arange(num_images), 500)
    for batch in images_batches:
        curr_batch = np.array(stack[batch])
        curr_batch = curr_batch if stack_in_fourier else fft2(curr_batch)
        fx = cryo_crop(np.fft.fftshift(curr_batch, axes=(-2, -1)), (-1, n, n)) * mask
        downsampled_images[batch] = ifft2(np.fft.ifftshift(fx, axes=(-2, -1))) * (size_out / size_in)
        print('finished {}/{}'.format(batch[-1] + 1, num_images))
    return downsampled_images


def normalize_background(stack, radius=None):
    n = stack.shape[1]
    radius = n // 2 if radius is None else radius
    circle = ~disc(n, radius)
    background_pixels = stack[circle]
    mean = np.mean(background_pixels, 0)
    std = np.std(background_pixels, 0, ddof=1)
    stack -= mean
    stack /= std
    return stack, mean, std


def global_phaseflip(stack):
    """ Apply global phase flip to an image stack if needed.

    Check if all images in a stack should be globally phase flipped so that
    the molecule corresponds to brighter pixels and the background corresponds
    to darker pixels. This is done by comparing the mean in a small circle
    around the origin (supposed to correspond to the molecule) with the mean
    of the noise, and making sure that the mean of the molecule is larger.

    Examples:
        >> import mrcfile
        >> stack = mrcfile.open('stack.mrcs')
        >> stack = global_phaseflip_stack(stack)

    :param stack: stack of images to phaseflip if needed
    :return stack: stack which might be phaseflipped when needed
    """

    n = stack.shape[0]
    image_center = (n + 1) / 2
    coor_mat_m, coor_mat_n = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    distance_from_center = np.sqrt((coor_mat_m - image_center) ** 2 + (coor_mat_n - image_center) ** 2)

    # calculate indices of signal and noise samples assuming molecule is around the center
    signal_indices = distance_from_center < round(n / 4)
    noise_indices = distance_from_center > round(n / 2 * 0.8)

    signal_mean = np.mean(stack[signal_indices], 0)
    noise_mean = np.mean(stack[noise_indices], 0)

    signal_mean = np.mean(signal_mean)
    noise_mean = np.mean(noise_mean)

    if signal_mean < noise_mean:
        stack *= -1
    return stack


# TODO - maybe these three functions can be added to some general utils as it is being used many times?
def disc(n, r=None, inner=False):
    """
    Return the points inside the circle of radius=r in a square with side n. if inner is True don't return only the
    strictly inside points.
    :param n: integer, the side of the square
    :param r: The radius of the circle (default: n // 2)
    :param inner:
    :return: nd array with 0 outside of the circle and 1 inside
    """
    r = n // 2 if r is None else r
    radiisq = cart2rad(n)
    if inner is True:
        return radiisq < r
    return radiisq <= r


def cart2rad(n):
    """ Compute the radii corresponding to the points of a cartesian grid of size NxN points
        XXX This is a name for this function. """

    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def image_grid(n):
    """
    Return the coordinates of Cartesian points in an nxn grid centered around the origin. The origin of the grid is
    always in the center, for both odd and even n.
    Args:
        n: int, size of grid

    Returns:

    """
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y
