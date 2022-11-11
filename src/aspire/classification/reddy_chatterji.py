import logging

import numpy as np
from skimage.filters import difference_of_gaussians, window
from skimage.transform import rotate, warp_polar

from aspire.numeric import fft
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)

__cache = dict()


def _phase_cross_correlation(img0, img1):
    """
    # Adapted from skimage.registration.phase_cross_correlation

    :param img0: Fixed image.
    :param img1: Translated image.
    :returns: (cross-correlation magnitudes (2D array), shifts)
    """

    # Cache img0 transform, this saves n_classes*(n_nbor-1) transforms
    # Note we use the `id` because ndarray are unhashable
    key = id(img0)
    if key not in __cache:
        __cache[key] = fft.fft2(img0)
    src_f = __cache[key]

    target_f = fft.fft2(img1)

    # Whole-pixel shifts - Compute cross-correlation by an IFFT
    shape = src_f.shape
    image_product = src_f * target_f.conj()
    cross_correlation = fft.ifft2(image_product)

    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    return np.abs(cross_correlation), shifts


def reddy_chatterji_register(
    images, reflection, mask=None, do_cross_corr_translations=True, dtype=None
):
    """
    Compute the Reddy Chatterji method registering images[1:] to images[0].

    This differs from papers and published scikit implimentations by
    computing the fixed base image[0] pipeline once then reusing.

    :param images: Image data (m_img, L, L)
    :param reflection: Image reflections (m_img,)
    :param mask: Support of image. Defaults to disk with radius images.shape[-1]//2.
    :param do_cross_corr_translations: Solve trnaslations by using cross correlation (log polar) method.
    :param dtype: Specify dtype.  Defaults to infer from images.dtype
    :returns: (rotations, shifts, correlations) corresponding to `images`
    """

    if mask is None:
        L = images.shape[-1]
        mask = grid_2d(L, normalized=False)["r"] < L // 2
    if dtype is None:
        dtype = images.dtype

    # Result arrays
    M = len(images)
    rotations = np.zeros(M, dtype=dtype)
    correlations = np.full(M, -np.inf, dtype=dtype)
    shifts = np.zeros((M, 2), dtype=int)

    # De-Mean
    images = images - images.mean(axis=(-1, -2))[:, np.newaxis, np.newaxis]

    # Precompute fixed_img data used repeatedly in the loop below.
    fixed_img = images[0]
    # Difference of Gaussians (Band Filter)
    fixed_img_dog = difference_of_gaussians(fixed_img, 1, 4)
    # Window Images (Fix spectral boundary)
    wfixed_img = fixed_img_dog * window("hann", fixed_img.shape)
    # Transform image to Fourier space
    fixed_img_fs = np.abs(fft.fftshift(fft.fft2(wfixed_img))) ** 2
    # Compute Log Polar Transform
    radius = fixed_img_fs.shape[0] // 8  # Low Pass
    warped_fixed_img_fs = warp_polar(
        fixed_img_fs,
        radius=radius,
        output_shape=fixed_img_fs.shape,
        scaling="log",
    )
    # Only use half of FFT, because it's symmetrical
    warped_fixed_img_fs = warped_fixed_img_fs[: fixed_img_fs.shape[0] // 2, :]

    # Now prepare for rotating original images,
    #   and searching for translations.
    # We start back at the raw fixed_img.
    twfixed_img = fixed_img * window("hann", fixed_img.shape)

    # Register image `m` against images[0]
    for m in range(1, len(images)):
        # Get the image to register
        regis_img = images[m]

        # Reflect images when necessary
        if reflection[m]:
            regis_img = np.flipud(regis_img)

        # Difference of Gaussians (Band Filter)
        regis_img_dog = difference_of_gaussians(regis_img, 1, 4)

        # Window Images (Fix spectral boundary)
        wregis_img = regis_img_dog * window("hann", regis_img.shape)

        # Transform image to Fourier space
        regis_img_fs = np.abs(fft.fftshift(fft.fft2(wregis_img))) ** 2

        # Compute Log Polar Transform
        warped_regis_img_fs = warp_polar(
            regis_img_fs,
            radius=radius,  # Low Pass
            output_shape=fixed_img_fs.shape,
            scaling="log",
        )

        # Only use half of FFT, because it's symmetrical
        warped_regis_img_fs = warped_regis_img_fs[: fixed_img_fs.shape[0] // 2, :]

        # Compute the Cross_Correlation to estimate rotation
        # Note that _phase_cross_correlation uses the mangnitudes (abs()),
        #  ie it is using both freq and phase information.
        cross_correlation, _ = _phase_cross_correlation(
            warped_fixed_img_fs, warped_regis_img_fs
        )

        # Rotating Cartesian space translates the angular log polar component.
        # Scaling Cartesian space translates the radial log polar component.
        # In common image resgistration problems, both components are used
        #   to simultaneously estimate scaling and rotation.
        # Since we are not currently concerned with scaling transformation,
        #   disregard the second axis of the `cross_correlation` returned by
        #   `_phase_cross_correlation`.
        cross_correlation_score = cross_correlation[:, 0].ravel()

        # Recover the angle from index representing maximal cross_correlation
        recovered_angle_degrees = (360 / regis_img_fs.shape[0]) * np.argmax(
            cross_correlation_score
        )

        # The recovered angle represents an estimate of the rotation from reference to image[m].
        # The registration angle for image[m],
        #   the angle to apply to image[m] to register with reference,
        #   would be the negation of this,
        r = -1 * recovered_angle_degrees

        # For now, try the hack below, attempting two cases ...
        # Some papers mention running entire algos /twice/,
        #   when admitting reflections, so this hack is not
        #   the worst you could do :).
        # Hack
        regis_img_estimated = rotate(regis_img, r)
        regis_img_rotated_p180 = rotate(regis_img, r + 180)
        da = np.dot(fixed_img[mask], regis_img_estimated[mask])
        db = np.dot(fixed_img[mask], regis_img_rotated_p180[mask])
        if db > da:
            regis_img_estimated = regis_img_rotated_p180
            r += 180

        # Assign estimated rotations results
        rotations[m] = r * np.pi / 180  # Convert to radians

        if do_cross_corr_translations:
            # Prepare for searching over translations using cross-correlation with the rotated image.
            twregis_img = regis_img_estimated * window("hann", regis_img.shape)
            cross_correlation, shift = _phase_cross_correlation(
                twfixed_img, twregis_img
            )

            # Compute the shifts as integer number of pixels,
            shift_x, shift_y = int(shift[1]), int(shift[0])
            # then apply the shifts
            regis_img_estimated = np.roll(regis_img_estimated, shift_y, axis=0)
            regis_img_estimated = np.roll(regis_img_estimated, shift_x, axis=1)
            # Assign estimated shift to results
            shifts[m] = shift[::-1].astype(int)

        else:
            shift = None  # For logger line

        # Estimated `corr` metric
        corr = np.dot(fixed_img[mask], regis_img_estimated[mask])
        correlations[m] = corr

    # Cleanup some cached stuff for this class
    __cache.pop(id(warped_fixed_img_fs), None)
    __cache.pop(id(twfixed_img), None)

    return rotations, shifts, correlations
