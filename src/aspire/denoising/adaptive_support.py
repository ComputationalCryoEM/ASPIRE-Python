import logging

import numpy as np

from aspire.image import Image
from aspire.numeric import fft
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)


def adaptive_support(images, energy_threshold=0.99):
    """
    Determine size of the compact support in both real and Fourier Space.

    Images are scaled in real space so that the noise variance in both
    real and Fourier domains is similar.

    Returns c_limit (support in Fourier space), R_limit (support in real space).

    Fourier c_limit is scaled in range [0, 0.5].
    R_limit is in pixels [0, Image.res/2].

    :param images: Input `Image` stack
    :param energy_threshold: [0, 1] threshold limit
    :return: (c_limit, R_limit)
    """

    # Sanity Check Input is Image stack
    if not isinstance(images, Image):
        raise RuntimeError("adaptive_support expects Image instance")

    # Sanity Check Threshold is in range
    if energy_threshold <= 0 or energy_threshold > 1:
        raise ValueError(
            f"Given energy_threshold {energy_threshold} outside sane range [0,1]"
        )

    L = images.res
    N = L // 2
    P = images.n_images

    r = grid_2d(L, shifted=False, normalized=False, dtype=images.dtype)["r"]

    # Transform to Fourier space
    imgf = fft.centered_fft2(images.asnumpy())

    # demean
    img = images.asnumpy() * L
    mean = np.mean(img, axis=0)
    img -= mean

    # Construct a mask of corner values, outside max radiaus N
    corner_mask = r.flatten() > N
    img_corner = img.reshape(P, L * L)[:, corner_mask]
    imgf_corner = imgf.reshape(P, L * L)[:, corner_mask]

    # Note that the noise theoretically same, but in practice differ
    # Take smaller so we do not yield negative variance or PSD later
    noise_var = min(np.var(img_corner), np.var(imgf_corner))

    variance_map = np.var(
        img,
        axis=0,
        ddof=1,
    )  # Note ddof=1 employed to match MATLAB [] vector syntax

    # Compute the Radial Variance
    radial_var = np.zeros(N)
    for i in range(N):
        mask = (r >= i) & (r < i + 1)
        radial_var[i] = np.mean(variance_map[mask])

    # Compute the Radial Power Spectrum
    img_ps = np.abs(imgf) ** 2
    pspec = np.mean(img_ps, axis=0)
    radial_pspec = np.zeros(N)
    for i in range(N):
        radial_pspec[i] = np.mean(pspec[(r >= i) & (r < i + 1)])

    # Subtract the noise variance
    radial_pspec -= noise_var
    radial_var -= noise_var

    # Construct range of Fourier limits
    c = np.linspace(0, 0.5, N)
    # Construct range of Real limits
    R = np.arange(N, dtype=int)

    # Calculate cumulative energy
    cum_pspec = np.zeros(N)
    cum_var = np.zeros(N)
    for i in range(N):
        cum_pspec[i] = np.sum(radial_pspec[: i + 1] * c[: i + 1])
        cum_var[i] = np.sum(radial_var[: i + 1] * R[: i + 1])

    # Normalize energies [0,1]
    cum_pspec /= cum_pspec[-1]
    cum_var /= cum_var[-1]

    # Note legacy code *=L for Fourier limit,
    #   but then only uses divided by L... so just removed here.
    # This makes it consistent with Nyquist, ie [0, .5]
    c_limit = c[np.argwhere(cum_pspec > energy_threshold)[0, 0]]
    R_limit = int(R[np.argwhere(cum_var > energy_threshold)[0, 0]])

    return c_limit, R_limit
