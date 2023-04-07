"""
This module contains code for estimating resolution achieved by reconstructions.
"""
import logging

import numpy as np

from aspire.numeric import fft
from aspire.utils import grid_2d

logger = logging.getLogger(__name__)


class _FourierCorrelation:
    r"""
    Compute the Fourier correlations between two arrays.

    Underlying data (images/volumes) are assumed to be well aligned.

    The Fourier correlation is defined as:

    .. math::

       c(i) = \frac{ \operatorname{Re}( \sum_i{ \mathcal{F}_1(i) * {\mathcal{F}^{*}_2(i) } } ) }{\
         \sqrt{ \sum_i { | \mathcal{F}_1(i) |^2 } * \sum_i{| \mathcal{F}^{*}_2}(i) |^2 } }
.
    """

    def __init__(self, a, b, pixel_size, cutoff=0.143, eps=1e-4):
        """
        :param a: Input array a, shape(..., *dim).
        :param b: Input array b, shape(..., *dim).
        :param pixel_size: Pixel size in Angstrom.
        :param cutoff: Cutoff value, traditionally `.143`.
        :param eps: Epsilon past boundary values, defaults 1e-4.
        """

        # Sanity checks
        if not hasattr(self, "dim"):
            raise RuntimeError("Subclass must assign `dim`")
        for x in (a, b):
            if not isinstance(x, np.ndarray):
                raise TypeError(f"`{x.__name__}` is not a Numpy array.")

        if not a.dtype == b.dtype:
            raise TypeError(
                f"Mismatched input types {a.dtype} != {b.dtype}. Cast `a` or `b`."
            )
        # TODO, check-math/avoid complex inputs.

        # Shape checks
        if not a.shape[-1] == b.shape[-1]:
            raise RuntimeError(
                f"`a` and `b` appear to have different data axis shapes, {a.shape[-1]} {b.shape[-1]}"
            )

        # To support arbitrary broadcasting simply,
        # we'll force all shapes to be (-1, *(L,)*dim)
        self._a, self._a_stack_shape = self._reshape(a)
        self._b, self._b_stack_shape = self._reshape(b)

        self._analyzed = False
        self.cutoff = cutoff
        self.pixel_size = float(pixel_size)
        self.eps = float(eps)
        self._correlations = None
        self.L = self._a.shape[-1]
        self.dtype = self._a.dtype

    @property
    def _fourier_axes(self):
        return tuple(range(-self.dim, 0))

    def _reshape(self, x):
        """
        Returns `x` with flattened stack axis and `x`'s original stack shape, as determined by `dim`.

        :param x: Numpy ndarray
        """
        # TODO, check 2d in put for dim=2 (singleton case)
        original_stack_shape = x.shape[: -self.dim]
        x = x.reshape(-1, *x.shape[-self.dim :])
        return x, original_stack_shape

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff_correlation):
        self._cutoff = float(cutoff_correlation)
        self._analyzed = False  # reset analysis

    @property
    def correlations(self):
        # There is no need to run this twice if we assume inputs are immutable
        if self._correlations is not None:
            return self._correlations

        # Compute shells from 2D grid.
        radii = grid_2d(self.L, shifted=True, normalized=False, dtype=self.dtype)["r"]

        # Compute centered Fourier transforms,
        #   upcasting when nessecary.
        f1 = fft.centered_fftn(self._a, axes=self._fourier_axes)
        f2 = fft.centered_fftn(self._b, axes=self._fourier_axes)

        # Construct an output table of correlations
        correlations = np.zeros(
            (self.L // 2, self._a.shape[0], self._b.shape[0]), dtype=self.dtype
        )

        inner_diameter = 0.5 + self.eps
        for i in range(0, self.L // 2):
            # Compute ring mask
            outer_diameter = 0.5 + (i + 1) + self.eps
            ring_mask = (radii > inner_diameter) & (radii < outer_diameter)
            logger.debug(f"Shell, Elements:  {i}, {np.sum(ring_mask)}")

            # Mask off values in Fourier space
            r1 = ring_mask * f1
            r2 = ring_mask * f2

            # Compute FRC
            num = np.real(np.sum(r1 * np.conj(r2), axis=self._fourier_axes))
            den = np.sqrt(
                np.sum(np.abs(r1) ** 2, axis=self._fourier_axes)
                * np.sum(np.abs(r2) ** 2, axis=self._fourier_axes)
            )
            # Assign
            correlations[i] = num / den
            # Update ring
            inner_diameter = outer_diameter

        # Repack the table as (_a, _b, L//2)
        correlations = np.swapaxes(correlations, 0, 2)
        # Then unpack the a and b shapes.
        self._correlations = correlations.reshape(
            *self._a_stack_shape, *self._b_stack_shape, self.L // 2
        )
        return self._correlations

    @property
    def estimated_resolution(self):
        """ """
        self.analyze_correlations()
        return self._resolutions

    def analyze_correlations(self):
        """
        Convert from the Fourier Correlations to frequencies and resolution.
        """
        if self._analyzed:
            return

        c_inds = np.zeros(self.correlations.shape[:-1], dtype=int)

        # All correlations are above cutoff,
        #   set index of highest sampled frequency.
        c_inds[np.min(self.correlations, axis=-1) > self.cutoff] = self.L // 2

        # # All correlations are below cutoff,
        # #   set index to 0
        # elif np.max(correlations) < cutoff:
        #     c_ind = 0
        # else:

        # Correlations cross the cutoff.
        # Find the first index of a correlation at `cutoff`.
        c_ind = np.maximum(c_inds, np.argmax(self.correlations <= self.cutoff, axis=-1))

        # Convert indices to frequency (as 1/Angstrom)
        frequencies = self._freq(c_ind)

        # Convert to resolution in Angstrom, smaller is higher frequency.
        self._resolutions = 1 / frequencies

    def _freq(self, k):
        """
        Converts `k` from index of Fourier transform to frequency (as length 1/A).

        From Shannon-Nyquist, for a given pixel-size, sampling theorem limits us to the sampled frequency 1/pixel_size.
        Thus the Bandwidth ranges from `[-1/pixel_size, 1/pixel_size]`,  so the total bandwidth is `2*(1/pixel_size)`.

        Given a real space signal observed with `L` bins (pixels/voxels), each with a `pixel_size` in Angstrom,
        we can compute the width of a Fourier space bin to be the `Bandwidth / L  = (2*(1/pixel_size)) / L`.
        Thus the frequency at an index `k` is `freq_k = k * 2 * (1 / pixel_size) / L  = 2*k / (pixel_size * L)        
        """
        
        # _freq(k) Units: 1 / (pixels * (Angstrom / pixel) = 1 / Angstrom
        # Similar idea to wavenumbers (cm-1).  Larger is higher frequency.
        return k * 2 / (self.L * self.pixel_size)

                 
    def plot(self, to_file=False):
        """
        Generates a Fourier correlation plot.
        """
        

class FourierRingCorrelation(_FourierCorrelation):
    """
    See `_FourierCorrelation`.
    """

    dim = 2


class FourierShellCorrelation(_FourierCorrelation):
    """
    See `_FourierCorrelation`.
    """

    dim = 3
