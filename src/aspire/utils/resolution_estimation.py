"""
This module contains code for estimating resolution achieved by reconstructions.
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np

from aspire import nufft, numeric
from aspire.utils import grid_2d, grid_3d

logger = logging.getLogger(__name__)

# FourierCorrelation holds a single implementation for both FSC and
# FRC based on dimension `dim`.


class FourierCorrelation:
    r"""
    Compute the Fourier correlations between two arrays.

    Underlying data (images/volumes) are assumed to be well aligned.

    The Fourier correlation is defined as:

    .. math::

       c(i) = \frac{ \operatorname{Re}( \sum_i{ \mathcal{F}_1(i) * {\mathcal{F}^{*}_2(i) } } ) }{\
         \sqrt{ \sum_i { | \mathcal{F}_1(i) |^2 } * \sum_i{| \mathcal{F}^{*}_2}(i) |^2 } }

    This implementation supports Numpy style broadcasting resulting in
    up to two stack dimensions.  For example, to compute all pairs
    supply signal with stack shapes (m,1) and (1,n) to yield an (m,n)
    table of results. Note that plotting is limited to a single
    reference signal.
    """

    def __init__(self, a, b, pixel_size=None, method="fft"):
        """
        :param a: Input array a, shape(..., *dim).
        :param b: Input array b, shape(..., *dim).
        :param pixel_size: Pixel size in angstrom.
            Default `None` implies "pixel" units.
        :param method: Selects either 'fft' (on Cartesian grid),
            or 'nufft' (on polar grid). Defaults to 'fft'.
        """

        # Sanity checks
        if not hasattr(self, "dim"):
            raise RuntimeError("Subclass must assign `dim`")
        for x in (a, b):
            if not isinstance(x, np.ndarray):
                raise TypeError(f"`{x}` is not a Numpy array.")

        if not a.dtype == b.dtype:
            raise TypeError(
                f"Mismatched input types {a.dtype} != {b.dtype}. Cast `a` or `b`."
            )
        # TODO, check-math/avoid complex inputs.

        # Shape checks
        if not a.shape[-self.dim :] == b.shape[-self.dim :]:
            raise RuntimeError(
                f"`a` and `b` appear to have different data axis shapes, {a.shape} {b.shape}"
            )

        # Method selection
        methods = {"fft": self._fft_correlations, "nufft": self._nufft_correlations}
        if method not in methods:
            raise RuntimeError(
                f"Requested method {method} not in available methods {list(methods.keys())}."
            )
        self.method = method
        self._correlation_method = methods[self.method]

        # To support arbitrary broadcasting simply,
        # we'll force all shapes to be (-1, *(L,)*dim)
        # and keep track of the stack shapes.
        self.a = a
        self.b = b
        self._a, self._a_stack_shape = self._reshape(a)
        self._b, self._b_stack_shape = self._reshape(b)
        self._result_stack_shape = np.broadcast_shapes(
            self._a_stack_shape, self._b_stack_shape
        )

        # Handle `pixel_size` and `pixel_mode`
        self._pixel_units = "angstrom"
        if pixel_size is None:
            pixel_size = 1.0
            self._pixel_units = "pixels"
        self.pixel_size = float(pixel_size)

        self._correlations = None
        self.L = self._a.shape[-1]
        self.dtype = self._a.dtype

    @property
    def _fourier_axes(self):
        """
        Returns tuple representing the axes containing signal data
        based on dimension `dim`.
        """
        return tuple(range(-self.dim, 0))

    def _reshape(self, x):
        """
        Returns `x` with flattened stack axis and `x`'s original stack
        shape, as determined by `dim`.

        :param x: Numpy ndarray

        :return: (stack flattened x, x_stack_shape)
        """
        # TODO, check 2d input for dim=2 (singleton case)
        original_stack_shape = x.shape[: -self.dim]
        x = x.reshape(-1, *x.shape[-self.dim :])
        return x, original_stack_shape

    @property
    def correlations(self):
        """
        Compute and return the Fourier correlations of signal stacks a
        cross b.

        :return: Numpy array
        """
        # Cache _correlations.
        # There is no need to run this twice assuming inputs are immutable.
        if self._correlations is None:
            # Compute the correlations
            self._correlations = self._correlation_method()

        return self._correlations

    def _fft_correlations(self):
        """
        Computes Fourier correlations using the FFT on a Cartesian grid.
        """

        # Compute shells from 2D grid.
        if self.dim == 2:
            grid_function = grid_2d
        elif self.dim == 3:
            grid_function = grid_3d

        radii = grid_function(self.L, shifted=True, normalized=False, dtype=self.dtype)[
            "r"
        ]

        # Compute centered Fourier transforms.
        f1 = numeric.fft.centered_fftn(self.a, axes=self._fourier_axes)
        f2 = numeric.fft.centered_fftn(self.b, axes=self._fourier_axes)

        # Construct an output table of correlations
        correlations = np.zeros(
            (self.L // 2, *self._result_stack_shape), dtype=self.dtype
        )

        inner_diameter = 0.5
        for i in range(0, self.L // 2):
            # Compute ring mask
            outer_diameter = 0.5 + (i + 1)
            ring_mask = (radii > inner_diameter) & (radii < outer_diameter)
            logger.debug(f"Shell, Elements:  {i}, {np.sum(ring_mask)}")

            # Mask off values in Fourier space
            r1 = ring_mask * f1
            r2 = ring_mask * f2

            # Compute Fourier correlations
            num = np.real(np.sum(r1 * np.conj(r2), axis=self._fourier_axes))
            den = np.sqrt(
                np.sum(np.abs(r1) ** 2, axis=self._fourier_axes)
                * np.sum(np.abs(r2) ** 2, axis=self._fourier_axes)
            )
            # Assign
            correlations[i] = num / den
            # Update ring
            inner_diameter = outer_diameter

        # Repack the table as (..., L//2)
        correlations = np.swapaxes(correlations, 0, -1)
        return correlations.reshape(*self._result_stack_shape, self.L // 2)

    def _nufft_correlations(self):
        """
        Computes Fourier correlations using the NUFFT on a polar grid.
        """

        # TODO, we could use an internal tool (Polar2D?) for this.
        # L//2 is intentionally used for compatibility with Cartesian grid.
        #   This avoids having to have multiple methods for computing resolutions later.
        r = np.linspace(0, np.pi, self.L // 2, endpoint=False, dtype=self.dtype)
        phi = np.linspace(0, 2 * np.pi, 2 * self.L, endpoint=False, dtype=self.dtype)
        if self.dim == 2:
            # 2D Polar points
            x = r[:, np.newaxis] * np.cos(phi[np.newaxis, :])
            y = r[:, np.newaxis] * np.sin(phi[np.newaxis, :])
            # Because the values will be summed later, ordering does not matter.
            fourier_pts = np.vstack((x.flatten(), y.flatten()))

        elif self.dim == 3:
            # 3D Spherical points
            theta = np.linspace(0, np.pi, self.L, endpoint=False, dtype=self.dtype)
            x = (
                r[:, np.newaxis, np.newaxis]
                * np.sin(theta[np.newaxis, :, np.newaxis])
                * np.cos(phi[np.newaxis, np.newaxis, :])
            )
            y = (
                r[:, np.newaxis, np.newaxis]
                * np.sin(theta[np.newaxis, :, np.newaxis])
                * np.sin(phi[np.newaxis, np.newaxis, :])
            )
            z = (
                r[:, np.newaxis, np.newaxis]
                * np.cos(theta[np.newaxis, :, np.newaxis])
                * np.ones((1, 1, 2 * self.L), dtype=self.dtype)
            )
            # Because the values will be summed later, ordering does not matter.
            fourier_pts = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        else:
            raise NotImplementedError(
                "`nufft` based correlations only implemented for dimensions 2 and 3."
            )

        # Stack signal data to create a larger NUFFT problem (better performance).
        #   Note, we want a complex result.
        signal = np.vstack((self._a, self._b))
        # Compute one large NUFFT for all the signal frames,
        f = nufft.nufft(signal, fourier_pts, real=False)
        # then unpack as two 1D stacks of the polar grid points, one for _a and _b.
        f = f.reshape(self._a.shape[0] + self._b.shape[0], len(r), -1)
        f1, f2 = np.vsplit(f, [self._a.shape[0]])

        # Compute the Fourier correlations.
        cov = np.sum(f1 * np.conj(f2), -1).real
        norm1 = np.sqrt(np.sum(np.abs(f1) ** 2, -1))
        norm2 = np.sqrt(np.sum(np.abs(f2) ** 2, -1))

        correlations = cov / (norm1 * norm2)

        # Then unpack as original a and b broadcasted shapes.
        return correlations.reshape(*self._result_stack_shape, r.shape[-1])

    def analyze_correlations(self, cutoff):
        """
        Convert from the Fourier correlations to frequencies and resolution.

        :param cutoff: Cutoff value, traditionally `.143`.
            Note `cutoff=None` evaluates as `cutoff=1`.
        """
        # Handle optional cutoff plotting.
        if cutoff is None:
            cutoff = 1

        cutoff = float(cutoff)
        if not (0 <= cutoff <= 1):
            raise ValueError("Supplied correlation `cutoff` not in [0,1], {cutoff}")

        c_inds = np.zeros(self.correlations.shape[:-1], dtype=int)

        # All correlations are above cutoff,
        #   set index of highest sampled frequency.
        c_inds[np.min(self.correlations, axis=-1) > cutoff] = self.L // 2

        # Correlations cross the cutoff.
        # Find the first index of a correlation at `cutoff`.
        # Should return 0 if not found, which corresponded to the case
        # where all correlations are below cutoff.
        c_ind = np.maximum(c_inds, np.argmax(self.correlations <= cutoff, axis=-1))

        # Convert indices to frequency (as 1/angstrom)
        frequencies = self._freq(c_ind)

        with warnings.catch_warnings():
            # When using high cutoff (eg. 1) it is possible `frequencies`
            # contains 0; capture and ignore that division warning.
            warnings.filterwarnings("ignore", r".*divide by zero.*")
            # Convert to resolution in angstrom, smaller is higher frequency.
            self._resolutions = 1 / frequencies

        return self._resolutions

    def _freq(self, k):
        """
        Converts `k` from index of Fourier transform to frequency (as
        length 1/A).

        :param k: Frequency index, integer or Numpy array of ints.
        :return: Frequency in 1/angstrom.
        """

        # _freq(k) Units: 1 / (pixels * (angstrom / pixel) = 1 / angstrom
        # Similar to wavenumbers.  Larger is higher frequency.
        return k / (self.L * self.pixel_size)

    def plot(self, cutoff=None, save_to_file=False, labels=None):
        """
        Generates a Fourier correlation plot.

        :param cutoff: Cutoff value, traditionally `.143`.
            Default `None` implies `cutoff=1` and excludes
            plotting cutoff line.
        :param save_to_file: Optionally, save plot to file.
            Defaults False, enabled by providing a string filename.
            User is responsible for providing reasonable filename.
            See `https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html`.
        """

        # Handle optional cutoff plotting.
        _plot_cutoff = True
        if cutoff is None:
            cutoff = 1
            _plot_cutoff = False

        if not (0 <= cutoff <= 1):
            raise ValueError("Supplied correlation `cutoff` not in [0,1], {cutoff}")

        # Construct x-axis labels
        x_inds = np.arange(self.correlations.shape[-1])
        freqs = self._freq(x_inds)
        # TODO: handle zero freq better
        with np.errstate(divide="ignore"):
            freqs_units = 1 / freqs

        # Check we're asking for a reasonable plot.
        stack = self.correlations.shape[:-1]
        if len(stack) > 2:
            raise RuntimeError(
                f"Unable to plot figure tables with more than 2 dim, stack shape {stack}. Try reducing to a simpler request."
            )

        if (
            self._a_stack_shape[0] > 1 and self._a_stack_shape != self._b_stack_shape
        ) or (len(stack) == 2 and 1 not in stack):
            raise RuntimeError(
                f"Unable to plot figure tables with more than 1 reference figures, stack shape {stack}. Try reducing to a simpler request."
            )

        # Check `labels` length when provided.
        if labels is not None:
            if len(labels) != len(self.correlations):
                raise ValueError(
                    f"Check `labels`. Provided len(labels) != len(self.correlations): {len(labels)} != {len(self.correlations)}."
                )

        plt.figure(figsize=(8, 6))
        plt.title(self._plot_title)
        plt.xlabel(f"Resolution ({self._pixel_units})")
        plt.ylabel("Correlation")
        plt.ylim([0, 1.1])
        for i, line in enumerate(self.correlations):
            # Set default label for single correlation (required by plt.legend() below).
            _label = "correlation"
            if len(self.correlations) > 1:
                _label = f"{i}"
                if labels is not None:
                    _label = labels[i]
            plt.plot(freqs_units, line, label=_label)

        estimated_resolution = self.analyze_correlations(cutoff)[0]

        # Display cutoff
        if _plot_cutoff:
            plt.axhline(y=cutoff, color="r", linestyle="--", label=f"cutoff={cutoff}")

            # Display resolution
            plt.axvline(
                x=estimated_resolution,
                color="b",
                linestyle=":",
                label=f"Resolution={estimated_resolution:.3f}",
            )

        # x-axis decreasing
        plt.gca().invert_xaxis()
        plt.legend(title=f"Method: {self.method}")

        if save_to_file:
            plt.savefig(save_to_file)
            plt.close()
        else:
            plt.show()


# The following are user facing classes, and simply wrap
# `FourierCorrelation` after assigning dimension `dim` and any
# dimension specific variables.


class FourierRingCorrelation(FourierCorrelation):
    """
    See `FourierCorrelation`.
    """

    dim = 2
    _plot_title = "Fourier Ring Correlation"


class FourierShellCorrelation(FourierCorrelation):
    """
    See `FourierCorrelation`.
    """

    dim = 3
    _plot_title = "Fourier Shell Correlation"
