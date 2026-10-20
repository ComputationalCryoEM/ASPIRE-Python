"""
Contains code supporting CTF parameter estimation.
This implements a version of ASPIRE-CTF originally written in MATLAB,

This Python version supports multiple basis and solvers.

See paper:

  |    "Reducing bias and variance for CTF estimation in single particle cryo-EM"
  |    Ayelet Heimowitz, Joakim Andén, Amit Singer
  |    Ultramicroscopy, Volume 212, 2020
  |    https://doi.org/10.1016/j.ultramic.2020.112950.

Note:
``CtfEstimator`` computes the background as a monotonically decreasing
function of spatial frequency. This practice may lead to an inaccurate
background estimation for experimental images produced using a K2
camera in counting mode, as the background in this case is not
monotonically decreasing. Despite this, CTF parameters are captured
successfully in such situations.

Created on Sep 10, 2019
@author: Ayelet Heimowitz, Amit Moscovich

Originally integrated into ASPIRE-Python Feb 2021,
optimized and refactored by Garrett Wright Oct 2026.

"""

import logging
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from numpy import linalg as npla
from scipy.optimize import linprog, minimize
from scipy.signal.windows import dpss

from aspire.basis import Coef, FFBBasis2D, SteerableBasis2D
from aspire.image import Image
from aspire.numeric import fft, xp
from aspire.operators import CTFFilter
from aspire.storage import StarFile
from aspire.utils import grid_2d, tqdm, voltage_to_wavelength

logger = logging.getLogger(__name__)


class CtfEstimator:
    """
    CtfEstimator Class ...
    """

    # Allow configuration for advanced users and future solvers.
    default_solver_opts = {
        "gd": {
            "gradient_descent_iterations": 6,
            "max_iterations": 400,
            "stop_cond": None,  # Base on dtype
        },
        "bfgs": {
            "maxiter": 200,
            "gtol": 1e-10,
        },
    }

    _solver_methods = default_solver_opts.keys()

    def __init__(
        self,
        pixel_size,
        cs,
        amplitude_contrast,
        voltage,
        psd_size,
        num_tapers=4,
        g_min=30,
        g_max=5,
        min_defocus=5000,
        max_defocus=100000,
        step_defocus=10,
        basis=None,
        solver_method="bfgs",
        solver_opts=None,
        linprog_method="highs-ds",
        diagnostic_plots=False,
        dtype=np.float32,
    ):
        """
        Instantiate a CtfEstimator instance.

        :param pixel_size: Size of the pixel in \u212b (angstrom).
        :param cs: Spherical aberration in mm.
        :param amplitude_contrast: Amplitude contrast.
        :param voltage: Voltage of electron microscope.
        :param psd_size: Block size (in pixels) for PSD estimation.
        :param num_tapers: Optional, number of tapers to apply in PSD estimation.
            Defaults to 4.
        :param g_min: Inverse of minimun resolution for PSD. Default 30.
        :param g_max: Inverse of maximum resolution for PSD. Default 5.
        :param min_defocus: Start of defocus loop scan (Angstrom). Default 5000.
        :param max_defocus: End of defocus loop scan (Angstrom). Default 100000.
        :param step_defocus: Step size of defocus loop scan (Angstrom). Default 10.
        :param basis: Optional user provided SteerableBasis2D.  Defaults to `FFBBasis2D`.
        :param solver_method: Solver used to optimize defocus params.  `gd` or `bfgs`.
            Default uses single call to `bfgs`.
            Original implementation used repeated gradient descent and is provided for reproducibility.
            Each solver may have unique configuration options, optionally passed via a dictionary.
        :param solver_opts: Optional dictionary of solver options.
            Populated with `default_solver_opts[solver]` by default.
        :param linprog_method: Method passed to linear program solver (scipy.optimize.linprog).
            Defaults to 'highs-ds'.
        :param diagnostic_plots: Optionally generate diagnostic_plots.
            Note, these are blocking plot calls.
            Defaults to False.
        :returns: CtfEstimator instance.
        """

        self.pixel_size = np.asarray(pixel_size).item()
        self.cs = np.asarray(cs).item()
        self.amplitude_contrast = np.asarray(amplitude_contrast).item()
        self.voltage = np.asarray(voltage).item()
        self.psd_size = int(psd_size)
        self.num_tapers = int(num_tapers)
        self.lmbd = (
            voltage_to_wavelength(np.asarray(voltage).item()) / 10.0
        )  # Convert angstrom to nm

        self.g_min = np.asarray(g_min).item()
        self.g_max = np.asarray(g_max).item()
        self.min_defocus = np.asarray(min_defocus, dtype=int).item()
        self.max_defocus = np.asarray(max_defocus, dtype=int).item()
        self.step_defocus = np.asarray(step_defocus, dtype=int).item()

        # Setup solvers and respective solver options
        if solver_method not in self._solver_methods:
            raise RuntimeError(
                f"Provided solver_method {solver_method} must be one of {self._solver_methods}"
            )
        self.solver_method = solver_method
        if solver_opts is None:
            solver_opts = self.default_solver_opts[solver_method]
        self.solver_opts = solver_opts
        self.linprog_method = linprog_method

        self.diagnostic_plots = diagnostic_plots
        self.dtype = np.dtype(dtype)

        # Grids
        grid = grid_2d(psd_size, normalized=True, indexing="yx", dtype=self.dtype)
        # Note range is -half to half.
        self.r_ctf = grid["r"] / 2 * (10 / pixel_size)  # units: inverse nm
        self.theta = grid["phi"]
        # Construct omega grids for use in ASPIRE's ctf_formula
        self.omega = np.pi * np.vstack((grid["x"].flatten(), grid["y"].flatten()))
        center = self.psd_size // 2
        r = np.pi * grid["r"][center, center:]
        self.radial_omega = np.stack((r, np.zeros_like(r)))  # 1d (ray)

        # Re-scale to unnormalized grids for X, Y, and R (avoids recompute)
        self.X = grid["x"] * psd_size / 2
        self.Y = grid["y"] * psd_size / 2
        self.R = grid["r"] * psd_size / 2

        # Setup Basis
        if basis is None:
            basis = FFBBasis2D((psd_size, psd_size), 2, dtype=dtype)
        if not isinstance(basis, SteerableBasis2D):
            raise RuntimeError("Basis must be subclass of SteerableBasis2D.")
        self.basis = basis

        # Only need to compute tapers once
        # Relating to MATLAB code, NW:= R * N / 2 = (2*L)/N * N / 2 = L
        tapers_1d = self.tapers(psd_size, num_tapers, num_tapers)
        self.tapers_1d = xp.asarray(tapers_1d, dtype=self.dtype)

    def micrograph_to_blocks(self, micrograph):
        """
        Preprocess micrograph into blocks using block_size.

        :param micrograph: Micrograph as NumPy array. #NOTE looks like F order
        :param blocksize: Size of the square blocks to partition micrograph.
        :return: NumPy array of blocks extracted from the micrograph.
        """

        # verify self.psd_size is even
        assert self.psd_size % 2 == 0

        if micrograph.ndim == 3:
            assert (
                micrograph.shape[0] == 1
            ), f"micrograph should be 2D or stack of 1 2D image: {micrograph.shape}"
            micrograph = micrograph[0]

        size_x = micrograph.shape[-1]
        size_y = micrograph.shape[-2]

        step_size = self.psd_size // 2
        range_y = size_y // step_size - 1
        range_x = size_x // step_size - 1

        block_list = [
            micrograph[
                j * step_size : (j + 2) * step_size, i * step_size : (i + 2) * step_size
            ]
            for j in range(range_y)
            for i in range(range_x)
        ]
        blocks = np.asarray(block_list, dtype=micrograph.dtype)

        return blocks

    def normalize_blocks(self, blocks):
        """
        Preprocess CTF of micrograph using block_size.

        :param blocks: NumPy array of blocks extracted from the micrograph.
        :return: NumPy array of normalized blocks.
        """

        # Take block size from blocks
        block_size = blocks.shape[1]
        assert block_size == blocks.shape[2]

        # Create a sum and reshape so it may be broadcast with `block`.
        blocks_sum = np.sum(blocks, axis=(-1, -2))[:, np.newaxis, np.newaxis]

        blocks -= blocks_sum / (block_size**2)

        return blocks

    def preprocess_micrograph(self, micrograph):
        """
        Preprocess micrograph into normalized blocks using block_size.

        :param micrograph: Micrograph as NumPy array.
        :return: NumPy array of normalized blocks extracted from the micrograph.
        """

        return self.normalize_blocks(self.micrograph_to_blocks(micrograph))

    def legacy_preprocess_micrograph(self, micrograph):
        """
        Returns half-overlapping blocks extracted from the micrograph.
        Equivalent to MATLAB's ctf_extract_blocks.

        Note, internally transposes micrograph to match MATLAB computations.
        Returns (num_blocks, block_size, block_size)

        :param micrograph: Micrograph as NumPy array.
        :return: NumPy array of normalized blocks extracted from the micrograph.
        """
        h, w = micrograph.shape

        if min(h, w) < self.psd_size:
            raise ValueError(
                f"Blocks of size {self.psd_size} x {self.psd_size} cannot be extracted from image."
            )

        block_size = self.psd_size - (self.psd_size % 2)  # force even
        half = block_size // 2

        # Compute crop bounds
        end_val1 = (h // block_size) * block_size + (
            half if (h % block_size) > half else 0
        )
        end_val2 = (w // block_size) * block_size + (
            half if (w % block_size) > half else 0
        )

        # Crop
        cropped_micrograph = micrograph[:end_val1, :end_val2]

        # Compute number of half blocks
        n_bh = end_val1 // half  # number of half-blocks along rows
        n_bw = end_val2 // half  # number of half-blocks along cols
        n_total = n_bh * n_bw

        # im2col `distinct` extracts non-overlapping half blocks
        # column-major block ordering to match MATLAB (down-first)
        # but keeping now for diagnostics.
        blocks = (
            cropped_micrograph.reshape(n_bh, half, n_bw, half)
            .transpose(1, 3, 0, 2)
            .reshape(half * half, n_total, order="F")
        )

        # Reshape to (half, half, n_total) so each slice is one half-block
        blocks = blocks.reshape(half, half, n_total, order="F")

        # First cat
        # stack each half-block with the one below it (rows overlap)
        # by shifting the block indices by 1
        row_shifted = np.concatenate([blocks[:, :, 1:], blocks[:, :, :1]], axis=2)
        blocks = np.concatenate(
            [blocks, row_shifted], axis=0
        )  # (block_size, half, n_total)

        # Second cat
        # stack each block with the one to its right (cols overlap)
        # by rolling the blocks
        col_shifted = np.roll(blocks, -n_bh, axis=2)
        blocks = np.concatenate(
            [blocks, col_shifted], axis=1
        )  # (block_size, block_size, n_total)

        # Reshape to blocksize-by-blocksize grid
        n_row_blocks = int(np.floor(2 * h / block_size))
        n_col_blocks = int(np.floor(2 * w / block_size))
        blocks = blocks.reshape(
            block_size, block_size, n_row_blocks, n_col_blocks, order="F"
        )

        # Drop last row and col (bounds artifacts)
        blocks = blocks[:, :, :-1, :-1]
        # Flatten
        block = blocks.reshape(block_size, block_size, -1, order="F")

        # De-mean each block
        block_mean = block.sum(axis=(0, 1), keepdims=True) / (block_size**2)
        block = block - block_mean

        # Transpose output to (n,block_size, block_size)
        return block.T

    def tapers(self, N, NW, L):
        """
        Compute data tapers (which are discrete prolate spheroidal sequences (dpss))

        Uses scipy implementation, see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.dpss.html

        :param N: Size of each taper
        :param NW: Half Bandwidth
        :param L: Number of tapers
        :return: NumPy array of data tapers
        """

        # Note the original ASPIRE MATLAB implementation computed these.
        # This call should be close up to sign, but is not identical.
        #
        # In the CtfEstimator application, the signs will pass through
        # the mul and fft steps, then are absolute valued.
        return dpss(M=N, NW=NW, Kmax=L, return_ratios=False, sym=False, norm=2)

    def estimate_psd(self, blocks):
        """
        Estimate the power spectrum of the micrograph using the multi-taper method

        :param blocks: 3-D NumPy array containing windows extracted from the micrograph in the preprocess function.
        :return: NumPy array of estimated power spectrum.
        """

        # Shapes
        num_blocks, N, _ = blocks.shape
        Nhalf = N // 2 + 1  # rfft size
        num_1d_tapers = self.tapers_1d.shape[0]

        # Ensure blocks are on GPU and consistent dtype
        blocks = xp.asarray(blocks, dtype=self.dtype)

        # Array to accumulate PSD
        # Use hermitian symmetry to cut PSD compute in half
        psd_half = xp.zeros((N, Nhalf), dtype=self.dtype)
        accum = xp.empty((N, Nhalf), dtype=self.dtype)

        for ax2 in range(num_1d_tapers):
            # taper along columns
            A = fft.rfft(
                blocks * self.tapers_1d[ax2][None, None, :], axis=-1
            )  # (num_1d_tapers, N, Nhalf)

            for ax1 in range(num_1d_tapers):
                # taper along rows
                Bf = fft.fft(
                    A * self.tapers_1d[ax1][None, :, None], axis=-2
                )  # (num_1d_tapers, N, Nhalf)
                # Accumulate abs2 summed over all blocks into temp array `accum`
                _abs2_sum0(Bf, accum)
                psd_half += accum

        # Rebuild the full spectrum from psd_half
        amplitude_spectrum = xp.empty((N, N), dtype=self.dtype)
        amplitude_spectrum[:, :Nhalf] = psd_half
        # reflect all the positive freq columns (1:N - Nhalf + 1), roll the single zero column.
        amplitude_spectrum[:, Nhalf:] = xp.roll(
            psd_half[:, 1 : N - Nhalf + 1][::-1, ::-1], 1, axis=0
        )

        # Normalize
        # Ask Ayelet about the num_blocks factor. Needed to match MATLAB intermediate.
        amplitude_spectrum /= N**2 * num_blocks**2

        amplitude_spectrum = fft.fftshift(amplitude_spectrum)

        return Image(xp.asnumpy(amplitude_spectrum))

    def radial_average(self, amplitude_spectrum):
        """
        Computes radial/elliptical average of the power spectrum

        :param amplitude_spectrum: Power spectrum to average.
        :return: PSD and noise as 2-tuple of NumPy arrays.
        """
        noise = amplitude_spectrum  # pass through
        coefs_s = self.basis.evaluate_t(amplitude_spectrum).asnumpy().copy()

        mask_s = (self.basis.angular_indices == 1) | (self.basis.angular_indices == 2)
        coefs_s[:, mask_s] = 0

        psd = Coef(
            self.basis, coefs_s, pixel_size=amplitude_spectrum.pixel_size
        ).evaluate()

        return psd, noise

    def elliptical_average(self, amplitude_spectrum):
        """
        Computes elliptical average of the power spectrum

        :param amplitude_spectrum: Power spectrum to average.
        :return: PSD and noise as 2-tuple of NumPy arrays.
        """

        coefs_s = self.basis.evaluate_t(amplitude_spectrum).asnumpy().copy()
        coefs_n = coefs_s.copy()

        mask_s = self.basis.angular_indices == 1
        coefs_s[:, mask_s] = 0

        mask_n = (self.basis.angular_indices == 0) | (self.basis.angular_indices == 2)
        coefs_n[:, mask_n] = 0

        noise = Coef(
            self.basis, coefs_n, pixel_size=amplitude_spectrum.pixel_size
        ).evaluate()
        psd = Coef(
            self.basis, coefs_s, pixel_size=amplitude_spectrum.pixel_size
        ).evaluate()

        return psd, noise

    def background_subtract_1d(
        self, amplitude_spectrum, linprog_method="highs-ds", n_low_freq_cutoffs=14
    ):
        """
        Estimate and subtract the background from the power spectrum

        :param amplitude_spectrum: Estimated power spectrum
        :param linprog_method: Method passed to linear program solver (scipy.optimize.linprog).
        :param n_low_freq_cutoffs: Low frequency cutoffs (loop iterations).
        :return: 2-tuple of NumPy arrays (PSD after noise subtraction and estimated noise)
        """
        # compute radial average
        center = amplitude_spectrum.shape[-1] // 2

        if amplitude_spectrum.ndim == 3:
            if amplitude_spectrum.shape[0] != 1:
                raise ValueError(
                    f"Invalid dimension 0 for amplitude_spectrum {amplitude_spectrum.shape}"
                )
            amplitude_spectrum = amplitude_spectrum[0]
        elif amplitude_spectrum.ndim > 3:
            raise ValueError(
                f"Invalid ndimension for amplitude_spectrum {amplitude_spectrum.shape}"
            )

        amplitude_spectrum = amplitude_spectrum.asnumpy()[0, center, center:]
        amplitude_spectrum = amplitude_spectrum[
            0 : 3 * amplitude_spectrum.shape[-1] // 4
        ]

        final_signal = np.zeros(
            (n_low_freq_cutoffs - 1, amplitude_spectrum.shape[-1]), dtype=self.dtype
        )
        final_background = np.ones(
            (n_low_freq_cutoffs - 1, amplitude_spectrum.shape[-1]), dtype=self.dtype
        )

        for low_freq_cutoff in range(1, n_low_freq_cutoffs):
            signal = amplitude_spectrum[low_freq_cutoff:]
            signal = np.ravel(signal)
            N = amplitude_spectrum.shape[-1] - low_freq_cutoff

            f = np.concatenate((np.ones(N), -1 * np.ones(N)), axis=0)
            lb = np.concatenate([signal, np.full_like(signal, -np.inf)])
            ub = np.concatenate([signal, np.full_like(signal, np.inf)])
            bounds = np.column_stack([lb, ub])

            superposition_condition = np.concatenate(
                (-1 * np.eye(N), np.eye(N)), axis=1
            )

            monotone_condition = np.diag(np.full((N - 1), -1), -1) + np.diag(
                np.ones(N), 0
            )
            monotone_condition = monotone_condition[1:]

            convex_condition = (
                np.diag(np.full((N - 1), -1), -1)
                + np.diag(np.full(N, 2), 0)
                + np.diag(np.full((N - 1), -1), 1)
            )
            convex_condition = np.concatenate(
                (np.zeros((N, N)), convex_condition), axis=1
            )
            convex_condition = np.roll(convex_condition, -1, axis=0)
            convex_condition[N - 2 :] = 0

            positivity_condition = np.concatenate(
                (np.zeros((N, N)), -1 * np.eye(N)), axis=1
            )

            A = np.concatenate(
                (superposition_condition, convex_condition, positivity_condition),
                axis=0,
            )

            res = linprog(
                f,
                A_ub=A,
                b_ub=np.zeros(A.shape[0]),
                method=linprog_method,
                bounds=bounds,
            )

            if not res.success:
                raise RuntimeError("Linear program did not succeed. Halting")

            background = res.x[N:]

            bs_psd = signal - background

            final_signal[low_freq_cutoff - 1, low_freq_cutoff:] = bs_psd.T
            final_background[low_freq_cutoff - 1, low_freq_cutoff:] = background.T

        return final_signal, final_background

    def opt1d(
        self,
        amplitude_spectrum,
    ):
        """
        Find optimal defocus for the radially symmetric case (where no astigmatism is present)

        :param amplitude_spectrum: Estimated power specrtum.
        :return: 2-tuple of NumPy arrays (Estimated average of defocus and low_freq_cutoff)
        """

        signal = xp.asarray(amplitude_spectrum.T)
        signal = xp.maximum(0.0, signal)
        signal = xp.sqrt(signal)
        signal = signal[: 3 * signal.shape[0] // 4]
        N, M = signal.shape

        # Compute the (radial) CTF filter stack in bulk
        defoci_angstrom = xp.arange(
            self.min_defocus, self.max_defocus, self.step_defocus, dtype=int
        )
        ctf_ims = xp.abs(
            CTFFilter.ctf_formula(
                xp.asarray(self.radial_omega),
                pixel_size=self.pixel_size,
                voltage=self.voltage,
                defocus_u_a=defoci_angstrom,
                defocus_v_a=defoci_angstrom,
                defocus_ang=0,
                Cs=self.cs,
                alpha=self.amplitude_contrast,
                B=0,
            )
        )

        # Compute `signal` statistics, independent of defocus
        LT = xp.tril(signal, k=-1)  # keep lower triangle
        sx = LT.sum(axis=0)  # sum of row values
        sxx = (LT * LT).sum(axis=0)  # sum of squared row values
        nrows = (N - 1 - xp.arange(M)).astype(
            self.dtype
        )  # rows below diagonal for each column
        Sxx = sxx - sx**2 / nrows  # centered

        # CTF statistics
        ctf_ims = xp.asarray(ctf_ims, dtype=self.dtype)[:, :N]
        sy = tail_sum(ctf_ims, M)
        syy = tail_sum(ctf_ims * ctf_ims, M)
        Syy = syy - sy**2 / nrows

        # Compute the cross corr using one GEMM
        c = ctf_ims @ LT
        c -= sy * sx[None, :] / nrows
        c = c / xp.sqrt(Sxx[None, :] * Syy)

        max_c = xp.argmax(c).item()
        avg_defocus, low_freq_cutoff_col = np.unravel_index(max_c, c.shape)[:2]
        avg_defocus += self.min_defocus / 10  # (nm)

        return avg_defocus, low_freq_cutoff_col

    def background_subtract_2d(self, signal, background_p1, max_col):
        """
        Subtract background from estimated power spectrum

        :param signal: Estimated power spectrum
        :param background_p1: 1-D background estimation
        :param max_col: Internal variable, returned as the second parameter from opt1d.
        :return: 2-tuple of NumPy arrays (Estimated PSD without noise and estimated noise).
        """

        signal = signal.asnumpy()

        background = np.zeros(signal.shape, dtype=self.dtype)
        idx = np.ceil(self.R).astype(int) - 1
        rings = (self.R > max_col + 1) & (self.R <= background_p1.shape[1])
        background[:, rings] = background_p1[max_col, idx[rings]]
        mask = self.R <= max_col + 1
        background[:, mask] = signal[:, mask]

        signal = signal - background
        signal = np.maximum(0, signal)

        return Image(signal, pixel_size=self.pixel_size), Image(
            background, pixel_size=self.pixel_size
        )

    def pca(self, signal):
        """
        Compute PCA to yield ratio for estimating  defocus values.

        :param signal: Estimated power spectrum.
        :return: ratio.
        """

        signal = signal.asnumpy()[0]  # Convert from Image to square array.

        N = signal.shape[0]
        center = N // 2

        signal = signal - np.min(signal)

        rad_sq_min = N * self.pixel_size / self.g_min
        rad_sq_max = N * self.pixel_size / self.g_max

        min_limit = self.r_ctf[center, (center + np.floor(rad_sq_min)).astype(int)]
        signal[self.r_ctf < min_limit] = 0

        max_limit = self.r_ctf[center, (center + np.ceil(rad_sq_max)).astype(int)]
        signal[self.r_ctf > max_limit] = 0

        moment_02 = self.Y**2 * signal
        moment_02 = np.sum(moment_02, axis=(0, 1))

        moment_11 = self.Y * self.X * signal
        moment_11 = np.sum(moment_11, axis=(0, 1))

        moment_20 = self.X**2 * signal
        moment_20 = np.sum(moment_20, axis=(0, 1))

        moment_mat = np.zeros((2, 2))
        moment_mat[0, 0] = moment_20
        moment_mat[1, 1] = moment_02
        moment_mat[0, 1] = moment_11
        moment_mat[1, 0] = moment_11

        moment_evals = npla.eigvalsh(moment_mat)
        ratio = moment_evals[0] / moment_evals[1]

        return ratio

    def _gd(
        self,
        signal,
        df1,
        df2,
        angle_ast,
        stop_cond=None,
        max_iterations=400,
    ):
        """
        Runs inner gradient ascent to optimize defocus parameters.

        :param signal: Estimated power spectrum
        :param df1: Defocus value in the direction perpendicular to df2.
        :param df2: Defocus value in the direction perpendicular to df1.
        :param angle_ast: Angle between df1 and the x-axis, Radians.
        :param stop_cond: Optionally select value to stop iterations early.
            Default `None` is 1e-10 for float32 `dtype` and 1e20 otherwise.
        :param max_iterations: Maximum number of inner gd steps.
            Default 400.
        :return: Optimal defocus parameters
        """

        if stop_cond is None:
            stop_cond = 1e-10 if self.dtype == np.float32 else 1e-20
        stop_cond = float(stop_cond)

        # step sizes for (x, y, z)
        alpha = np.array([1e5, 1e4, 1e4])

        # initialization
        x = df1 + df2
        y = (df1 - df2) * np.cos(2 * angle_ast)
        z = (df1 - df2) * np.sin(2 * angle_ast)

        signal = signal.asnumpy()[0].T
        N = signal.shape[1]
        center = N // 2

        rad_sq_min = N * self.pixel_size / self.g_min
        rad_sq_max = N * self.pixel_size / self.g_max

        max_val = self.r_ctf[center, int(center - 1 + np.floor(rad_sq_max))]
        min_val = self.r_ctf[center, int(center - 1 + np.ceil(rad_sq_min))]
        mask = (self.r_ctf <= max_val) & (self.r_ctf > min_val)

        # everything below depends only on the ring of pixels
        r2 = self.r_ctf[mask] ** 2
        a = (np.pi * self.lmbd / 2) * r2
        b = (np.pi * self.lmbd**3 * self.cs * 1e6 / 2) * r2**2 - self.amplitude_contrast

        sig = np.ascontiguousarray(signal[mask])  # repack for compute below
        theta2 = 2 * self.theta[mask]

        # Referencing old implementation,
        # psi = x*a + y*a*cos2t + z*a*sin2t - b  ==  v @ A - b
        A = np.ascontiguousarray(np.stack([a, a * np.cos(theta2), a * np.sin(theta2)]))

        sum_A = sig @ sig
        v = np.array([x, y, z], dtype=A.dtype)
        W = np.empty((sig.size, 2), dtype=A.dtype)

        def evaluate(v, need_grad=True):
            psi = v @ A
            psi -= b
            s = np.sin(psi)
            c1 = np.abs(s) @ sig
            c2 = np.sqrt(sum_A * (s @ s))
            grad = None
            if need_grad:
                cs = np.cos(psi)
                W[:, 0] = np.sign(s) * cs
                W[:, 0] *= sig  # sign(s) * cos * signal
                W[:, 1] = s * cs
                W[:, 1] *= 2  # d(sin^2)/d(psi)
                G = A @ W  # compute all gradient sums
                dc1 = G[:, 0]
                dc2 = sum_A * G[:, 1] / (2 * c2)  # derivative_sqrt = 1 / (2 * c2)
                grad = (dc1 * c2 - dc2 * c1) / c2**2
            return c1, c2, grad

        dx = dy = dz = 1.0
        iter_no = 1

        while max(dx, dy, dz) > stop_cond:
            _, _, grad = evaluate(v)
            dx, dy, dz = grad
            v += alpha * grad

            if iter_no < 2:
                stop_cond = min(dx, dy, dz) / 1000
            if iter_no > max_iterations:
                stop_cond = max(dx, dy, dz) + 1
            iter_no += 1

        x, y, z = v
        df1 = (x + np.hypot(y, z)) / 2
        df2 = (x - np.hypot(y, z)) / 2
        angle_ast = np.arctan2(z, y) / 2

        c1, c2, _ = evaluate(v, need_grad=False)
        p = c1 / c2

        return df1, df2, angle_ast, p, iter_no

    def gd(
        self, signal, initial_df1, initial_df2, gradient_descent_iterations=6, **kwargs
    ):
        """
        Calls gradient descent loop multiple times and
        returns best optimum found.

        This implementation was adapted and optimized from the original MATLAB code.

        :param signal: Signal PSD array.
        :param initial_df1: Initial major defocus (nm)
        :param initial_df2: Initial minor defocus (nm)
        :param gradient_descent_iterations: Optional, number of Gradient Descent attempts.
            Defaults to 6.
        :return: defocus_u_nm, defocus_v_nm, defocus_ang_radian
        """

        # This array holds results from each of the `gradient_descent_iterations`
        cc_array = np.zeros((gradient_descent_iterations, 4))
        for a, angle in enumerate(
            np.linspace(0, np.pi, gradient_descent_iterations, endpoint=False)
        ):
            df1, df2, angle_ast, p, iters = self._gd(
                signal,
                initial_df1,
                initial_df2,
                angle,
                **kwargs,
            )

            cc_array[a, 0] = df1
            cc_array[a, 1] = df2
            cc_array[a, 2] = angle_ast  # Radians
            cc_array[a, 3] = p
            logger.debug(
                f"Gradient Descent {a+1} of {gradient_descent_iterations}:\tp={p}\titers={iters}"
            )

        # Select the best result index
        ml = np.argmax(cc_array[:, 3], -1)
        return cc_array[ml, 0:3]

    def bfgs(
        self,
        signal,
        initial_df1,
        initial_df2,
        gtol=1e-10,
        maxiter=200,
    ):
        """
        Calls quasi-newton method BFGS and returns found optimized values.

        :param signal: Signal PSD array.
        :param initial_df1: Initial major defocus (nm)
        :param initial_df2: Initial minor defocus (nm)
        :param gtol: Optionally select value to stop iterations.
            Default is 1e-10.
        :param maxiter: Optionally provide a max number of iterations for BFGS.
            Defaults to 200.
        :return: defocus_u_nm, defocus_v_nm, defocus_ang_radian
        """

        # Initial v = [x, y, z]
        angle_ast = np.deg2rad(-75)  # legacy value, shouldn't matter in bfgs
        v0 = np.array(
            [
                initial_df1 + initial_df2,
                (initial_df1 - initial_df2) * np.cos(2 * angle_ast),
                (initial_df1 - initial_df2) * np.sin(2 * angle_ast),
            ]
        )

        signal = signal.asnumpy()[0].T
        N = signal.shape[1]
        center = N // 2
        rad_sq_min = N * self.pixel_size / self.g_min
        rad_sq_max = N * self.pixel_size / self.g_max

        max_val = self.r_ctf[center, int(center - 1 + np.floor(rad_sq_max))]
        min_val = self.r_ctf[center, int(center - 1 + np.ceil(rad_sq_min))]
        mask = (self.r_ctf <= max_val) & (self.r_ctf > min_val)

        # Everything that does not depend on (x, y, z)
        # psi = x*a + y*a*cos(2t) + z*a*sin(2t) - b  ==  v @ A - b
        r2 = self.r_ctf[mask] ** 2
        a = (np.pi * self.lmbd / 2) * r2
        b = (np.pi * self.lmbd**3 * self.cs * 1e6 / 2) * r2**2 - self.amplitude_contrast
        sig = np.ascontiguousarray(signal[mask])
        theta2 = 2 * self.theta[mask]
        A = np.ascontiguousarray(np.stack([a, a * np.cos(theta2), a * np.sin(theta2)]))
        sum_A = sig @ sig

        # Variable scaling
        scale = np.array([1e2, 1e1, 1e1])

        def neg_p_and_grad(u):
            v = u * scale
            psi = v @ A
            psi -= b
            s = np.sin(psi)
            c = np.cos(psi)

            c1 = np.abs(s) @ sig
            c2 = np.sqrt(sum_A * (s @ s))

            dc1 = A @ (np.sign(s) * c * sig)
            dc2 = (sum_A / (2 * c2)) * (A @ (2 * s * c))
            grad = (dc1 * c2 - dc2 * c1) / c2**2

            return -c1 / c2, -grad * scale

        res = minimize(
            neg_p_and_grad,
            v0 / scale,
            jac=True,
            method="BFGS",
            options={"gtol": gtol, "maxiter": maxiter},
        )

        x, y, z = res.x * scale
        df1 = (x + np.hypot(y, z)) / 2  # nm
        df2 = (x - np.hypot(y, z)) / 2  # nm
        angle_ast = np.arctan2(z, y) / 2  # Radians

        logger.debug(f"BFGS :\tp={-res.fun}\titers={res.nit}")

        return df1, df2, angle_ast

    def maximize_correlation(self, signal, initial_df1, initial_df2):
        """
        Selects which solver to use for optimizing CTF estimates.

        :param signal: Signal PSD array
        :param initial_df1: Initial major defocus (nm)
        :param initial_df2: Initial minor defocus (nm)
        :return: defocus_u_nm, defocus_v_nm, defocus_ang_rad
        """
        if self.solver_method == "gd":
            return self.gd(signal, initial_df1, initial_df2, **self.solver_opts)
        elif self.solver_method == "bfgs":
            return self.bfgs(signal, initial_df1, initial_df2, **self.solver_opts)

    def write_star(self, star_filepath, params_dict):
        """
        Writes CTF parameters to starfile for a single micrograph.
        """
        data_block = {}
        data_block["_rlnMicrographName"] = star_filepath
        data_block["_rlnDefocusU"] = params_dict["defocus_u"]  # Should already be in A
        data_block["_rlnDefocusV"] = params_dict["defocus_v"]  # Should already be in A
        data_block["_rlnDefocusAngle"] = np.rad2deg(params_dict["defocus_ang"])
        data_block["_rlnSphericalAberration"] = params_dict["cs"]
        data_block["_rlnAmplitudeContrast"] = params_dict["amplitude_contrast"]
        data_block["_rlnVoltage"] = params_dict["voltage"]
        data_block["_rlnMicrographPixelSize"] = params_dict["pixel_size"]
        blocks = OrderedDict()
        blocks["root"] = data_block
        star = StarFile(blocks=blocks)
        star.write(star_filepath)

    def _plot_1d_psd(self, signal_1d, background_1d):
        """
        Helper function plots 1D PSD.

        :param signal_1d: Background subtracted radial PSD
        :param background_1d: Background radial PSD.
        """
        center = self.r_ctf.shape[0] // 2
        x = self.r_ctf[center, center : center + signal_1d.shape[1]]
        plt.plot(x, signal_1d[1] + background_1d[1])
        plt.title("1-d PSD")
        plt.xlabel(r"$\vert \mathbf{k} \vert^2 \, \, [\mathrm{nm}^{-1}]$")
        plt.ylabel(r"$S_y \left( \mathbf{k} \right)$")
        plt.show()

    def _plot_1d_bg_sub_psd(self, signal_1d):
        """
        Helper function plots 1D PSD.

        :param signal_1d: Background subtracted radial PSD
        """
        center = self.r_ctf.shape[0] // 2
        x = self.r_ctf[center, center : center + signal_1d.shape[1]]
        plt.plot(x, signal_1d[1])
        plt.title("Background subtracted 1-d PSD")
        plt.xlabel(r"$\vert \mathbf{k} \vert^2 \, \, [\mathrm{nm}^{-1}]$")
        plt.ylabel(r"$S_y \left( \mathbf{k} \right) - S_e \left( \mathbf{k} \right)$")
        plt.show()

    def _plot_2d_psd(self, psd):
        """
        Helper function plots 2D PSD.

        :param psd: 2D PSD array
        """
        plt.imshow(psd)
        plt.title("2-d PSD")
        plt.show()

    def _plot_2d_bg_sub_psd(self, bg_sub_psd):
        """
        Helper function plots 2D background subtracted PSD.

        :param psd: 2D background subtracted PSD
        """
        plt.imshow(bg_sub_psd)
        plt.title("Background subtracted 2-d PSD")
        plt.show()

    def estimate(self, micrograph):
        """
        Estimate the CTF parameters of `micrograph`.

        This is the main method implementing the algorithm at a high level.

        :param micrograph: 2D Array of micrograph data.
        :return: CTF params, estimated background PSD, estimated signal PSD
        """
        # TODO Ask Ayelet about legacy version intent and rm if not needed.
        micrograph_blocks = self.preprocess_micrograph(micrograph)
        # micrograph_blocks = self.legacy_preprocess_micrograph(micrograph)

        signal_observed = self.estimate_psd(micrograph_blocks)

        amplitude_spectrum, _ = self.radial_average(signal_observed)

        signal_1d, background_1d = self.background_subtract_1d(
            amplitude_spectrum, linprog_method=self.linprog_method
        )

        if self.diagnostic_plots:
            self._plot_1d_psd(signal_1d, background_1d)
            self._plot_1d_bg_sub_psd(signal_1d)

        avg_defocus, low_freq_skip_col = self.opt1d(signal_1d)

        signal, background_2d = self.background_subtract_2d(
            signal_observed, background_1d, low_freq_skip_col
        )

        if self.diagnostic_plots:
            self._plot_2d_psd(amplitude_spectrum.asnumpy()[0])
            self._plot_2d_bg_sub_psd(signal.asnumpy()[0])

        ratio = self.pca(signal_observed)

        signal, additional_background = self.elliptical_average(signal.sqrt())

        background_2d = background_2d + additional_background

        initial_df1 = (avg_defocus * 2) / (1 + ratio)
        initial_df2 = (avg_defocus * 2) - initial_df1

        defocus_u, defocus_v, defocus_ang = self.maximize_correlation(
            signal, initial_df1, initial_df2
        )

        params = {
            "defocus_u": defocus_u * 10,  # Convert from nm to A
            "defocus_v": defocus_v * 10,  # Convert from nm to A
            "defocus_ang": defocus_ang,  # radians
            "cs": self.cs,
            "voltage": self.voltage,
            "pixel_size": self.pixel_size,
            "amplitude_contrast": self.amplitude_contrast,
        }

        return params, background_2d, signal

    def _create_ctf_image(self, signal, df1, df2, dfang):
        """
        Create a diagnostic array comparing estimated CTF and signal.

        :param signal: PSD of signal
        :param df1: Major defocus (Angstrom)
        :param df2: Minor defocus (Angstrom)
        :param dfang: Defocus angle (radians)
        :return: 2d array containing image data.
        """

        ctf_im = CTFFilter.ctf_formula(
            self.omega,
            pixel_size=self.pixel_size,
            voltage=self.voltage,
            defocus_u_a=df1,
            defocus_v_a=df2,
            defocus_ang=dfang,
            Cs=self.cs,
            alpha=self.amplitude_contrast,
            B=0,
        ).reshape(self.psd_size, self.psd_size)

        # Overwrite half the `ctf_im` with `signal`
        center = self.psd_size // 2
        ctf_im[center:, :] = signal.asnumpy()[0, center:, :]

        return ctf_im


def estimate_ctf(
    data_folder,
    pixel_size=1.0,
    cs=2.0,
    amplitude_contrast=0.07,
    voltage=300,
    num_tapers=4,
    psd_size=1024,
    g_min=30,
    g_max=5,
    min_defocus=5000,
    max_defocus=100000,
    step_defocus=10,
    basis=None,
    output_dir="results",
    dtype=np.float32,
    save_ctf_images=False,
    save_noise_images=False,
    rglob=r"*.mrc",
    solver_method="gd",
    solver_opts=None,
    linprog_method="highs-ds",
):
    """
    Given paramaters estimates CTF from experimental data in `data_folder
    and returns dictionary of estimated CTF parameters.

    Saves STAR files to `output_dir`, which can be loaded into ASPIRE etc.

    Optionally saves diagnostic CTF and noise image (mrc) files.

    :param pixel_size: Size of the pixel in \u212b (angstrom).
    :param cs: Spherical aberration in mm.
    :param amplitude_contrast: Amplitude contrast.
    :param voltage: Voltage of electron microscope.
    :param num_tapers: Optional, number of tapers to apply in PSD estimation.
        Defaults to 4.
    :param psd_size: Block size (in pixels) for PSD estimation.
    :param g_min: Inverse of minimun resolution for PSD. Default 30.
    :param g_max: Inverse of maximum resolution for PSD. Default 5.
    :param min_defocus: Start of defocus loop scan (Angstrom). Default 5000.
    :param max_defocus: End of defocus loop scan (Angstrom). Default 100000.
    :param step_defocus: Step size of defocus loop scan (Angstrom). Default 10.
    :param basis: Optional user provided SteerableBasis2D.  Defaults to `FFBBasis2D`.
    :param linprog_method: Method passed to linear program solver (scipy.optimize.linprog).
        Defaults to 'highs-ds'.
    :param diagnostic_plots: Optionally generate diagnostic_plots.
        Note, these are blocking plot calls.
        Defaults to False.
    :param rglob: File search pattern.
        By default will search for **/*.mrc files.
        Using "*.mrc[s]" will additonally include mrcs files, etc.
    :param save_ctf_images: Save diagnostic ctf images.
    :param save_noise_images: Save diagnostic noise images.
    :param solver_method: Solver used to optimize defocus params.  `gd` or `bfgs`.
        Default uses single call to `bfgs`.
        Original implementation used repeated gradient descent and is provided for reproducibility.
        Each solver may have unique configuration options, optionally passed via a dictionary.
    :param solver_opts: Optional dictionary of solver options.
        Populated with `CtfEstimator.default_solver_opts[solver]` by default.
    :param linprog_method: Method passed to linear program solver (scipy.optimize.linprog).
        Defaults to 'highs-ds'.
    """

    # Checks ------------------------------------------------------------------
    dtype = np.dtype(dtype)
    assert dtype in (np.float32, np.float64)

    # File inputs -------------------------------------------------------------
    data_folder = Path(data_folder)
    # Handles recursive directory searching for both `rglob` patten
    file_paths = list(data_folder.rglob(rglob))
    if len(file_paths) == 0:
        raise RuntimeError(f"No mrc or mrcs files found in {data_folder}.")
    logger.info(f"Found {len(file_paths)} files in '{data_folder}'.")

    # File outputs -----------------------------------------------------------
    # Ensure we can write outputs before we get too far
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup a CTFEstimator -----------------------------------------------------
    ctf_est = CtfEstimator(
        pixel_size,
        cs,
        amplitude_contrast,
        voltage,
        psd_size,
        num_tapers=num_tapers,
        g_min=g_min,
        g_max=g_max,
        min_defocus=min_defocus,
        max_defocus=max_defocus,
        step_defocus=step_defocus,
        basis=basis,
        diagnostic_plots=False,
        dtype=dtype,
    )

    # Main Loop ---------------------------------------------------------------
    results = {}
    for input_file in tqdm(file_paths, desc="Processing files"):

        # Handle path stuff first, in case it fails
        relative_input_path = input_file.relative_to(
            data_folder
        )  # File path relative to data_folder
        directory_stub = relative_input_path.parent  # directory relative to data_folder
        output_file_dir = output_dir / directory_stub  # Join onto output_dir
        output_file_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure output_file_dir exists
        output_file_prefix = output_file_dir / input_file.stem

        with mrcfile.open(input_file, mode="r", permissive=True) as mrc:
            micrograph = mrc.data

        micrograph = micrograph.astype(dtype, copy=False)

        params, background_2d, signal = ctf_est.estimate(micrograph)

        # Store files' params into results dictionary
        results[str(relative_input_path)] = params

        # Write each micrograph's ctf parameters to an individual STAR file
        star_filepath = output_file_prefix.with_suffix(".star")
        ctf_est.write_star(star_filepath, params)

        # Save the noise
        if save_noise_images:
            noise_filepath = Path(f"{output_file_prefix}_noise.mrc")
            background_2d.save(noise_filepath, overwrite=True)

        if save_ctf_images:
            ctf_im = ctf_est._create_ctf_image(
                signal,
                params["defocus_u"],
                params["defocus_v"],
                params["defocus_ang"],
            )

            ctf_filepath = output_file_prefix.with_suffix(".ctf")
            with mrcfile.new(ctf_filepath, overwrite=True) as mrc:
                mrc.set_data(ctf_im.astype(np.float32))  # MRC only supports singles
                mrc.voxel_size = ctf_est.pixel_size

    return results


# Helper function for sum(abs^2(x))
# CuPy
if hasattr(xp, "ReductionKernel"):
    _abs2_sum0_kernel = xp.ReductionKernel(
        "T x",
        "Y y",
        "x.real() * x.real() + x.imag() * x.imag()",
        "a + b",
        "y = a",
        "0",
        name="abs2_sum0",
    )

    def _abs2_sum0(x, out):
        _abs2_sum0_kernel(x, axis=0, out=out)

else:
    # numpy expression fallback
    def _abs2_sum0(x, out):
        out[:] = (x.real**2 + x.imag**2).sum(axis=0)


# Helper function
def tail_sum(x, M):
    """sum over i > m  (exclusive reverse cumsum)"""
    cs = xp.cumsum(x, axis=1)
    return cs[:, -1:] - cs[:, :M]
