"""
Contains code supporting CTF parameter estimation.
Generally, this is a Python version of ASPIRE-CTF from MATLAB.

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

Integrated into ASPIRE-Python by Garrett Wright Feb 2021.
"""

import logging
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from numpy import linalg as npla
from scipy.optimize import linprog
from scipy.signal.windows import dpss

from aspire.basis import Coef, FFBBasis2D
from aspire.image import Image
from aspire.numeric import fft, xp
from aspire.operators import CTFFilter
from aspire.storage import StarFile
from aspire.utils import abs2, complex_type, grid_2d, tqdm, voltage_to_wavelength

logger = logging.getLogger(__name__)


class CtfEstimator:
    """
    CtfEstimator Class ...
    """

    def __init__(
        self,
        pixel_size,
        cs,
        amplitude_contrast,
        voltage,
        psd_size,
        num_tapers,
        gradient_descent_iterations=6,
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
        :param gradient_descent_iterations: Optional, number of Gradient Descent attempts.
            Defaults to 6.
        :param linprog_method: Method passed to linear program solver (scipy.optimize.linprog).
            Defaults to 'highs-ds'.
        :param diagnostic_plots: Optionally generate diagnostic_plots.
            Note, these are blocking plot calls.
            Defaults to False.
        :returns: CtfEstimator instance.
        """

        self.pixel_size = float(pixel_size)
        self.cs = float(cs)
        self.amplitude_contrast = float(amplitude_contrast)
        self.voltage = float(voltage)
        self.psd_size = int(psd_size)
        self.num_tapers = int(num_tapers)
        self.lmbd = voltage_to_wavelength(voltage) / 10.0  # Convert angstrom to nm
        self.gd_iters = int(gradient_descent_iterations)
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

        # TODO, try to extend to any steerable basis
        self.basis = FFBBasis2D((psd_size, psd_size), 2, dtype=dtype)

        # Only need to compute tapers once
        # Relating to MATLAB code, NW:= R * N / 2 = (2*L)/N * N / 2 = L
        tapers_1d = self.tapers(psd_size, num_tapers, num_tapers)
        self.tapers_1d = xp.asarray(tapers_1d, dtype=complex_type(self.dtype))

    def micrograph_to_blocks(self, micrograph, block_size):
        """
        Preprocess micrograph into blocks using block_size.

        :param micrograph: Micrograph as NumPy array. #NOTE looks like F order
        :param blocksize: Size of the square blocks to partition micrograph.
        :return: NumPy array of blocks extracted from the micrograph.
        """

        # verify block_size is even
        assert block_size % 2 == 0

        if micrograph.ndim == 3:
            assert (
                micrograph.shape[0] == 1
            ), f"micrograph should be 2D or stack of 1 2D image: {micrograph.shape}"
            micrograph = micrograph[0]

        size_x = micrograph.shape[-1]
        size_y = micrograph.shape[-2]

        step_size = block_size // 2
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

    def preprocess_micrograph(self, micrograph, block_size):
        """
        Preprocess micrograph into normalized blocks using block_size.

        :param micrograph: Micrograph as NumPy array. #NOTE looks like F order
        :param blocksize: Size of the square blocks to partition micrograph.
        :return: NumPy array of normalized blocks extracted from the micrograph.
        """

        return self.normalize_blocks(self.micrograph_to_blocks(micrograph, block_size))

    def legacy_preprocess_micrograph(self, micrograph, block_size):
        """
        Returns half-overlapping blocks extracted from the micrograph.
        Equivalent to MATLAB's ctf_extract_blocks.

        Note, internally transposes micrograph to match MATLAB computations.
        Returns (num_blocks, block_size, block_size)
        """
        h, w = micrograph.shape

        if min(h, w) < block_size:
            raise ValueError(
                f"Blocks of size {block_size} x {block_size} cannot be extracted from image."
            )

        block_size = block_size - (block_size % 2)  # force even
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
        # TODO, I don't think the order will matter here later,
        # but keeping now for diagnostics.
        blocks = (
            cropped_micrograph.reshape(n_bh, half, n_bw, half)
            .transpose(1, 3, 0, 2)
            .reshape(half * half, n_total, order="F")
        )

        # Reshape to (half, half, n_total) so each slice is one half-block
        blocks = blocks.reshape(half, half, n_total, order="F")

        # TODO, compare which method is faster later, if we really need these...
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

        # Transpose output to C order (n,block_size, block_size)
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

        num_1d_tapers = self.tapers_1d.shape[0]

        blocks_mt = xp.zeros(blocks[0, :, :].shape, dtype=self.dtype)

        blocks_tapered = xp.zeros(blocks[0, :, :].shape, dtype=complex_type(self.dtype))

        taper_2d = xp.zeros(
            (blocks.shape[1], blocks.shape[2]), dtype=complex_type(self.dtype)
        )

        # Ensure blocks are on GPU
        blocks = xp.asarray(blocks)

        for ax1 in range(num_1d_tapers):
            for ax2 in range(num_1d_tapers):
                xp.matmul(
                    self.tapers_1d[ax1, :, np.newaxis],
                    self.tapers_1d[ax2, :, np.newaxis].T,
                    out=taper_2d,
                )

                for m in range(blocks.shape[0]):
                    xp.multiply(blocks[m, :, :], taper_2d, out=blocks_tapered)
                    blocks_mt_post_fft = fft.fftn(blocks_tapered, axes=(-2, -1))
                    blocks_mt += abs2(blocks_mt_post_fft)
                    # blocks_mt += np.abs(blocks_mt_post_fft)**2

        blocks_mt /= blocks.shape[1] ** 2
        blocks_mt /= blocks.shape[0] ** 2

        amplitude_spectrum = fft.fftshift(blocks_mt)

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
        min_defocus=500,
        max_defocus=10000,
    ):
        """
        Find optimal defocus for the radially symmetric case (where no astigmatism is present)

        :param amplitude_spectrum: Estimated power specrtum.
        :param min_defocus: Start of defocus loop scan (xxx units).
        :param max_defocus: End of defocus loop scan (xxx units).
        :return: 2-tuple of NumPy arrays (Estimated average of defocus and low_freq_cutoff)
        """

        signal = amplitude_spectrum.T
        signal = np.maximum(0.0, signal)
        signal = np.sqrt(signal)
        signal = signal[: 3 * signal.shape[0] // 4]

        # Ask Ay about original range units, I read as nm, but that is 5000-100000A ...? xxx
        c = np.zeros((max_defocus - min_defocus, signal.shape[1]), dtype=self.dtype)

        for f in range(min_defocus, max_defocus):

            ctf_im = np.abs(
                CTFFilter.ctf_formula(
                    self.radial_omega,
                    pixel_size=self.pixel_size,
                    voltage=self.voltage,
                    defocus_u_a=10 * f,
                    defocus_v_a=10 * f,
                    defocus_ang=0,
                    Cs=self.cs,
                    alpha=self.amplitude_contrast,
                    B=0,
                )[0]
            )
            # Shouldn't normally be needed within class calls,
            #    but if user running manually clip to signal len.
            ctf_im = ctf_im[: signal.shape[0]]

            ctf_im = np.reshape(ctf_im, (ctf_im.shape[0], 1))
            ctf_im = np.tile(ctf_im, (1, signal.shape[1]))

            for m in range(0, signal.shape[1]):
                signal[:, m] = signal[:, m] - np.mean(signal[m + 1 :, m], axis=0)
                ctf_im[:, m] = ctf_im[:, m] - np.mean(ctf_im[m + 1 :, m], axis=0)
                ctf_im[: m + 1, m] = 0
                signal[: m + 1, m] = 0

            Sx = np.sqrt(np.sum(ctf_im**2, axis=0))
            Sy = np.sqrt(np.sum(signal**2, axis=0))
            c[f - min_defocus, :] = np.sum(ctf_im * signal, axis=0) / (Sx * Sy)

        avg_defocus, low_freq_cutoff = np.unravel_index(np.argmax(c), c.shape)[:2]
        avg_defocus += min_defocus

        return avg_defocus, low_freq_cutoff

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
        rings = (self.R > max_col + 2) & (self.R <= background_p1.shape[1])
        background[:, rings] = background_p1[max_col, idx[rings]]
        mask = self.R <= max_col + 2
        background[:, mask] = signal[:, mask]

        signal = signal - background
        signal = np.maximum(0, signal)

        return Image(signal, pixel_size=self.pixel_size), Image(
            background, pixel_size=self.pixel_size
        )

    def pca(self, signal, g_min, g_max):
        """

        :param signal: Estimated power spectrum.
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :return: ratio.
        """

        signal = signal.asnumpy()[0]  # Convert from Image to square array.

        N = signal.shape[0]
        center = N // 2

        signal = signal - np.min(signal)

        rad_sq_min = N * self.pixel_size / g_min
        rad_sq_max = N * self.pixel_size / g_max

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

    def gd(
        self,
        signal,
        df1,
        df2,
        angle_ast,
        g_min,
        g_max,
    ):
        """
        Runs gradient ascent to optimize defocus parameters

        :param signal: Estimated power spectrum
        :param df1: Defocus value in the direction perpendicular to df2.
        :param df2: Defocus value in the direction perpendicular to df1.
        :param angle_ast: Angle between df1 and the x-axis, Radians.
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :return: Optimal defocus parameters
        """

        # step size
        alpha1 = 1e5
        alpha2 = 1e4

        # initialization
        x = df1 + df2
        y = (df1 - df2) * np.cos(2 * angle_ast)
        z = (df1 - df2) * np.sin(2 * angle_ast)

        a = np.pi * self.lmbd * self.r_ctf**2 / 2
        # Ask Ay, Does this amplitude_contrast need to be the arctan'd one?
        # amplitude_contrast = np.arctan(
        #     self.amplitude_contrast / np.sqrt(1 - self.amplitude_contrast**2)
        # )
        b = np.pi * self.lmbd**3 * self.cs * 1e6 * self.r_ctf**4 / 2 - np.full(
            shape=self.r_ctf.shape, fill_value=self.amplitude_contrast, dtype=self.dtype
        )

        signal = signal.asnumpy()[0].T
        N = signal.shape[1]
        center = N // 2

        rad_sq_min = N * self.pixel_size / g_min
        rad_sq_max = N * self.pixel_size / g_max

        max_val = self.r_ctf[center, int(center - 1 + np.floor(rad_sq_max))]
        min_val = self.r_ctf[center, int(center - 1 + np.ceil(rad_sq_min))]

        mask = (self.r_ctf <= max_val) & (self.r_ctf > min_val)
        a = a[mask]
        b = b[mask]
        signal = signal[..., mask]
        # r = self.r_ctf[mask]
        theta = self.theta[mask]

        sum_A = np.sum(signal**2)

        dx = 1
        dy = 1
        dz = 1

        stop_cond = 1e-20
        iter_no = 1

        while np.maximum(np.maximum(dx, dy), dz) > stop_cond:
            inner_cosine = y * np.cos(2 * theta) + z * np.sin(2 * theta)
            psi = a * x + a * inner_cosine - b
            outer_sine = np.sin(psi)
            outer_cosine = np.cos(psi)

            sine_x_term = a
            sine_y_term = a * np.cos(2 * theta)
            sine_z_term = a * np.sin(2 * theta)

            c1 = np.sum(np.abs(outer_sine) * signal)
            c2 = np.sqrt(sum_A * np.sum(outer_sine**2))

            # gradients of numerator
            dx_c1 = np.sum(np.sign(outer_sine) * outer_cosine * a * signal)

            dy_c1 = np.sum(
                np.sign(outer_sine) * outer_cosine * a * np.cos(2 * theta) * signal
            )
            dz_c1 = np.sum(
                np.sign(outer_sine) * outer_cosine * a * np.sin(2 * theta) * signal
            )

            derivative_sqrt = 1 / (2 * np.sqrt(sum_A * np.sum(outer_sine**2)))

            derivative_sine2 = 2 * outer_sine * outer_cosine

            #  gradients of denomenator
            dx_c2 = derivative_sqrt * sum_A * np.sum(derivative_sine2 * sine_x_term)
            dy_c2 = derivative_sqrt * sum_A * np.sum(derivative_sine2 * sine_y_term)
            dz_c2 = derivative_sqrt * sum_A * np.sum(derivative_sine2 * sine_z_term)

            # gradients
            dx = (dx_c1 * c2 - dx_c2 * c1) / c2**2
            dy = (dy_c1 * c2 - dy_c2 * c1) / c2**2
            dz = (dz_c1 * c2 - dz_c2 * c1) / c2**2

            # update
            x = x + alpha1 * dx
            y = y + alpha2 * dy
            z = z + alpha2 * dz

            if iter_no < 2:
                stop_cond = np.minimum(np.minimum(dx, dy), dz) / 1000

            if iter_no > 400:
                stop_cond = np.maximum(np.maximum(dx, dy), dz) + 1

            iter_no = iter_no + 1

        df1 = (x + np.abs(y + z * 1j)) / 2
        df2 = (x - np.abs(y + z * 1j)) / 2
        angle_ast = np.angle(y + z * 1j) / 2  # Radians

        inner_cosine = y * np.cos(2 * theta) + z * np.sin(2 * theta)
        outer_sine = np.sin(a * x + a * inner_cosine - b)
        outer_cosine = np.cos(a * x + a * inner_cosine - b)

        sine_x_term = a
        sine_y_term = a * np.cos(2 * theta)
        sine_z_term = a * np.sin(2 * theta)

        c1 = np.sum(np.abs(outer_sine) * signal)
        c2 = np.sqrt(sum_A * np.sum(outer_sine**2))

        p = c1 / c2

        return df1, df2, angle_ast, p, iter_no

    # Note, This doesn't actually use anything from the class.
    # It is used in a solver loop of some sort, so it may not be correct
    # to just use what is avail in the obj.
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
        center = self.r_ctf.shape[0] // 2
        x = self.r_ctf[center, center : center + signal_1d.shape[1]]
        plt.plot(x, signal_1d[1] + background_1d[1])
        # xlabel('$\vert \mathbf{k} \vert^2 \, \, [\mathrm{nm}^{-1}]$', 'Interpreter','latex');
        # ylabel('$S_y \left( \mathbf{k} \right)$', 'Interpreter','latex');
        # title('1-d PSD')
        plt.show()

    def _plot_1d_bg_sub_psd(self, signal_1d):
        center = self.r_ctf.shape[0] // 2
        x = self.r_ctf[center, center : center + signal_1d.shape[1]]
        plt.plot(x, signal_1d[1])
        # title('Background subtracted 1-d PSD')
        # xlabel('$\vert \mathbf{k} \vert^2 \, \, [\mathrm{nm}^{-1}]$', 'Interpreter','latex');
        # ylabel('$S_y \left( \mathbf{k} \right) - S_e \left( \mathbf{k} \right)$', 'Interpreter','latex');
        plt.show()

    def _plot_2d_psd(self, psd):
        plt.imshow(psd)
        # title('2-d PSD')
        plt.show()

    def _plot_2d_bg_sub_psd(self, bg_sub_psd):
        plt.imshow(bg_sub_psd)
        # title('Background subtracted 2-d PSD')
        plt.show()

    def estimate(self, micrograph, g_min=30, g_max=5):
        # TODO compare now that remainder of code repros.
        # micrograph_blocks = ctf_object.preprocess_micrograph(micrograph, psd_size)
        micrograph_blocks = self.legacy_preprocess_micrograph(micrograph, self.psd_size)

        signal_observed = self.estimate_psd(micrograph_blocks)
        # np.save("signal_observed.npy", signal_observed)

        amplitude_spectrum, _ = self.radial_average(signal_observed)
        # np.save("amplitude_spectrum.npy", amplitude_spectrum)

        signal_1d, background_1d = self.background_subtract_1d(
            amplitude_spectrum, linprog_method=self.linprog_method
        )

        if self.diagnostic_plots:
            self._plot_1d_psd(signal_1d, background_1d)
            self._plot_1d_bg_sub_psd(signal_1d)

        avg_defocus, low_freq_skip = self.opt1d(signal_1d)

        # TODO check the low_freq_skip +-1, it is col ind or value
        signal, background_2d = self.background_subtract_2d(
            signal_observed, background_1d, low_freq_skip
        )

        if self.diagnostic_plots:
            self._plot_2d_psd(amplitude_spectrum.asnumpy()[0])
            self._plot_2d_bg_sub_psd(signal.asnumpy()[0])

        ratio = self.pca(signal_observed, g_min, g_max)

        signal, additional_background = self.elliptical_average(signal.sqrt())

        background_2d = background_2d + additional_background

        initial_df1 = (avg_defocus * 2) / (1 + ratio)
        initial_df2 = (avg_defocus * 2) - initial_df1

        angle = -5 / 12 * np.pi  # Radians (-75 degrees)

        # This array holds results from each of the `gd_iters`
        cc_array = np.zeros((self.gd_iters, 4))
        for a, angle in enumerate(np.linspace(0, np.pi, self.gd_iters)):
            df1, df2, angle_ast, p, iters = self.gd(
                signal,
                initial_df1,
                initial_df2,
                angle,
                g_min,
                g_max,
            )

            cc_array[a, 0] = df1
            cc_array[a, 1] = df2
            cc_array[a, 2] = angle_ast  # Radians
            cc_array[a, 3] = p
            logger.debug(
                f"Gradient Descent {a+1} of {self.gd_iters}:\tp={p}\titers={iters}"
            )

        # Select the best result index
        ml = np.argmax(cc_array[:, 3], -1)

        params = {
            "defocus_u": cc_array[ml, 0] * 10,  # Convert from nm to A
            "defocus_v": cc_array[ml, 1] * 10,  # Convert from nm to A
            "defocus_ang": cc_array[ml, 2],  # Radians
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
    gradient_descent_iterations=6,
    output_dir="results",
    dtype=np.float32,
    save_ctf_images=False,
    save_noise_images=False,
    rglob=r"*.mrc",
):
    """
    Given paramaters estimates CTF from experimental data
    and returns CTF as a mrc file.

    :param rglob: File search pattern.
        By default will search for **/*.mrc files.
        Using "*.mrc[s]" will additonally include mrcs files, etc.
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
        gradient_descent_iterations=gradient_descent_iterations,
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
        breakpoint()

        with mrcfile.open(input_file, mode="r", permissive=True) as mrc:
            micrograph = mrc.data

        micrograph = micrograph.astype(dtype, copy=False)

        params, background_2d, signal = ctf_est.estimate(
            micrograph, g_min=g_min, g_max=g_max
        )

        # Store files' params into results dictionary
        results[relative_input_path] = params

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
