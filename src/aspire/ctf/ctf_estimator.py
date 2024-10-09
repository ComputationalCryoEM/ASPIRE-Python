"""
Contains code supporting CTF parameter estimation.
Generally, this is a port of ASPIRE-CTF from MATLAB.

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
import os
from collections import OrderedDict

import mrcfile
import numpy as np
from numpy import linalg as npla
from scipy.optimize import linprog
from scipy.signal.windows import dpss

from aspire.basis import Coef, FFBBasis2D
from aspire.image import Image
from aspire.numeric import fft
from aspire.storage import StarFile
from aspire.utils import abs2, complex_type, grid_1d, grid_2d, voltage_to_wavelength

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
        dtype=np.float32,
    ):
        """
        Instantiate a CtfEstimator instance.

        :param pixel_size: Size of the pixel in \u212b (angstrom).
        :param cs: Spherical aberration in mm.
        :param amplitude_contrast: Amplitude contrast.
        :param voltage: Voltage of electron microscope.
        :param psd_size: Block size (in pixels) for PSD estimation.
        :param num_tapers: Number of tapers to apply in PSD estimation.
        :returns: CtfEstimator instance.
        """

        self.pixel_size = pixel_size
        self.cs = cs
        self.amplitude_contrast = amplitude_contrast
        self.voltage = voltage
        self.psd_size = psd_size
        self.num_tapers = num_tapers
        self.lmbd = voltage_to_wavelength(voltage) / 10.0  # Convert angstrom to nm
        self.dtype = np.dtype(dtype)

        grid = grid_2d(psd_size, normalized=True, indexing="yx", dtype=self.dtype)

        # Note range is -half to half.
        self.r_ctf = grid["r"] / 2 * (10 / pixel_size)  # units: inverse nm

        self.theta = grid["phi"]
        self.defocus1 = 0
        self.defocus2 = 0
        self.angle = 0  # Radians
        self.h = 0

    def set_df1(self, df):
        """
        Sets defocus.

        :param df: Defocus value in the direction perpendicular to df2.
        """

        self.defocus1 = df

    def set_df2(self, df):
        """
        Sets defocus.

        :param df: Defocus value in the direction perpendicular to df1.
        """

        self.defocus2 = df

    def set_angle(self, angle):
        """
        Sets angle.

        :param angle: Angle (in Radians) between df1 and the x-axis.
        """

        self.angle = angle

    def generate_ctf(self):
        """
        Generates internal representation of the Contrast Transfer Function using parameters from this instance.
        """

        astigmatism_angle = np.full(
            shape=self.theta.shape, fill_value=self.angle, dtype=self.dtype
        )

        defocus_sum = np.full(
            shape=self.theta.shape,
            fill_value=self.defocus1 + self.defocus2,
            dtype=self.dtype,
        )

        defocus = defocus_sum + (
            (self.defocus1 - self.defocus2)
            * np.cos(2 * (self.theta - astigmatism_angle))
        )
        defocus_factor = np.pi * self.lmbd * self.r_ctf * defocus / 2
        amplitude_contrast_term = self.amplitude_contrast / np.sqrt(
            1 - self.amplitude_contrast**2
        )

        chi = (
            defocus_factor
            - np.pi * self.lmbd**3 * self.cs * 1e6 * self.r_ctf**2 / 2
            + amplitude_contrast_term
        )
        h = -np.sin(chi)
        self.h = h

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

        # Note the original ASPIRE implementation is negated from original scipy...
        #  but at time of writing subsequent code was agnostic to sign.
        return dpss(M=N, NW=NW, Kmax=L, return_ratios=False).T

    def estimate_psd(self, blocks, tapers_1d):
        """
        Estimate the power spectrum of the micrograph using the multi-taper method

        :param blocks: 3-D NumPy array containing windows extracted from the micrograph in the preprocess function.
        :param tapers_1d: NumPy array of data tapers.
        :return: NumPy array of estimated power spectrum.
        """

        num_1d_tapers = tapers_1d.shape[-1]
        tapers_1d = tapers_1d.astype(complex_type(self.dtype), copy=False)

        blocks_mt = np.zeros(blocks[0, :, :].shape, dtype=self.dtype)

        blocks_tapered = np.zeros(blocks[0, :, :].shape, dtype=complex_type(self.dtype))

        taper_2d = np.zeros(
            (blocks.shape[1], blocks.shape[2]), dtype=complex_type(self.dtype)
        )

        for ax1 in range(num_1d_tapers):
            for ax2 in range(num_1d_tapers):
                np.matmul(
                    tapers_1d[:, ax1, np.newaxis],
                    tapers_1d[:, ax2, np.newaxis].T,
                    out=taper_2d,
                )

                for m in range(blocks.shape[0]):
                    np.multiply(blocks[m, :, :], taper_2d, out=blocks_tapered)
                    blocks_mt_post_fft = fft.fftn(blocks_tapered, axes=(-2, -1))
                    blocks_mt += abs2(blocks_mt_post_fft)

        blocks_mt /= blocks.shape[0] ** 2
        blocks_mt /= tapers_1d.shape[0] ** 2

        amplitude_spectrum = fft.fftshift(
            blocks_mt
        )  # max difference 10^-13, max relative difference 10^-14

        return Image(amplitude_spectrum)

    def elliptical_average(self, ffbbasis, amplitude_spectrum, circular):
        """
        Computes radial/elliptical average of the power spectrum

        :param ffbbasis: FFBBasis instance.
        :param amplitude_spectrum: Power spectrum.
        :param circular: True for radial averaging and False for elliptical averaging.
        :return: PSD and noise as 2-tuple of NumPy arrays.
        """

        # RCOPT, come back and change the indices for this method
        coefs_s = ffbbasis.evaluate_t(amplitude_spectrum).asnumpy().copy().T
        coefs_n = coefs_s.copy()

        coefs_s[np.argwhere(ffbbasis.angular_indices == 1)] = 0
        if circular:
            coefs_s[np.argwhere(ffbbasis.angular_indices == 2)] = 0
            noise = amplitude_spectrum
        else:
            coefs_n[np.argwhere(ffbbasis.angular_indices == 0)] = 0
            coefs_n[np.argwhere(ffbbasis.angular_indices == 2)] = 0
            noise = Coef(ffbbasis, coefs_n.T).evaluate()

        psd = Coef(ffbbasis, coefs_s.T).evaluate()

        return psd, noise

    def background_subtract_1d(
        self, amplitude_spectrum, linprog_method="highs", n_low_freq_cutoffs=14
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

            # The original code used `bounds`,
            #   but for many problems, linprog reports infeasable constraints.
            # In practice for a micrograph from the paper, and our tutorial,
            #   the code seems to work better without it...
            # ASPIRE #417

            x = linprog(
                f,
                A_ub=A,
                b_ub=np.zeros(A.shape[0]),
                method=linprog_method,
            )

            if not x.success:
                raise RuntimeError("Linear program did not succeed. Halting")

            background = x.x[N:]

            bs_psd = signal - background

            final_signal[low_freq_cutoff - 1, low_freq_cutoff:] = bs_psd.T
            # expected difference: 10^-7 (absolute)
            final_background[low_freq_cutoff - 1, low_freq_cutoff:] = background.T

        return final_signal, final_background

    def opt1d(
        self,
        amplitude_spectrum,
        pixel_size,
        cs,
        lmbd,
        w,
        N,
        min_defocus=500,
        max_defocus=10000,
    ):
        """
        Find optimal defocus for the radially symmetric case (where no astigmatism is present)

        :param amplitude_spectrum: Estimated power specrtum.
        :param pixel_size: Pixel size in \u212b (angstrom).
        :param cs: Spherical aberration in mm.
        :param lmbd: Electron wavelength \u212b (angstrom).
        :param w: Amplitude contrast.
        :param N: Number of rows (or columns) in the estimate power spectrum.
        :param min_defocus: Start of defocus loop scan.
        :param max_defocus: End of defocus loop scan.
        :return: 2-tuple of NumPy arrays (Estimated average of defocus and low_freq_cutoff)
        """

        center = N // 2

        grid = grid_1d(N, normalized=True, dtype=self.dtype)
        rb = grid["r"][center:] / 2

        r_ctf = rb * (10 / pixel_size)  # units: inverse nm

        signal = amplitude_spectrum.T
        signal = np.maximum(0.0, signal)
        signal = np.sqrt(signal)
        signal = signal[: 3 * signal.shape[0] // 4]

        r_ctf_sq = r_ctf**2
        c = np.zeros((max_defocus - min_defocus, signal.shape[1]), dtype=self.dtype)

        for f in range(min_defocus, max_defocus):
            ctf_im = np.abs(
                np.sin(
                    np.pi * lmbd * f * r_ctf_sq
                    - 0.5 * np.pi * (lmbd**3) * cs * 1e6 * r_ctf_sq**2
                    + w
                )
            )
            ctf_im = ctf_im[: signal.shape[0]]

            ctf_im = np.reshape(ctf_im, (ctf_im.shape[0], 1))
            ctf_im = np.tile(ctf_im, (1, signal.shape[1]))

            for m in range(0, signal.shape[1]):
                signal[:, m] = signal[:, m] - np.mean(signal[m + 1 :, m], axis=0)
                ctf_im[:, m] = ctf_im[:, m] - np.mean(ctf_im[m + 1 :, m], axis=0)
                ctf_im[: m + 1, m] = np.zeros((m + 1))
                signal[: m + 1, m] = np.zeros((m + 1))

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

        N = signal.shape[1]
        grid = grid_2d(N, normalized=False, indexing="yx", dtype=self.dtype)

        radii = np.sqrt((grid["x"] / 2) ** 2 + (grid["y"] / 2) ** 2)

        background = np.zeros(signal.shape, dtype=self.dtype)
        for r in range(max_col + 2, background_p1.shape[1]):
            background[:, (r < radii) & (radii <= r + 1)] = background_p1[max_col, r]
        mask = radii <= max_col + 2
        background[:, mask] = signal[:, mask]

        signal = signal - background
        signal = np.maximum(0, signal)

        return Image(signal), Image(background)

    def pca(self, signal, pixel_size, g_min, g_max):
        """

        :param signal: Estimated power spectrum.
        :param pixel_size: Pixel size in \u212b (angstrom).
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :return: ratio.
        """

        # RCOPT
        signal = signal.asnumpy()[0].T

        N = signal.shape[0]
        center = N // 2

        grid = grid_2d(N, normalized=True, indexing="yx", dtype=self.dtype)

        r_ctf = grid["r"] / 2 * (10 / pixel_size)

        grid = grid_2d(N, normalized=False, indexing="yx", dtype=self.dtype)
        X = grid["x"]
        Y = grid["y"]

        signal = signal - np.min(signal)

        rad_sq_min = N * pixel_size / g_min
        rad_sq_max = N * pixel_size / g_max

        min_limit = r_ctf[center, (center + np.floor(rad_sq_min)).astype(int)]
        signal[r_ctf < min_limit] = 0

        max_limit = r_ctf[center, (center + np.ceil(rad_sq_max)).astype(int)]
        signal = np.where(r_ctf > max_limit, 0, signal)

        moment_02 = Y**2 * signal
        moment_02 = np.sum(moment_02, axis=(0, 1))

        moment_11 = Y * X * signal
        moment_11 = np.sum(moment_11, axis=(0, 1))

        moment_20 = X**2 * signal
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
        r,
        theta,
        pixel_size,
        g_min,
        g_max,
        amplitude_contrast,
        lmbd,
        cs,
    ):
        """
        Runs gradient ascent to optimize defocus parameters

        :param signal: Estimated power spectrum
        :param df1: Defocus value in the direction perpendicular to df2.
        :param df2: Defocus value in the direction perpendicular to df1.
        :param angle_ast: Angle between df1 and the x-axis, Radians.
        :param r: Magnitude of spatial frequencies.
        :param theta: Phase of spatial frequencies.
        :param pixel_size: Pixel size in \u212b (angstrom).
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :param amplitude_contrast: Amplitude contrast.
        :param lmbd: Electron wavelength \u212b (angstrom).
        :param cs: Spherical aberration in mm.
        :return: Optimal defocus parameters
        """

        # step size
        alpha1 = 1e5
        alpha2 = 1e4

        # initialization
        x = df1 + df2
        y = (df1 - df2) * np.cos(2 * angle_ast)
        z = (df1 - df2) * np.sin(2 * angle_ast)

        a = np.pi * lmbd * r**2 / 2
        b = np.pi * lmbd**3 * cs * 1e6 * r**4 / 2 - np.full(
            shape=r.shape, fill_value=amplitude_contrast, dtype=self.dtype
        )

        signal = signal.asnumpy()[0].T
        N = signal.shape[1]
        center = N // 2

        rad_sq_min = N * pixel_size / g_min
        rad_sq_max = N * pixel_size / g_max

        max_val = r[center, int(center - 1 + np.floor(rad_sq_max))]
        min_val = r[center, int(center - 1 + np.ceil(rad_sq_min))]

        mask = (r <= max_val) & (r > min_val)
        a = a[mask]
        b = b[mask]
        signal = signal[..., mask]
        r = r[mask]
        theta = theta[mask]

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

        return df1, df2, angle_ast, p

    # Note, This doesn't actually use anything from the class.
    # It is used in a solver loop of some sort, so it may not be correct
    # to just use what is avail in the obj.
    def write_star(self, name, params_dict, output_dir):
        """
        Writes CTF parameters to starfile for a single micrograph.
        """
        data_block = {}
        data_block["_rlnMicrographName"] = name
        data_block["_rlnDefocusU"] = params_dict["defocus_u"]  # Should already be in A
        data_block["_rlnDefocusV"] = params_dict["defocus_v"]  # Should already be in A
        data_block["_rlnDefocusAngle"] = (
            params_dict["defocus_ang"] * 180 / np.pi
        )  # Convert from radian to degree
        data_block["_rlnSphericalAberration"] = params_dict["cs"]
        data_block["_rlnAmplitudeContrast"] = params_dict["amplitude_contrast"]
        data_block["_rlnVoltage"] = params_dict["voltage"]
        data_block["_rlnMicrographPixelSize"] = params_dict["pixel_size"]
        blocks = OrderedDict()
        blocks["root"] = data_block
        star = StarFile(blocks=blocks)
        star.write(os.path.join(output_dir, os.path.splitext(name)[0]) + ".star")


def estimate_ctf(
    data_folder,
    pixel_size=1.0,
    cs=2.0,
    amplitude_contrast=0.07,
    voltage=300,
    num_tapers=2,
    psd_size=512,
    g_min=30,
    g_max=5,
    output_dir="results",
    dtype=np.float32,
    save_ctf_images=False,
    save_noise_images=False,
):
    """
    Given paramaters estimates CTF from experimental data
    and returns CTF as a mrc file.
    """

    dtype = np.dtype(dtype)
    assert dtype in (np.float32, np.float64)

    dir_content = os.scandir(data_folder)

    mrc_files = [f.name for f in dir_content if os.path.splitext(f)[1] == ".mrc"]
    mrcs_files = [f.name for f in dir_content if os.path.splitext(f)[1] == ".mrcs"]
    file_names = mrc_files + mrcs_files

    amp = amplitude_contrast
    amplitude_contrast = np.arctan(
        amplitude_contrast / np.sqrt(1 - amplitude_contrast**2)
    )

    lmbd = voltage_to_wavelength(voltage) / 10  # Convert from angstrom to nm

    ctf_object = CtfEstimator(
        pixel_size, cs, amplitude_contrast, voltage, psd_size, num_tapers, dtype=dtype
    )

    # Note for repro debugging, suggest use of doubles,
    #   closer to original code.
    ffbbasis = FFBBasis2D((psd_size, psd_size), 2, dtype=dtype)

    results = {}
    for name in file_names:
        with mrcfile.open(
            os.path.join(data_folder, name), mode="r", permissive=True
        ) as mrc:
            micrograph = mrc.data

        # Try to match dtype used in Basis instance
        micrograph = micrograph.astype(dtype, copy=False)

        micrograph_blocks = ctf_object.preprocess_micrograph(micrograph, psd_size)

        tapers_1d = ctf_object.tapers(psd_size, num_tapers / 2, num_tapers)

        signal_observed = ctf_object.estimate_psd(micrograph_blocks, tapers_1d)

        amplitude_spectrum, _ = ctf_object.elliptical_average(
            ffbbasis, signal_observed, True
        )  # absolute differenceL 10^-14. Relative error: 10^-7

        # Optionally changing to: linprog_method='simplex',
        # will more deterministically repro results in exchange for speed.
        # linprog_method was changed from 'interior-point' to 'highs' due to
        # "interior-point' being deprecated.
        signal_1d, background_1d = ctf_object.background_subtract_1d(
            amplitude_spectrum, linprog_method="highs"
        )

        avg_defocus, low_freq_skip = ctf_object.opt1d(
            signal_1d,
            pixel_size,
            cs,
            lmbd,  # (angstrom)
            amplitude_contrast,
            signal_observed.shape[-1],
        )

        low_freq_skip = 12
        signal, background_2d = ctf_object.background_subtract_2d(
            signal_observed, background_1d, low_freq_skip
        )

        ratio = ctf_object.pca(signal_observed, pixel_size, g_min, g_max)

        signal, additional_background = ctf_object.elliptical_average(
            ffbbasis, signal.sqrt(), False
        )

        background_2d = background_2d + additional_background

        initial_df1 = (avg_defocus * 2) / (1 + ratio)
        initial_df2 = (avg_defocus * 2) - initial_df1

        grid = grid_2d(psd_size, normalized=True, indexing="yx", dtype=dtype)

        r_ctf = grid["r"] / 2 * (10 / pixel_size)
        theta = grid["phi"]

        angle = -5 / 12 * np.pi  # Radians (-75 degrees)
        cc_array = np.zeros((6, 4))
        for a in range(0, 6):
            df1, df2, angle_ast, p = ctf_object.gd(
                signal,
                initial_df1,
                initial_df2,
                angle + a * np.pi / 6.0,  # Radians, + a*30degrees
                r_ctf,
                theta,
                pixel_size,
                g_min,
                g_max,
                amplitude_contrast,
                lmbd,  # (angstrom)
                cs,
            )

            cc_array[a, 0] = df1
            cc_array[a, 1] = df2
            cc_array[a, 2] = angle_ast  # Radians
            cc_array[a, 3] = p
        ml = np.argmax(cc_array[:, 3], -1)

        result = {
            "defocus_u": cc_array[ml, 0] * 10,  # Convert from nm to A
            "defocus_v": cc_array[ml, 1] * 10,  # Convert from nm to A
            "defocus_ang": cc_array[ml, 2],  # Radians
            "cs": cs,
            "voltage": voltage,
            "pixel_size": pixel_size,
            "amplitude_contrast": amp,
        }
        results[name] = result

        # we write each micrograph's ctf parameters to an individual starfile
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        ctf_object.write_star(name, result, output_dir)

        if save_noise_images:
            with mrcfile.new(
                os.path.join(output_dir, os.path.splitext(name)[0] + "_noise.mrc"),
                overwrite=True,
            ) as mrc:
                mrc.set_data(background_2d.asnumpy()[0].astype(np.float32))
                mrc.voxel_size = pixel_size

        if save_ctf_images:
            ctf_object.set_df1(cc_array[ml, 0])  # angstrom
            ctf_object.set_df2(cc_array[ml, 1])  # angstrom
            ctf_object.set_angle(cc_array[ml, 2])  # Radians
            ctf_object.generate_ctf()
            df = (cc_array[ml, 0] + cc_array[ml, 1]) * np.ones(
                theta.shape, theta.dtype
            ) + (cc_array[ml, 0] - cc_array[ml, 1]) * np.cos(
                2 * theta - 2 * cc_array[ml, 2] * np.ones(theta.shape, theta.dtype)
            )
            ctf_im = -np.sin(
                np.pi * lmbd * r_ctf**2 / 2 * (df - lmbd**2 * r_ctf**2 * cs * 1e6)
                + amplitude_contrast
            )
            ctf_signal = np.zeros(ctf_im.shape, ctf_im.dtype)
            ctf_signal[: ctf_im.shape[0] // 2, :] = ctf_im[: ctf_im.shape[0] // 2, :]
            ctf_signal[ctf_im.shape[0] // 2 + 1 :, :] = signal.asnumpy()[
                :, :, ctf_im.shape[0] // 2 + 1
            ]

            with mrcfile.new(
                os.path.join(output_dir, os.path.splitext(name)[0] + ".ctf"),
                overwrite=True,
            ) as mrc:
                mrc.set_data(np.float32(ctf_signal))
                mrc.voxel_size = pixel_size

    return results
