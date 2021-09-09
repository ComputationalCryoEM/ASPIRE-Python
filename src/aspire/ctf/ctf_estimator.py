"""
Created on Sep 10, 2019
@author: Ayelet Heimowitz, Amit Moscovich

Integrated into ASPIRE by Garrett Wright Feb 2021.
"""

import logging
import os

import mrcfile
import numpy as np
from numpy import linalg as npla
from scipy.optimize import linprog
from scipy.signal.windows import dpss

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.image import Image
from aspire.numeric import fft
from aspire.operators import voltage_to_wavelength
from aspire.utils import abs2, complex_type
from aspire.utils.coor_trans import grid_1d, grid_2d

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

        :param pixel_size: Size of the pixel in \u212b (Angstrom).
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
        self.lmbd = voltage_to_wavelength(voltage) / 10.0  # (Angstrom)
        self.dtype = np.dtype(dtype)

        grid = grid_2d(psd_size, normalized=True, dtype=self.dtype)

        # Note this mesh for x,y is transposed, and range is -half to half.
        rb = np.sqrt((grid["x"] / 2) ** 2 + (grid["y"] / 2) ** 2).T

        self.r_ctf = rb * (10 / pixel_size)  # units: inverse nm
        # Note this mesh for theta is transposed.
        self.theta = grid["phi"].T
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
            1 - self.amplitude_contrast ** 2
        )

        chi = (
            defocus_factor
            - np.pi * self.lmbd ** 3 * self.cs * 1e6 * self.r_ctf ** 2 / 2
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

        size_x = micrograph.shape[1]
        size_y = micrograph.shape[0]

        step_size = block_size // 2
        range_y = size_y // step_size - 1
        range_x = size_x // step_size - 1

        block_list = [
            micrograph[
                i * step_size : (i + 2) * step_size, j * step_size : (j + 2) * step_size
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

        blocks -= blocks_sum / (block_size ** 2)

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
        coeffs_s = ffbbasis.evaluate_t(amplitude_spectrum).T
        coeffs_n = coeffs_s.copy()

        coeffs_s[np.argwhere(ffbbasis._indices["ells"] == 1)] = 0
        if circular:
            coeffs_s[np.argwhere(ffbbasis._indices["ells"] == 2)] = 0
            noise = amplitude_spectrum
        else:
            coeffs_n[np.argwhere(ffbbasis._indices["ells"] == 0)] = 0
            coeffs_n[np.argwhere(ffbbasis._indices["ells"] == 2)] = 0
            noise = ffbbasis.evaluate(coeffs_n.T)

        psd = ffbbasis.evaluate(coeffs_s.T)

        return psd, noise

    def background_subtract_1d(
        self, amplitude_spectrum, linprog_method="interior-point", n_low_freq_cutoffs=14
    ):
        """
        Estimate and subtract the background from the power spectrum

        :param amplitude_spectrum: Estimated power spectrum
        :param linprog_method: Method passed to linear progam solver (scipy.optimize.linprog).  Defaults to 'interior-point'.
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

        amplitude_spectrum = amplitude_spectrum[center, center:]
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
            convex_condition = convex_condition[1 : N - 1]

            positivity_condition = np.concatenate(
                (np.zeros((N, N)), -1 * np.eye(N)), axis=1
            )

            A = np.concatenate(
                (superposition_condition, convex_condition, positivity_condition),
                axis=0,
            )

            x_bound_lst = [
                (signal[i], signal[i], -1 * np.inf, np.inf)
                for i in range(signal.shape[0])
            ]
            x_bound = np.asarray(x_bound_lst, A.dtype)
            x_bound = np.concatenate((x_bound[:, :2], x_bound[:, 2:]), axis=0)

            x = linprog(
                f,
                A_ub=A,
                b_ub=np.zeros(A.shape[0]),
                bounds=x_bound,
                method=linprog_method,
            )
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
        :param pixel_size: Pixel size in \u212b (Angstrom).
        :param cs: Spherical aberration in mm.
        :param lmbd: Electron wavelength \u212b (Angstrom).
        :param w: Amplitude contrast.
        :param N: Number of rows (or columns) in the estimate power spectrum.
        :param min_defocus: Start of defocus loop scan.
        :param max_defocus: End of defocus loop scan.
        :return: 2-tuple of NumPy arrays (Estimated average of defocus and low_freq_cutoff)
        """

        center = N // 2

        grid = grid_1d(N, normalized=True, dtype=self.dtype)
        rb = grid["x"][0][center:] / 2

        r_ctf = rb * (10 / pixel_size)  # units: inverse nm

        signal = amplitude_spectrum.T
        signal = np.maximum(0.0, signal)
        signal = np.sqrt(signal)
        signal = signal[: 3 * signal.shape[0] // 4]

        r_ctf_sq = r_ctf ** 2
        c = np.zeros((max_defocus - min_defocus, signal.shape[1]), dtype=self.dtype)

        for f in range(min_defocus, max_defocus):
            ctf_im = np.abs(
                np.sin(
                    np.pi * lmbd * f * r_ctf_sq
                    - 0.5 * np.pi * (lmbd ** 3) * cs * 1e6 * r_ctf_sq ** 2
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

            Sx = np.sqrt(np.sum(ctf_im ** 2, axis=0))
            Sy = np.sqrt(np.sum(signal ** 2, axis=0))
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
        grid = grid_2d(N, normalized=False, dtype=self.dtype)

        radii = np.sqrt((grid["x"] / 2) ** 2 + (grid["y"] / 2) ** 2).T

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
        :param pixel_size: Pixel size in \u212b (Angstrom).
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :return: ratio.
        """

        # RCOPT
        signal = signal.asnumpy()[0].T

        N = signal.shape[0]
        center = N // 2

        grid = grid_2d(N, normalized=True, dtype=self.dtype)
        rb = np.sqrt((grid["x"] / 2) ** 2 + (grid["y"] / 2) ** 2).T

        r_ctf = rb * (10 / pixel_size)

        grid = grid_2d(N, normalized=False, dtype=self.dtype)
        X = grid["x"].T
        Y = grid["y"].T

        signal -= np.min(signal)

        rad_sq_min = N * pixel_size / g_min
        rad_sq_max = N * pixel_size / g_max

        min_limit = r_ctf[center, (center + np.floor(rad_sq_min)).astype(int)]
        signal[r_ctf < min_limit] = 0

        max_limit = r_ctf[center, (center + np.ceil(rad_sq_max)).astype(int)]
        signal = np.where(r_ctf > max_limit, 0, signal)

        moment_02 = Y ** 2 * signal
        moment_02 = np.sum(moment_02, axis=(0, 1))

        moment_11 = Y * X * signal
        moment_11 = np.sum(moment_11, axis=(0, 1))

        moment_20 = X ** 2 * signal
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
        :param pixel_size: Pixel size in \u212b (Angstrom).
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :param amplitude_contrast: Amplitude contrast.
        :param lmbd: Electron wavelength \u212b (Angstrom).
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

        a = np.pi * lmbd * r ** 2 / 2
        b = np.pi * lmbd ** 3 * cs * 1e6 * r ** 4 / 2 - np.full(
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

        sum_A = np.sum(signal ** 2)

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
            c2 = np.sqrt(sum_A * np.sum(outer_sine ** 2))

            # gradients of numerator
            dx_c1 = np.sum(np.sign(outer_sine) * outer_cosine * a * signal)

            dy_c1 = np.sum(
                np.sign(outer_sine) * outer_cosine * a * np.cos(2 * theta) * signal
            )
            dz_c1 = np.sum(
                np.sign(outer_sine) * outer_cosine * a * np.sin(2 * theta) * signal
            )

            derivative_sqrt = 1 / (2 * np.sqrt(sum_A * np.sum(outer_sine ** 2)))

            derivative_sine2 = 2 * outer_sine * outer_cosine

            #  gradients of denomenator
            dx_c2 = derivative_sqrt * sum_A * np.sum(derivative_sine2 * sine_x_term)
            dy_c2 = derivative_sqrt * sum_A * np.sum(derivative_sine2 * sine_y_term)
            dz_c2 = derivative_sqrt * sum_A * np.sum(derivative_sine2 * sine_z_term)

            # gradients
            dx = (dx_c1 * c2 - dx_c2 * c1) / c2 ** 2
            dy = (dy_c1 * c2 - dy_c2 * c1) / c2 ** 2
            dz = (dz_c1 * c2 - dz_c2 * c1) / c2 ** 2

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
        c2 = np.sqrt(sum_A * np.sum(outer_sine ** 2))

        p = c1 / c2

        return df1, df2, angle_ast, p

    # Note, This doesn't actually use anything from the class.
    # It is used in a solver loop of some sort, so it may not be correct
    # to just use what is avail in the obj.
    def write_star(self, df1, df2, ang, cs, voltage, pixel_size, amp, name):
        """
        Writes starfile.
        """

        if os.path.isdir("results") is False:
            os.mkdir("results")

        f = open("results/" + os.path.splitext(name)[0] + ".log", "w")
        f.write(
            "data_root\n\nloop_\n_rlnMicrographName #1\n_rlnDefocusU #2\n_rlnDefocusV #3\n_rlnDefocusAngle #4\n"
        )
        f.write(
            "_rlnSphericalAberration #5\n_rlnAmplitudeContrast #6\n_rlnVoltage #7\n_rlnDetectorPixelSize #8\n"
        )
        f.write(name)
        f.write("\t")
        f.write("%5.8f" % (df1))
        f.write("\t")
        f.write("%5.8f" % (df2))
        f.write("\t")
        f.write("%5.8f" % (ang))
        f.write("\t")
        f.write("%5.2f" % (cs))
        f.write("\t")
        f.write("%5.4f" % (amp))
        f.write("\t")
        f.write("%5.4f" % (voltage))
        f.write("\t")
        f.write("%5.4f" % (pixel_size))
        f.close()


def estimate_ctf(
    data_folder,
    pixel_size,
    cs,
    amplitude_contrast,
    voltage,
    num_tapers,
    psd_size,
    g_min,
    g_max,
    output_dir,
    dtype=np.float32,
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
        amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2)
    )

    lmbd = voltage_to_wavelength(voltage) / 10  # (Angstrom)

    ctf_object = CtfEstimator(
        pixel_size, cs, amplitude_contrast, voltage, psd_size, num_tapers, dtype=dtype
    )

    # Note for repro debugging, suggest use of doubles,
    #   closer to original code.
    ffbbasis = FFBBasis2D((psd_size, psd_size), 2, dtype=dtype)

    results = []
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
        signal_1d, background_1d = ctf_object.background_subtract_1d(
            amplitude_spectrum, linprog_method="interior-point"
        )

        avg_defocus, low_freq_skip = ctf_object.opt1d(
            signal_1d,
            pixel_size,
            cs,
            lmbd,  # (Angstrom)
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

        grid = grid_2d(psd_size, normalized=True, dtype=dtype)

        rb = np.sqrt((grid["x"] / 2) ** 2 + (grid["y"] / 2) ** 2).T
        r_ctf = rb * (10 / pixel_size)
        theta = grid["phi"].T

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
                lmbd,  # (Angstrom)
                cs,
            )

            cc_array[a, 0] = df1
            cc_array[a, 1] = df2
            cc_array[a, 2] = angle_ast  # Radians
            cc_array[a, 3] = p
        ml = np.argmax(cc_array[:, 3], -1)

        result = (
            cc_array[ml, 0],
            cc_array[ml, 1],
            cc_array[ml, 2],  # Radians
            cs,
            voltage,
            pixel_size,
            amp,
            name,
        )

        ctf_object.write_star(*result)
        results.append(result)

        ctf_object.set_df1(cc_array[ml, 0])
        ctf_object.set_df2(cc_array[ml, 1])
        ctf_object.set_angle(cc_array[ml, 2])  # Radians
        ctf_object.generate_ctf()

        with mrcfile.new(
            output_dir + "/" + os.path.splitext(name)[0] + "_noise.mrc", overwrite=True
        ) as mrc:
            mrc.set_data(background_2d[0].astype(np.float32))
            mrc.voxel_size = pixel_size
            mrc.close()

        df = (cc_array[ml, 0] + cc_array[ml, 1]) * np.ones(theta.shape, theta.dtype) + (
            cc_array[ml, 0] - cc_array[ml, 1]
        ) * np.cos(2 * theta - 2 * cc_array[ml, 2] * np.ones(theta.shape, theta.dtype))
        ctf_im = -np.sin(
            np.pi * lmbd * r_ctf ** 2 / 2 * (df - lmbd ** 2 * r_ctf ** 2 * cs * 1e6)
            + amplitude_contrast
        )
        ctf_signal = np.zeros(ctf_im.shape, ctf_im.dtype)
        ctf_signal[: ctf_im.shape[0] // 2, :] = ctf_im[: ctf_im.shape[0] // 2, :]
        ctf_signal[ctf_im.shape[0] // 2 + 1 :, :] = signal[
            :, :, ctf_im.shape[0] // 2 + 1
        ]

        with mrcfile.new(
            output_dir + "/" + os.path.splitext(name)[0] + ".ctf", overwrite=True
        ) as mrc:
            mrc.set_data(np.float32(ctf_signal))
            mrc.voxel_size = pixel_size
            mrc.close()

    return results
