"""
Created on Sep 10, 2019
@author: Ayelet Heimowitz, Amit Moscovich

Integrated into ASPIRE by Garrett Wright Feb 2021.
"""

import logging
import os

import numpy as np
from numpy import linalg as npla
from scipy.optimize import linprog

from aspire.image import Image
from aspire.numeric import fft
from aspire.operators import voltage_to_wavelength
from aspire.utils import abs2, complex_type, eigs

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
        self.lmbd = 10 * voltage_to_wavelength(voltage)
        self.dtype = np.dtype(dtype)

        center = psd_size // 2
        [X, Y] = np.meshgrid(
            np.arange(0 - center, psd_size - center, dtype=self.dtype) / psd_size,
            np.arange(0 - center, psd_size - center, dtype=self.dtype) / psd_size,
        )

        rb = np.sqrt(np.square(X) + np.square(Y))

        self.r_ctf = rb * (10 / pixel_size)
        self.theta = np.arctan2(Y, X)
        self.defocus1 = 0
        self.defocus2 = 0
        self.angle = 0
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

        :param angle: Angle (in degrees) between df1 and the x-axis.
        """

        self.angle = angle

    def generate_ctf(self):
        """
        Generates internal representation of the Contrast Transfer Function using parameters from this instance.
        """

        astigmatism_angle = np.reshape(
            np.repeat(
                self.angle, np.multiply(self.theta.shape[0], self.theta.shape[1])
            ),
            self.theta.shape,
        )
        defocus_sum = np.reshape(
            np.repeat(
                self.defocus1 + self.defocus2,
                np.multiply(self.theta.shape[0], self.theta.shape[1]),
            ),
            self.theta.shape,
        )
        defocus = defocus_sum + np.multiply(
            self.defocus1 - self.defocus2, np.cos(2 * (self.theta - astigmatism_angle))
        )
        defocus_factor = np.pi * self.lmbd * np.multiply(self.r_ctf, defocus) / 2
        amplitude_contrast_term = np.divide(
            self.amplitude_contrast, np.sqrt(1 - np.square(self.amplitude_contrast))
        )
        chi = (
            defocus_factor
            - np.pi
            * np.power(self.lmbd, 3)
            * self.cs
            * np.power(10, 6)
            * np.square(self.r_ctf)
            / 2
            + amplitude_contrast_term
        )
        h = -np.sin(chi)
        self.h = h

    def preprocess(self, micrograph, block_size):
        """
        Preprocess CTF of micrograph using block_size.

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
        block = np.asarray(block_list, dtype=micrograph.dtype)

        # Create a sum and reshape so it may be broadcast with `block`.
        block_sum = np.sum(block, axis=(-1, -2))[:, np.newaxis, np.newaxis]

        block = block - (
            block_sum / (block_size ** 2)
        )  # equals to the matlab version (11-7-19)

        return block

    def tapers(self, N, R, L):
        """
        Compute data tapers (which are discrete prolate spheroidal sequences (dpss))

        :param N: Size of each taper
        :param R: Spectral resolution
        :param L: Number of tapers
        :return: NumPy array of data tapers
        """

        k, el = np.meshgrid(
            np.arange(N, dtype=self.dtype), np.arange(N, dtype=self.dtype)
        )

        denom = np.pi * (k - el)
        denom = denom + np.eye(N, dtype=self.dtype)
        phi_R = np.divide(np.sin(np.pi * R * (k - el)), denom)
        np.fill_diagonal(phi_R, 1)  # absolute difference from Matlab 10^-18

        data_tapers, _ = eigs(phi_R, L)

        return data_tapers

    def estimate_psd(self, blocks, tapers_1d, num_1d_tapers):
        """
        Estimate the power spectrum of the micrograph using the multi-taper method

        :param blocks: 3-D NumPy array containing windows extracted from the micrograph in the preprocess function.
        :param tapers_1d: NumPy array of data tapers.
        :param num_1d_tapers: number of 1-D data tapers
        :return: NumPy array of estimated power spectrum.
        """

        tapers_1d = tapers_1d.astype(complex_type(self.dtype), copy=False)

        blocks_mt = np.zeros(blocks[0, :, :].shape, dtype=self.dtype)

        blocks_tapered = np.zeros(blocks[0, :, :].shape, dtype=complex_type(self.dtype))

        taper_2d = np.zeros(
            (blocks.shape[1], blocks.shape[2]), dtype=complex_type(self.dtype)
        )

        for x in range(num_1d_tapers ** 2):
            np.matmul(
                np.reshape(tapers_1d[:, x // num_1d_tapers], (tapers_1d.shape[0], 1)),
                np.reshape(tapers_1d[:, x % num_1d_tapers], (1, tapers_1d.shape[0])),
                out=taper_2d,
            )
            for m in range(blocks.shape[0]):
                np.multiply(blocks[m, :, :], taper_2d, out=blocks_tapered)
                blocks_mt_post_fft = fft.fftn(blocks_tapered, axes=(-2, -1))
                blocks_mt += abs2(blocks_mt_post_fft)

        blocks_mt /= blocks.shape[0] ** 2
        blocks_mt /= tapers_1d.shape[0] ** 2

        thon_rings = fft.fftshift(
            blocks_mt
        )  # max difference 10^-13, max relative difference 10^-14

        return Image(thon_rings)

    def elliptical_average(self, ffbbasis, thon_rings, circular):
        """
        Computes radial/elliptical average of the power spectrum

        :param ffbbasis: FFBBasis instance.
        :param thon_rings: Power spectrum.
        :param circular: True for radial averaging and False for elliptical averaging.
        :return: PSD and noise as 2-tuple of NumPy arrays.
        """

        # RCOPT, come back and change the indices for this method
        coeffs_s = ffbbasis.evaluate_t(thon_rings).T
        coeffs_n = coeffs_s.copy()

        coeffs_s[np.argwhere(ffbbasis._indices["ells"] == 1)] = 0
        if circular:
            coeffs_s[np.argwhere(ffbbasis._indices["ells"] == 2)] = 0
            noise = thon_rings
        else:
            coeffs_n[np.argwhere(ffbbasis._indices["ells"] == 0)] = 0
            coeffs_n[np.argwhere(ffbbasis._indices["ells"] == 2)] = 0
            noise = ffbbasis.evaluate(coeffs_n.T)

        psd = ffbbasis.evaluate(coeffs_s.T)

        return psd, noise

    def background_subtract_1d(self, thon_rings, linprog_method="interior-point"):
        """
        Estimate and subtract the background from the power spectrum

        :param thon_rings: Estimated power spectrum
        :param linprog_method: Method passed to linear progam solver (scipy.optimize.linprog).  Defaults to 'interior-point'.
        :return: 2-tuple of NumPy arrays (PSD after noise subtraction and estimated noise)
        """

        # compute radial average
        center = thon_rings.shape[-1] // 2

        thon_rings = thon_rings[..., center, center:]

        thon_rings = thon_rings[..., 0 : 3 * thon_rings.shape[-1] // 4]

        final_signal = np.zeros((thon_rings.shape[-1], 13), dtype=self.dtype)
        final_background = np.ones((thon_rings.shape[-1], 13), dtype=self.dtype)

        for m in range(1, 14):
            signal = thon_rings[..., m:]
            signal = np.ravel(signal)
            N = thon_rings.shape[-1] - m

            f = np.concatenate((np.ones(N), -1 * np.ones(N)), axis=0)

            superposition_condition = np.concatenate(
                (-1 * np.eye(N), np.eye(N)), axis=1
            )

            monotone_condition = np.zeros((N, N))
            monotone_condition[np.arange(N - 1), np.arange(N - 1)] = -1
            monotone_condition[np.arange(N - 1), np.arange(1, N)] = 1
            monotone_condition = np.concatenate(
                (np.zeros((N, N)), monotone_condition), axis=1
            )

            convex_condition = np.zeros((N, N))
            convex_condition[np.arange(N - 2), np.arange(N - 2)] = -1
            convex_condition[np.arange(N - 2), np.arange(1, N - 1)] = 2
            convex_condition[np.arange(N - 2), np.arange(2, N)] = -1
            convex_condition = np.concatenate(
                (np.zeros((N, N)), convex_condition), axis=1
            )

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

            final_signal[m:, m - 1] = bs_psd
            final_background[m:, m - 1] = background  # difference: 10^-7 (absolute)

        return final_signal, final_background

    def opt1d(self, thon_rings, pixel_size, cs, lmbd, w, N):
        """
        Find optimal defocus for the radially symmetric case (where no astigmatism is present)

        :param thon_rings: Estimated power specrtum.
        :param pixel_size: Pixel size in \u212b (Angstrom).
        :param cs: Spherical aberration in mm.
        :param lmbd: Electron wavelength.
        :param w: Amplitude contrast.
        :param N: Number of rows (or columns) in the estimate power spectrum.
        :return: 2-tuple of NumPy arrays (Estimated average of defocus and an internal variable)
        """

        center = N // 2
        [X, Y] = np.meshgrid(
            np.arange(0 - center, N - center, dtype=self.dtype) / N,
            np.arange(0 - center, N - center, dtype=self.dtype) / N,
        )

        rb = np.sqrt(np.square(X) + np.square(Y))
        rb = rb[center, center:]
        r_ctf = rb * (10 / pixel_size)

        signal = thon_rings
        signal = np.maximum(0.0, signal)
        signal = np.sqrt(signal)
        signal = signal[: 3 * signal.shape[0] // 4]

        r_ctf_sq = np.square(r_ctf)
        c = np.zeros((9500, 13), dtype=self.dtype)

        for f in range(500, 10000):
            ctf_im = np.abs(
                np.sin(
                    np.pi * lmbd * f * r_ctf_sq
                    - 0.5 * np.pi * (lmbd ** 3) * cs * (10 ** 6) * np.square(r_ctf_sq)
                    + w
                )
            )
            ctf_im = ctf_im[: signal.shape[0]]

            ctf_im = np.reshape(ctf_im, (ctf_im.shape[0], 1))
            ctf_im = np.tile(ctf_im, (1, 13))

            for m in range(0, 13):
                signal[:, m] = signal[:, m] - np.mean(signal[m + 1 :, m], axis=0)
                ctf_im[:, m] = ctf_im[:, m] - np.mean(ctf_im[m + 1 :, m], axis=0)
                ctf_im[: m + 1, m] = np.zeros((m + 1))
                signal[: m + 1, m] = np.zeros((m + 1))

            Sx = np.sqrt(np.sum(np.square(ctf_im), axis=0))
            Sy = np.sqrt(np.sum(np.square(signal), axis=0))
            c[f - 500, :] = np.sum(np.multiply(ctf_im, signal), axis=0) / (Sx * Sy)

        max_c = np.argmax(c)
        arr_max = np.unravel_index(max_c, c.shape)
        avg_defocus = arr_max[0] + 500
        max_col = arr_max[1]

        return avg_defocus, max_col

    def background_subtract_2d(self, signal, background_p1, max_col):
        """
        Subtract background from estimated power spectrum

        :param signal: Estimated power spectrum
        :param background_p1: 1-D background estimation
        :param max_col: Internal variable, returned as the second parameter from opt1d.
        :return: 2-tuple of NumPy arrays (Estimated PSD without noise and estimated noise).
        """

        # RCOPT
        signal = signal.asnumpy().T
        # background_p1 is still np array in old ordering for now.

        N = signal.shape[0]
        center = N // 2
        [X, Y] = np.meshgrid(
            np.arange(0 - center, N - center, dtype=self.dtype),
            np.arange(0 - center, N - center, dtype=self.dtype),
        )
        radii = np.sqrt(X ** 2 + Y ** 2)

        background = np.zeros(signal.shape, dtype=self.dtype)
        background_p1 = background_p1[:, max_col]
        for r in range(background_p1.shape[0] - 1, 0, -1):
            background[radii <= r + 1] = background_p1[r]
        mask = radii <= max_col + 2
        background[mask] = signal[mask]

        signal = signal - background
        signal = np.maximum(0, signal)

        return Image(signal.T), Image(background.T)

    def PCA(self, signal, pixel_size, g_min, g_max):
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
        [X, Y] = np.meshgrid(
            np.arange(0 - center, N - center) / N, np.arange(0 - center, N - center) / N
        )

        rb = np.sqrt(np.square(X) + np.square(Y))
        r_ctf = rb * (10 / pixel_size)

        [X, Y] = np.meshgrid(np.arange(-center, center), np.arange(-center, center))

        signal -= np.min(signal)

        rad_sq_min = N * pixel_size / g_min
        rad_sq_max = N * pixel_size / g_max

        min_limit = r_ctf[center, (center + np.floor(rad_sq_min)).astype(int)]
        signal[r_ctf < min_limit] = 0

        max_limit = r_ctf[center, (center + np.ceil(rad_sq_max)).astype(int)]
        signal = np.where(r_ctf > max_limit, 0, signal)

        moment_02 = np.multiply(Y ** 2, signal)
        moment_02 = np.sum(moment_02, axis=(0, 1))

        moment_11 = np.multiply(np.multiply(Y, X), signal)
        moment_11 = np.sum(moment_11, axis=(0, 1))

        moment_20 = np.multiply(X ** 2, signal)
        moment_20 = np.sum(moment_20, axis=(0, 1))

        moment_mat = np.zeros((2, 2))
        moment_mat[0, 0] = moment_20
        moment_mat[1, 1] = moment_02
        moment_mat[0, 1] = moment_11
        moment_mat[1, 0] = moment_11

        moment_evals = npla.eigvalsh(moment_mat)
        ratio = np.divide(moment_evals[0], moment_evals[1])

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
        :param angle_ast: Angle between df1 and the x-axis
        :param r: Magnitude of spatial frequencies.
        :param theta: Phase of spatial frequencies.
        :param pixel_size: Pixel size in \u212b (Angstrom).
        :param g_min: Inverse of minimun resolution for PSD.
        :param g_max: Inverse of maximum resolution for PSD.
        :param amplitude_contrast: Amplitude contrast.
        :param lmbd: Electron wavelength.
        :param cs: Spherical aberration in mm.
        :return: Optimal defocus parameters
        """

        angle_ast = angle_ast / 180 * np.pi

        # step size
        alpha1 = np.power(10, 5)
        alpha2 = np.power(10, 4)

        # initialization
        x = df1 + df2
        y = (df1 - df2) * np.cos(2 * angle_ast)
        z = (df1 - df2) * np.sin(2 * angle_ast)

        a = np.pi * lmbd * np.power(r, 2) / 2
        b = np.pi * np.power(lmbd, 3) * cs * np.power(10, 6) * np.power(
            r, 4
        ) / 2 - amplitude_contrast * np.ones(r.shape)

        signal = signal.asnumpy()[0].T
        N = signal.shape[1]
        center = N // 2

        rad_sq_min = N * pixel_size / g_min
        rad_sq_max = N * pixel_size / g_max

        max_val = r[center, np.int(center - 1 + np.floor(rad_sq_max))]
        min_val = r[center, np.int(center - 1 + np.ceil(rad_sq_min))]

        mask = (r <= max_val) & (r > min_val)
        a = a[mask]
        b = b[mask]
        signal = signal[..., mask]
        r = r[mask]
        theta = theta[mask]

        sum_A = np.sum(np.power(signal, 2))

        dx = 1
        dy = 1
        dz = 1

        stop_cond = 10 ^ -20
        iter_no = 1

        # for iter_no in range(0, 399):
        while np.maximum(np.maximum(dx, dy), dz) > stop_cond:
            inner_cosine = np.multiply(y, np.cos(2 * theta)) + np.multiply(
                z, np.sin(2 * theta)
            )
            outer_sine = np.sin(np.multiply(a, x) + np.multiply(a, inner_cosine) - b)
            outer_cosine = np.cos(np.multiply(a, x) + np.multiply(a, inner_cosine) - b)

            sine_x_term = a
            sine_y_term = np.multiply(a, np.cos(2 * theta))
            sine_z_term = np.multiply(a, np.sin(2 * theta))

            c1 = np.sum(np.multiply(np.abs(outer_sine), signal))
            c2 = np.sqrt(np.multiply(sum_A, np.sum(np.power(outer_sine, 2))))

            # gradients of numerator
            dx_c1 = np.sum(
                np.multiply(
                    np.multiply(np.multiply(np.sign(outer_sine), outer_cosine), a),
                    signal,
                )
            )
            dy_c1 = np.sum(
                np.multiply(
                    np.multiply(
                        np.multiply(np.multiply(np.sign(outer_sine), outer_cosine), a),
                        np.cos(2 * theta),
                    ),
                    signal,
                )
            )
            dz_c1 = np.sum(
                np.multiply(
                    np.multiply(
                        np.multiply(np.multiply(np.sign(outer_sine), outer_cosine), a),
                        np.sin(2 * theta),
                    ),
                    signal,
                )
            )

            derivative_sqrt = np.divide(
                1, 2 * np.sqrt(np.multiply(sum_A, np.sum(np.power(outer_sine, 2))))
            )
            derivative_sine2 = 2 * np.multiply(outer_sine, outer_cosine)

            #  gradients of denomenator
            dx_c2 = np.multiply(
                derivative_sqrt,
                np.multiply(sum_A, np.sum(np.multiply(derivative_sine2, sine_x_term))),
            )
            dy_c2 = np.multiply(
                derivative_sqrt,
                np.multiply(sum_A, np.sum(np.multiply(derivative_sine2, sine_y_term))),
            )
            dz_c2 = np.multiply(
                derivative_sqrt,
                np.multiply(sum_A, np.sum(np.multiply(derivative_sine2, sine_z_term))),
            )

            # gradients
            dx = np.divide(
                np.multiply(dx_c1, c2) - np.multiply(dx_c2, c1), np.power(c2, 2)
            )
            dy = np.divide(
                np.multiply(dy_c1, c2) - np.multiply(dy_c2, c1), np.power(c2, 2)
            )
            dz = np.divide(
                np.multiply(dz_c1, c2) - np.multiply(dz_c2, c1), np.power(c2, 2)
            )

            # update
            x = x + np.multiply(alpha1, dx)
            y = y + np.multiply(alpha2, dy)
            z = z + np.multiply(alpha2, dz)

            if iter_no < 2:
                stop_cond = np.minimum(np.minimum(dx, dy), dz) / 1000

            if iter_no > 400:
                stop_cond = np.maximum(np.maximum(dx, dy), dz) + 1

            iter_no = iter_no + 1

        df1 = np.divide(x + np.abs(y + z * 1j), 2)
        df2 = np.divide(x - np.abs(y + z * 1j), 2)
        angle_ast = np.angle(y + z * 1j) / 2

        inner_cosine = np.multiply(y, np.cos(2 * theta)) + np.multiply(
            z, np.sin(2 * theta)
        )
        outer_sine = np.sin(np.multiply(a, x) + np.multiply(a, inner_cosine) - b)
        outer_cosine = np.cos(np.multiply(a, x) + np.multiply(a, inner_cosine) - b)

        sine_x_term = a
        sine_y_term = np.multiply(a, np.cos(2 * theta))
        sine_z_term = np.multiply(a, np.sin(2 * theta))

        c1 = np.sum(np.multiply(np.abs(outer_sine), signal))
        c2 = np.sqrt(np.multiply(sum_A, np.sum(np.power(outer_sine, 2))))

        p = np.divide(c1, c2)

        return df1, df2, angle_ast, p

    def locate_minima(self, signal):
        """
        Find zero-crossings of power spectrum.

        :param signal: Estimated power spectrum
        :return: array of zero-crossings indices.
        """

        N = signal.shape[0]

        derivatives = np.zeros((8, signal.shape[0], signal.shape[1]))
        derivatives[0, 1:, :] = signal[: N - 1, :] - signal
        derivatives[1, : N - 1, :] = signal[1:, :] - signal
        derivatives[2, :, 1:] = signal[:, 1:] - signal
        derivatives[3, :, : N - 1] = signal[:, 1:] - signal
        derivatives[4, 1:, 1:] = signal[: N - 1, : N - 1] - signal
        derivatives[5, 1:, : N - 1] = signal[: N - 1, 1:] - signal
        derivatives[6, : N - 1, 1:] = signal[1:, : N - 1] - signal
        derivatives[7, : N - 1, : N - 1] = signal[1:, 1:] - signal

        derivative_sign = np.where(derivatives >= 0, 1, 0)
        objective = np.sum(derivative_sign, axis=-1)
        zero_cross_map = np.where(objective >= 6, 1, 0)

        return zero_cross_map

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
