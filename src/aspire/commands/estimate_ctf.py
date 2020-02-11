import logging
import click
import os
import numpy as np
import mrcfile
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.ctf.aspire_ctf import CtfEstimator
import matplotlib.pyplot as plt
import scipy.io as sio

logger = logging.getLogger('aspire')


@click.command()
@click.option('--data_folder', default=None, help='Path to mrc or mrcs files')
@click.option('--pixel_size', default=1, type=float, help='Pixel size in A')
@click.option('--cs', default=2.0, type=float, help='Spherical aberration')
@click.option('--amplitude_contrast', default=0.07, type=float, help='Amplitude contrast')
@click.option('--voltage', default=300, type=float, help='Voltage in electron microscope')
@click.option('--num_tapers', default=2, type=int, help='Number of tapers to apply in PSD estimation')
@click.option('--psd_size', default=512, type=int, help='Size of blocks for use in PSD estimation')
@click.option('--g_min', default=30, type=float, help='Inverse of minimum resolution for PSD')
@click.option('--g_max', default=5, type=float, help='Inverse of maximum resolution for PSD')
@click.option('--corr', default=1, type=float, help='Select method')
@click.option('--output_dir', default="results", help='Path to output files')

def estimate_ctf(data_folder, pixel_size, cs, amplitude_contrast, voltage, num_tapers, psd_size, g_min, g_max, corr, output_dir):

    dir_content = os.scandir(data_folder)

    mrc_files = [f.name for f in dir_content if os.path.splitext(f)[1]=='.mrc']
    mrcs_files = [f.name for f in dir_content if os.path.splitext(f)[1]=='.mrcs']
    file_names = mrc_files + mrcs_files

    amp = amplitude_contrast
    amplitude_contrast = np.arctan(amplitude_contrast / np.sqrt(1 - np.square(amplitude_contrast)))
    lmbd = 1.22639 / np.sqrt(voltage * 1000 + 0.97845 * np.square(voltage))

    ctf_object = CtfEstimator(pixel_size, cs, amplitude_contrast, voltage, psd_size, num_tapers)

    ffbbasis = FFBBasis2D((psd_size, psd_size), 2)

    defocus = np.zeros((3, len(file_names)))

    atan_ampcon = np.arctan(amplitude_contrast)

    for name in file_names:
        with mrcfile.open(os.path.join(data_folder, name), mode='r+', permissive=True) as mrc:
            micrograph = mrc.data

        micrograph = micrograph.astype('float')

        micrograph = np.double(micrograph)

        micrograph = np.transpose(micrograph)

        micrograph_blocks = ctf_object.ctf_preprocess(micrograph, psd_size)

        tapers_1d = ctf_object.ctf_tapers(psd_size, (2 * num_tapers) / psd_size, num_tapers)

        signal_observed = ctf_object.ctf_estimate_psd(micrograph_blocks, tapers_1d, num_tapers)

        thon_rings, g_out = ctf_object.ctf_elliptical_average(ffbbasis, signal_observed, 0)  # absolute differenceL 10^-14. Relative error: 10^-7

        signal_1d, background_1d = ctf_object.ctf_background_subtract_1d(thon_rings)

        if corr:
            avg_defocus, low_freq_skip = ctf_object.ctf_opt1d(signal_1d, pixel_size, cs, lmbd, amplitude_contrast, signal_observed.shape[0])
            signal, background_2d = ctf_object.ctf_background_subtract_2d(signal_observed, background_1d, low_freq_skip)
            ratio = ctf_object.ctf_PCA(signal_observed, pixel_size, g_min, g_max, amplitude_contrast)
            signal, additional_background = ctf_object.ctf_elliptical_average(ffbbasis, np.sqrt(signal), 2)
        else:
            signal, background_2d = ctf_object.ctf_background_subtract_2d(signal_observed, background_1d, 12)
            ratio = ctf_object.ctf_PCA(signal_observed, pixel_size, g_min, g_max, amplitude_contrast)
            signal, additional_background = ctf_object.ctf_elliptical_average(ffbbasis, np.sqrt(signal), 2)
            zero_map = ctf_object.ctf_locate_minima(signal)

        if additional_background.shape[-1]==1:
            additional_background = np.squeeze(additional_background, -1)
        background_2d = background_2d + additional_background

        initial_df1 = (avg_defocus * 2) / (1 + ratio)
        initial_df2 = (avg_defocus * 2) - initial_df1

        center = psd_size // 2
        [X, Y] = np.meshgrid(np.arange(0 - center, psd_size - center) / psd_size, np.arange(0 - center, psd_size - center) / psd_size)

        rb = np.sqrt(np.square(X) + np.square(Y))
        r_ctf = rb * (10 / pixel_size)
        theta = np.arctan2(Y, X)

        signal = np.squeeze(signal, -1)
        # signal = np.divide(signal, 2*np.pi*r_ctf)

        angle = -75
        cc_array = np.zeros((6,4))
        for a in range(0,6):
            df1, df2, angle_ast, p = ctf_object.ctf_gd(signal, initial_df1, initial_df2, angle + np.multiply(a, 30), r_ctf, theta, pixel_size, g_min, g_max, amplitude_contrast, lmbd, cs)
            cc_array[a, 0] = df1
            cc_array[a, 1] = df2
            cc_array[a, 2] = angle_ast
            cc_array[a, 3] = p
        ml = np.argmax(cc_array[:,3],-1)
        ctf_object.write_star(cc_array[ml, 0], cc_array[ml, 1], cc_array[ml, 2], cs, voltage, pixel_size, amp, name)

        noise_image = np.subtract(signal_observed, signal)

        with mrcfile.new(output_dir + '/' +  os.path.splitext(name)[0] + '_noise.mrc', overwrite=True) as mrc:
            mrc.set_data(np.float32(background_2d))
            mrc.voxel_size = pixel_size
            mrc.close()

        df = (cc_array[ml,0] + cc_array[ml, 1])*np.ones(theta.shape, theta.dtype) + (cc_array[ml,0] - cc_array[ml, 1])*np.cos(2*theta - 2*cc_array[ml, 2]*np.ones(theta.shape, theta.dtype))
        ctf_im = - np.sin(np.pi*lmbd*np.power(r_ctf, 2)/2 * (df - np.power(lmbd, 2)*np.power(r_ctf, 2)*(cs*np.power(10,6))) + amplitude_contrast)
        ctf_signal = np.zeros(ctf_im.shape, ctf_im.dtype)
        ctf_signal[:ctf_im.shape[0]//2, :] = ctf_im[:ctf_im.shape[0]//2, :]
        ctf_signal[ctf_im.shape[0]//2+1:, :] = signal[ctf_im.shape[0]//2+1:, :]

        with mrcfile.new(output_dir + '/' +  os.path.splitext(name)[0] + '.ctf', overwrite=True) as mrc:
            mrc.set_data(np.float32(ctf_signal))
            mrc.voxel_size = pixel_size
            mrc.close()

