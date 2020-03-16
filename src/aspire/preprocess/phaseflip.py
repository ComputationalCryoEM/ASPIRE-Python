import os
import numpy as np
from aspire.aspire.utils.parse_star import read_star
from pyfftw.interfaces.numpy_fft import fft2, ifft2, rfft2, irfft2
from scipy.io import loadmat
import mrcfile
from box import Box
# from aspire.aspire.utils.common import create_struct
import time


def phaseflip_star_file(star_file, pixel_size=None, return_in_fourier=False):
    """

    :param star_file:
    :param pixel_size:
    :param return_in_fourier: To save computation can skip the ifft.
    :return:
    """
    # star is a list of star lines describing projections
    star_records = read_star(star_file)['__root__']
    dir_path = os.path.dirname(star_file)
    stack_info = organize_star_records(star_records)

    # Initializing projections
    stack_name = star_records[0].rlnImageName.split('@')[1]
    mrc_path = os.path.join(os.path.dirname(star_file), stack_name)
    stack = load_stack_from_file(mrc_path)
    resolution = stack.shape[1]
    num_projections = len(star_records)
    # rfft_resolution = resolution // 2 + 1
    # imhat_stack = np.zeros((num_projections, resolution, rfft_resolution), dtype='complex64')
    # Todo - add temporary dir
    if return_in_fourier:
        projections = np.memmap('tmp_phaseflipped_projections.dat', dtype='complex64', mode='w+',
                                shape=(num_projections, resolution, resolution))
    else:
        projections = np.memmap('tmp_phaseflipped_projections.dat', dtype='float32', mode='w+',
                                shape=(num_projections, resolution, resolution))

    # Initializing pixel_size
    if pixel_size is None:
        tmp = Box(cryo_parse_Relion_CTF_struct(star_records[0]))
        if tmp.tmppixA != -1:
            pixel_size = tmp.tmppixA
        else:
            raise ValueError("Pixel size not provided and does not appear in STAR file")

    # Initializing parameter for cryo_CTF_Relion_fast
    # a, b, c = precompute_cryo_CTF_Relion_fast(resolution, r=True)
    a, b, c = precompute_cryo_CTF_Relion_fast(resolution, r=False)

    num_finished = 0
    for stack_name in stack_info:
        mrc_path = os.path.join(dir_path, stack_name)
        stack = load_stack_from_file(mrc_path)
        pos_in_stack = stack_info[stack_name].pos_in_stack
        pos_in_records = stack_info[stack_name].pos_in_records
        tic = time.time()
        for i, j in zip(pos_in_stack, pos_in_records):
            curr_image = stack[i]
            curr_records = Box(cryo_parse_Relion_CTF_struct(star_records[j]))
            curr_records.pixel_size = pixel_size

            # reference code
            # h = cryo_CTF_Relion(resolution, curr_records)
            # imhat = np.fft.fftshift(fft2(curr_image))
            # imhat *= np.sign(h)
            # pfim = ifft2(np.fft.ifftshift(imhat))

            # Instead of shifting and shifting back im, shift h (when computing h it is already shifted).
            h = cryo_CTF_Relion_fast(a, b, c, curr_records)
            imhat = fft2(curr_image)
            if return_in_fourier:
                np.multiply(imhat, np.sign(h), out=projections[j])
            else:
                imhat *= np.sign(h)
                projections[j] = ifft2(imhat)

            # Use real fft instead.
            # h = cryo_CTF_Relion_fast(a, b, c, curr_records)
            # np.multiply(imhat, np.sign(h), out=imhat_stack[j])  # can irfft2 back the whole stack
            # imhat2 *= np.sign(h)
            # pfim2 = irfft2(imhat2)
            num_finished += 1
        toc = time.time()
        print('Finished {} images in {} seconds. In total finished {}/{}'.format(
            len(pos_in_stack), toc - tic, num_finished, len(star_records)))
    return projections


def cryo_parse_Relion_CTF_struct(star_record):
    voltage = star_record.rlnVoltage
    DefocusU = star_record.rlnDefocusU/10  # Relion uses Angstrom. Convert to nm

    if hasattr(star_record, 'rlnDefocusV'):
        DefocusV = star_record.rlnDefocusV/10  # Relion uses Angstrom. Convert to nm
    else:
        DefocusV = DefocusU

    if hasattr(star_record, 'rlnDefocusAngle'):
        DefocusAngle = star_record.rlnDefocusAngle * np.pi/180  # convert to radians
    else:
        DefocusAngle = 0

    spherical_aberration = star_record.rlnSphericalAberration  # in mm, no conversion is needed
    pixel_size = None

    if hasattr(star_record, 'rlnDetectorPixelSize'):
        pixel_size = star_record.rlnDetectorPixelSize  # in Microns
        mag = star_record.rlnMagnification
        pixel_size = pixel_size * 10**4 / mag  # in Angstrem
    elif hasattr(star_record, 'pixA'):
        pixel_size = star_record.pixA

    return {
        'amplitude_contrast': star_record.rlnAmplitudeContrast,
        'voltage': voltage,
        'DefocusU': DefocusU,
        'DefocusV': DefocusV,
        'DefocusAngle': DefocusAngle,
        'spherical_aberration': spherical_aberration,
        'tmppixA': pixel_size,
        'pixel_size': pixel_size,
        }


def cryo_CTF_Relion(square_side, star_record):
    """
        Compute the contrast transfer function corresponding an n x n image with
        the sampling interval DetectorPixelSize.

    """
    #  wavelength in nm
    wave_length = 1.22639 / np.sqrt(star_record.voltage * 1000 + 0.97845 * star_record.voltage**2)

    # Divide by 10 to make pixel size in nm. BW is the bandwidth of
    #  the signal corresponding to the given pixel size
    bw = 1 / (star_record.pixel_size / 10)

    s, theta = radius_norm(square_side, origin=fctr(square_side))

    # RadiusNorm returns radii such that when multiplied by the
    #  bandwidth of the signal, we get the correct radial frequnecies
    #  corresponding to each pixel in our nxn grid.
    s *= bw

    DFavg = (star_record.DefocusU + star_record.DefocusV) / 2
    DFdiff = (star_record.DefocusU - star_record.DefocusV)
    df = DFavg + DFdiff * np.cos(2 * (theta - star_record.DefocusAngle)) / 2
    k2 = np.pi * wave_length * df
    # 10**6 converts spherical_aberration from mm to nm
    k4 = np.pi / 2*10**6 * star_record.spherical_aberration * wave_length**3
    chi = k4 * s**4 - k2 * s**2

    return np.sqrt(1 - star_record.amplitude_contrast ** 2) * np.sin(chi) - star_record.amplitude_contrast * np.cos(chi)


def cryo_CTF_Relion_fast(shifted_cut_s_squared, shifted_cut_s_fourth_power, shifted_cut_theta, star_record):
    """
    A faster version of cryo_CTF_Relion that assumes we already computed s and theta, shifted them and cut it to be of
    size Nx(N // 2 + 1). s is also squared and raised to the fourth power to save some computations.
    :param shifted_cut_s_squared:
    :param shifted_cut_s_fourth_power:
    :param shifted_cut_theta:
    :param star_record:
    :return:
    """
    wave_length = 1.22639 / np.sqrt(star_record.voltage * 1000 + 0.97845 * star_record.voltage**2)

    # Divide by 10 to make pixel size in nm. BW is the bandwidth of
    #  the signal corresponding to the given pixel size
    bw = 1 / (star_record.pixel_size / 10)

    # RadiusNorm returns radii such that when multiplied by the
    #  bandwidth of the signal, we get the correct radial frequnecies
    #  corresponding to each pixel in our nxn grid.

    DFavg = (star_record.DefocusU + star_record.DefocusV) / 2
    DFdiff = (star_record.DefocusU - star_record.DefocusV)
    df = DFavg + DFdiff * np.cos(2 * (shifted_cut_theta - star_record.DefocusAngle)) / 2
    k2 = np.pi * wave_length * df
    # 10**6 converts spherical_aberration from mm to nm
    k4 = np.pi / 2*10**6 * star_record.spherical_aberration * wave_length**3
    chi = k4 * bw ** 4 * shifted_cut_s_fourth_power - k2 * bw ** 2 * shifted_cut_s_squared

    return np.sqrt(1 - star_record.amplitude_contrast ** 2) * np.sin(chi) - star_record.amplitude_contrast * np.cos(chi)


def precompute_cryo_CTF_Relion_fast(square_side, r=True):
    s, theta = radius_norm(square_side, origin=fctr(square_side))
    s, theta = np.fft.fftshift(s), np.fft.fftshift(theta)
    if r:
        rfft_side = square_side // 2 + 1
        s, theta = s[:, :rfft_side], theta[:, :rfft_side]
    a = s ** 2
    b = s ** 4
    c = theta.copy()
    return a, b, c


def radius_norm(n: int, origin=None):
    """
        Create an n(1) x n(2) array where the value at (x,y) is the distance from the
        origin, normalized such that a distance equal to the width or height of
        the array = 1.  This is the appropriate function to define frequencies
        for the fft of a rectangular image.

        For a square array of size n (or [n n]) the following is true:
        RadiusNorm(n) = Radius(n)/n.
        The org argument is optional, in which case the FFT center is used.

        Theta is the angle in radians.

        (Transalted from Matlab RadiusNorm.m)
    """

    if isinstance(n, int):
        n = np.array([n, n])

    if origin is None:
        origin = np.ceil((n + 1) / 2)

    a, b = origin[0], origin[1]
    y, x = np.meshgrid(np.arange(1-a, n[0]-a+1)/n[0],
                       np.arange(1-b, n[1]-b+1)/n[1])  # zero at x,y
    radius = np.sqrt(x ** 2 + y ** 2)

    theta = np.arctan2(x, y)

    return radius, theta


def load_stack_from_file(filepath, c_contiguous=True, return_format=None):
    """ Load projection-stack from file. Try different formats.
        Supported formats are MRC/MRCS/MAT/NPY. """

    # try MRC/MRCS
    try:
        stack = mrcfile.open(filepath).data
        if c_contiguous:
            stack = fortran_to_c(stack)
        if return_format:
            return stack, 'mrc'
        return stack
    except ValueError:
        pass

    # try NPY format
    try:
        stack = np.load(filepath)
        if not isinstance(stack, np.ndarray):
            raise ValueError(f"File {filepath} doesn't contain a stack!")

        if c_contiguous:
            stack = fortran_to_c(stack)

        if return_format:
            return stack, 'npy'
        return stack

    except OSError:
        pass

    # try MAT format
    try:
        content = loadmat(filepath)
        # filter actual data
        data = [content[key] for key in content.keys() if key == key.strip('_')]
        if len(data) == 1 and hasattr(data[0], 'shape'):
            stack = data[0]

            if c_contiguous:
                stack = fortran_to_c(stack)

            if return_format:
                return stack, 'mat'
            return stack
        raise ValueError(f"MAT file {filepath} doesn't contain a stack!")
    except ValueError:
        pass

    raise NotImplementedError(f"Couldn't determine stack format! {filepath}!")


def fortran_to_c(stack):
    """ Convert Fortran-contiguous array to C-contiguous array. """
    return stack.T if stack.flags.f_contiguous else stack


def fctr(n):
    """ Center of an FFT-shifted image. We use this center
        coordinate for all rotations and centering operations. """

    if isinstance(n, int):
        n = np.array([n, n])

    return np.ceil((n + 1) / 2)


def organize_star_records(star_records):
    stacks_info = {}
    for i, rec in enumerate(star_records):
        pos, path = rec.rlnImageName.split('@')
        pos = int(pos) - 1
        if path in stacks_info.keys():
            stack_struct = stacks_info[path]
            stack_struct.pos_in_stack.append(pos)
            stack_struct.pos_in_records.append(i)
        else:
            stack_struct = create_struct({'pos_in_stack': [pos], 'pos_in_records': [i]})
            stacks_info[path] = stack_struct

    for path in stacks_info:
        stack_struct = stacks_info[path]
        stack_struct.pos_in_stack = np.array(stack_struct.pos_in_stack)
        stack_struct.pos_in_records = np.array(stack_struct.pos_in_records)
    return stacks_info


def fill_struct(obj=None, att_vals=None, overwrite=None):
    """
    Fill object with attributes in a dictionary.
    If a struct is not given a new object will be created and filled.
    If the given struct has a field in att_vals, the original field will stay, unless specified otherwise in overwrite.
    att_vals is a dictionary with string keys, and for each key:
    if hasattr(s, key) and key in overwrite:
        pass
    else:
        setattr(s, key, att_vals[key])
    :param obj:
    :param att_vals:
    :param overwrite
    :return:
    """
    # TODO should consider making copy option - i.e that the input won't change
    if obj is None:
        class DisposableObject:
            pass

        obj = DisposableObject()

    if att_vals is None:
        return obj

    if overwrite is None or not overwrite:
        overwrite = []
    if overwrite is True:
        overwrite = list(att_vals.keys())

    for key in att_vals.keys():
        if hasattr(obj, key) and key not in overwrite:
            continue
        else:
            setattr(obj, key, att_vals[key])

    return obj


def create_struct(att_vals=None):
    """
    Creates object
    :param att_vals:
    :return:
    """
    return fill_struct(att_vals=att_vals)
