import numpy as np
import pickle
import finufftpy
from numpy.random import rand
from scipy.special import erfinv
from aspire.utils.data_utils import cfftn, icfftn, mat_to_npy, mat_to_npy_vec
from aspire.utils.array_utils import cfft2, icfft2
from pyfftw.interfaces import numpy_fft


def cryo_gen_projections(n, k, snr, max_shift=0, shift_step=0, ref_shifts=None, rots_ref=None, precision='single'):
    '''
     Simulate projections.
     Same functionality as gen_projections_v2, but uses the new simulation
     code.
     See test_cryo_gen_projections.m for an example.
    :param n:   Size of of each projection (nxn).
    :param k:   Number of projections to generate.
    :param snr: Signal-to-noise-ratio of each projection, defined as the
                variance of the clean projection divded by the variance of the
                noise.
    :param max_shift:   Maximal random shift (in pixels) introduced to each
                projection. max_shift must be an integer and the resulting
                random shifts are integers between -max_shift to +max_shift.
                Default is 0 (no shift). Pass [] is ref_shifts are given.
    :param shift_step:  Resolution used to generate shifts. step_size=1 allows
                for all integer shifts from -max_shift to max_shift (in both x
                and y directions). step_size=2 generates shifts between
                -max_shift and max_shift with steps of 2 pixels, that is,
                shifts of 0 pixels, 2 pixels, 4 pixels, and so on. Default
                is 0. Pass [] is ref_shifts are given.
    :param ref_shifts:  A two column table with the 2D shift of each projection.
                Number of rows must be equal to the number of proejctions. If
                this parameter is provided, max_shift and shift_step are
                ignored.
    :param rots_ref:    An array of rotation matrices used to generate the projections.
                The size of the third dimension must equal the number of
                projections. If this parameter is missing, the functions draws
                rotation matrices uniformly at random.
    :param precision:   Accuracy of the projections. 'single' or 'double'. Default
                is 'single' (faster)
    :return:
    projections: Clean (shifted) simulated projections.
    noisy_projections:   Noisy (shifted) simulated projections.
    shifts:     A two column table with the 2D shift introduced to
                each projections. If ref_shift is given, this is equal to
                ref_shifts, otherwise, the shifts are random.
    rots:       A 3-by-3-by-K array of rotation matrices used to generate each of
                the projections. If rots_ref is given, this is equal to that array.
                Otherwise, the rotations re random and uniformly distributed on
                SO(3).
    '''
    if rots_ref is not None:
        rots = rots_ref
    else:
        rots = rand_rots(k)
    shifts = None
    if ref_shifts is not None:
        shifts = ref_shifts

    volref = pickle.load(open('cleanrib.p', 'rb'))
    projections = cryo_project(volref, rots, n, precision)

    # Swap dimensions for compitability with old gen_projections.
    projections = projections.transpose((1, 0, 2))

    # Add shifts
    if shifts is not None:
        # Adding user-provided shifts to projections
        [projections, shifts] = cryo_addshifts(projections, shifts)
    else:
        # Adding randomly-generated shifts to projections
        [projections, shifts] = cryo_addshifts(projections, None, max_shift, shift_step)

    # Add noise
    [noisy_projections, noise, i, sigma] = cryo_addnoise(projections, snr, 'gaussian')

    return [projections, noisy_projections, shifts, rots]


def cryo_project(volume, rot, n=None, precision='single', batch_size=100):
    """
    Project the given volume in a direction given by the rotations rot.
    :param volume: 3D array of size nxnxn of thevolume to project.
    :param rot: Array of size 3-by-3-by-K containing the projection directions
           of each projected image.
    :param n:
    :param precision: Accuracy of the projection. 'single' or 'double'. Default is single (faster).
    :param batch_size: Process the images in batches of batch_size. Large batch
           size is faster but requires more memory. Default is 100.
    :return: projections: 3D stack of size of projections. Each slice
             projections(:,:,k) is a projection image of size nxn which
             corresponds to the rotation rot(:,:,k). The number of
             projections in the stack is equal to the number of
             quaternions.

     The function supports creating projections whose size is different from
     the size of the proejcted volume. However, this code is still
     experimental and was not tested to verify that the resulting projections
     match the analytical ones to double precision.

     Note:
     To make the output of this function compatible with cryo_gen_projections
     call
       projections=permute(projections,[2 1 3]);
     The FIRM reconstruction functions rely upon this permuted order. See
     'examples' directory for detailed examples.

     Example:
         voldef='C1_params';
         rot = rand_rots(1);
         n=129;
         rmax=1;
         vol=cryo_gaussian_phantom_3d(n,rmax,voldef);
         p=cryo_project(vol,rot);
         imagesc(p);

     Yoel Shkolnisky, February 2018.
    """

    if precision == 'single':
        dtype = 'complex64'
    elif precision == 'double':
        dtype = 'complex128'
    else:
        raise Exception("the precision input is not 'single' or 'double'")

    precision_eps = np.finfo(precision).eps
    imagtol = precision_eps * 5

    # The function zeros gets as an argument 'single' or 'double'. We want to
    # use 'single' when possible to save space. presicion_str is the
    # appropriate string to pass to the functions 'zeros' later on.
    if n is None:
        n = volume.shape(0)

    if np.mod(n, 2) == 1:
        n_range = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1)
    else:
        n_range = np.arange(-n / 2 + 1 / 2, n / 2 - 1 / 2 + 1)

    [i, j] = np.meshgrid(n_range, n_range)
    i = i.ravel()
    j = j.ravel()

    rn = np.size(n_range)
    nv = np.size(volume, 1)
    if rn > nv + 1:
        if np.mod(rn - nv, 2) == 1:
            raise Exception('Upsampling from odd to even sizes or vice versa is currently not supported')
        dn = np.math.floor((rn - nv) / 2)
        fv = cfftn(volume)
        padded_volume = np.zeros((rn, rn, rn), dtype=dtype)
        padded_volume[dn:dn + nv, dn:dn + nv, dn:dn + nv] = fv
        volume = icfftn(padded_volume)
        assert (np.linalg.norm(np.imag(volume.flatten(order='F'))) / np.linalg.norm(volume.flatten(order='F')) < 1.0e-5)
        nv = rn
    k = np.size(rot, 2)
    batch_size = min(batch_size, k)
    projection_batches = np.zeros((rn, rn, batch_size, np.math.ceil(k / batch_size)), dtype=dtype)

    # Each batch of images is processed and stored independently. All batches
    # will be merged below into a single output array.
    # Verify that we have only small imaginary components in the
    # projcetions. How 'small' dependes on the accuracy.
    for batch in range(0, np.math.ceil(k / batch_size)):
        # It may be that the number of remained images is less than batch_size.
        # So compute the actual_batch_size.
        actual_batch_size = min(batch_size, k - batch * batch_size)
        # Sampling points in Fourier domain for all images of the current batch.
        p = np.zeros((np.size(i) * actual_batch_size, 3))
        startidx = (batch) * batch_size

        for ind in range(0, actual_batch_size):
            r = rot[:, :, startidx + ind]
            rt = r.T
            # n_x, n_y, n_z are the image of the unit vectors in the x, y, z
            # directions under the inverse rotation
            n_x = rt[:, 0]
            n_y = rt[:, 1]
            # n_z = rt[:, 2] #not used - just for completeness

            p[ind * np.size(i): (ind + 1) * np.size(i), :] = (i * np.array([n_x]).T + j * np.array([n_y]).T).T

        p = -2 * np.pi * p / nv

        # NUFFT all images in the current batch
        # projection_fourier = nufft3(volume, -p.T, nufft_opt)
        projection_fourier = nufft3(volume, -p.T)
        projection_fourier = projection_fourier.reshape((np.size(i), actual_batch_size), order='F')
        p = p.reshape([np.size(i), actual_batch_size, 3], order='F')

        if np.mod(n, 2) == 0:
            projection_fourier = projection_fourier * np.exp(1j * p.sum(axis=2) / 2)
            i_rep = np.tile(i, (actual_batch_size, 1)).T
            j_rep = np.tile(j, (actual_batch_size, 1)).T
            projection_fourier = projection_fourier * np.exp(2 * np.pi * 1j * (i_rep + j_rep - 1) / (2 * n))

        projections_temp = np.zeros((n, n, batch_size), dtype=dtype)
        for ind in range(0, actual_batch_size):
            temp = (projection_fourier[:, ind]).reshape((n, n), order='F')
            temp = numpy_fft.ifftshift(temp)
            projection = numpy_fft.fftshift(numpy_fft.ifft2(temp))

            if np.mod(n, 2) == 0:
                projection = projection * np.exp(2 * np.pi * 1j * (i + j) / (2 * n)).reshape((n, n), order='F')

            if np.linalg.norm(projection.flatten(order='F').imag) / np.linalg.norm(projection.flatten(order='F')) > imagtol:
                raise Exception('GCAR:imaginaryComponents', 'projection has imaginary components');
            projection = projection.real
            projections_temp[:, :, ind] = projection

        projection_batches[:, :, :, batch] = projections_temp

    # Merge projection_batches into a single output array
    projections = np.zeros((n, n, k), dtype=dtype)
    for batch in range(0, np.math.ceil(k / batch_size)):
        actual_batch_size = min(batch_size, k - batch * batch_size)
        startidx = batch * batch_size
        projections[:, :, startidx: startidx + actual_batch_size] = projection_batches[:, :, 0:actual_batch_size, batch]

    projections = projections.astype('float64')
    return projections


def randn2(*args, **kwargs):
    '''
    Calls rand and applies inverse transform sampling to the output.
    '''
    uniform = rand(*args, **kwargs)
    return np.sqrt(2) * erfinv(2 * uniform - 1)


def rand_rots(n):
    '''
    RAND_ROTS Generate random rotations
    Usage:      rot_matrices = rand_rots(n);
    :param n:   The number of rotations to generate.
    :return:    An array of size 3-by-3-by-n containing n rotation matrices
                sampled from the unifoorm distribution on SO(3).
    Note:
    This function depends on the random state of the `randn` function, so to
    obtain consistent outputs, its state must be controlled prior to calling
    using the `randn('state', s)` command.

    '''
    qs = qrand(n)
    return q_to_rot(qs)


def qrand(k):
    '''
    Generate K random uniformly distributed quaternions.
    Each quaternions is a four-elements column vector. Returns a matrix of
    size 4xK.

    The 3-sphere S^3 in R^4 is a double cover of the rotation group SO(3),
    SO(3) = RP^3.
    We identify unit norm quaternions a^2+b^2+c^2+d^2=1 with group elements.
    The antipodal points (-a,-b,-c,-d) and (a,b,c,d) are identified as the
    same group elements, so we take a>=0.
    '''
    q = (randn2(k, 4)).T
    l2_norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    for i in range(0, 4):
        q[i] = q[i] / l2_norm
    for j in range(0, k):
        if q[0, j] < 0:
            q[:, j] = -q[:, j]
    return q


def q_to_rot(q):
    '''
    Convert a quaternion into a rotation matrix.
    :param q:   quaternion. May be a vector of dimensions 4 x n
    :return:    3x3xn rotation matrix
    '''
    n = q.shape[1]
    rot_matrix = np.zeros((3, 3, n))

    rot_matrix[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    rot_matrix[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    rot_matrix[0, 2] = 2 * q[0] * q[2] + 2 * q[1] * q[3]

    rot_matrix[1, 0] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    rot_matrix[1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    rot_matrix[1, 2] = -2 * q[0] * q[1] + 2 * q[2] * q[3]

    rot_matrix[2, 0] = -2 * q[0] * q[2] + 2 * q[1] * q[3]
    rot_matrix[2, 1] = 2 * q[0] * q[1] + 2 * q[2] * q[3]
    rot_matrix[2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    return rot_matrix


def nufft3(vol, freqs):
    freqs = np.mod(freqs + np.pi, 2 * np.pi) - np.pi
    out = np.empty(freqs.shape[1], dtype='complex128', order='C')

    finufftpy.nufft3d2(freqs[0].copy(), freqs[1].copy(), freqs[2].copy(), out, -1, 1e-15, vol)
    return out


def cryo_addshifts(projections, shifts=None, max_shift=0, shift_step=0):
    '''
    Add shifts to projection images.
    :param projections: 3D stack of projections. The slice projections(:,:,k)
                is the k'th projections.
    :param shifts:(Optional) A two column table with the 2D shift of each
                projection. Number of rows must be equal to the number
                of proejctions. If this parameter is not provided, pass
                an empty array. If provided, the following to
                parameters can be omitted.
    :param max_shift: (Optional) Maximal random shift (in pixels) introduced
               to each projection. max_shift must be an integer and
               the resulting random shiftsare integers between
               -max_shift to +max_shift. Default max_shift is 0 (no
               shift).
    :param shift_step: (Optional) Resolution used to generate shifts.
               shift_step=1 allows for all integer shifts from
               -max_shift to max_shift (in both x and y directions).
               shift_step=2 generates shifts between -max_shift and
               max_shift with steps of 2 pixels, that is, shifts of 0
               pixels, 2 pixels, 4 pixels, and so on. Default
               shift_step is 0.
    :return: shifted_projections:  Shifted stack of images of the same size as the projections input array.
             ref_shifts: A two column table with the 2D shift introduced to each projections.
    Yoel Shkolnisky, September 2013.
    '''
    k = np.size(projections, 2)
    if shifts is not None:
        if np.size(shifts, 0) is not k:
            raise Exception('malformed "shifts". Must be of the same length as "projections"')
        ref_shifts = shifts
    else:
        ref_shifts = np.zeros((k, 2))
        if shift_step > 0:
            ref_shifts = np.round((rand(2, k).T - 1 / 2) * 2 * max_shift / shift_step) * shift_step

    nx = np.size(projections, 0)
    ny = np.size(projections, 1)

    if np.mod(nx, 2) == 1:
        range_x = np.arange(-(nx - 1) / 2, ((nx - 1) / 2) + 1)
    else:
        # Note that in the case of an even image, the center is not at the middle.
        # This can be easily fixed by switching to the appropriate FFT routines.
        range_x = np.arange(-nx/2, nx/2)

    if np.mod(ny, 2) == 1:
        range_y = np.arange(-(ny - 1) / 2, ((ny - 1) / 2) + 1)
    else:
        range_y = np.arange(-ny/2, ny/2)

    [omega_x, omega_y] = np.meshgrid(range_x, range_y)

    omega_x = -2 * np.pi * omega_x.T / nx
    omega_y = -2 * np.pi * omega_y.T / ny
    shifted_projections = np.zeros(np.shape(projections))

    for i in range(0, k):
        p = projections[:, :, i]
        pf = numpy_fft.fftshift(numpy_fft.fft2(numpy_fft.ifftshift(p)))
        phase_x = omega_x * ref_shifts[i, 0]
        phase_y = omega_y * ref_shifts[i, 1]
        pf = pf * np.exp(1j * (phase_x + phase_y))
        p2 = numpy_fft.fftshift(numpy_fft.ifft2(numpy_fft.ifftshift(pf)))
        shifted_projections[:, :, i] = np.real(p2)

    return [shifted_projections, ref_shifts]


def cryo_addnoise(projections, snr, noise_type, seed=None):
    '''
     Add additive noise to projection images.
    :param projections: 3D stack of projections. The slice projections(:,:,k)
                    is the k'th projections.
    :param snr: Signal to noise of the output noisy projections.
    :param noise_type: 'color' or 'gaussian'
    :param seed: Seed parameter for initializing the the random number
                    generator of the noise samples. If not provided, the
                    current state of the random number generator is used.
                    In such a case the results of the function are not
                    reproducible.
    :return:
    noisy_projections:  Stack of noisy projections. Same size as the input
                    projections.
    noise: Stack containing the additive noise added to each
                    projection. Same size as input projections.
    I: Normalized power spectrum of the noise. Has
                    dimension equal to a single projection.
    sigma: The standard deviation of Gaussian noise resulting
                    in the required SNR. Computed using the first
                    projection in the stack.
    '''
    p = np.size(projections, 0)
    k = np.size(projections, 2)
    noisy_projections = np.zeros(np.shape(projections))
    noise = np.zeros(np.shape(projections))
    sigma = np.sqrt(np.var(np.reshape(projections[:, :, 0], p ** 2, 1)) / snr)

    if seed is not None:
        np.random.seed(seed)

    if np.mod(p, 2) == 1:
        lowidx = -(p - 1) // 2 + p
        highidx = (p - 1) // 2 + p + 1
    else:
        lowidx = -p // 2 + p
        highidx = p // 2 + p

    # Color Noise Response
    i = cart2rad(2 * p + 1)
    i1 = np.ones(np.shape(i))
    i = 1 / np.sqrt((1 + i ** 2))
    i = 0 * i1 + 1 * i  # TODO: ask yoel
    i = i / np.linalg.norm(i.flatten(order='F'))
    noise_response = np.sqrt(i)

    for j in range(0, k):
        np.random.seed(1137)
        gn = randn2(2 * p + 1, 2 * p + 1).T
        if noise_type == 'gaussian':
            cn = gn
        else:
            cn = np.real(icfft2(cfft2(gn) * noise_response))

        cn = cn[lowidx:highidx, lowidx: highidx]
        cn = cn / np.std(cn.flatten(order='F'))
        cn = cn * sigma
        noisy_projections[:, :, j] = projections[:, :, j] + cn
        noise[:, :, j] = cn

    return [noisy_projections, noise, i, sigma]


def cart2rad(n):
    n = np.math.floor(n)
    p = (n - 1) / 2
    [x, y] = np.meshgrid(np.arange(-p, p+1), np.arange(-p, p+1))
    return np.sqrt(x ** 2 + y ** 2)

