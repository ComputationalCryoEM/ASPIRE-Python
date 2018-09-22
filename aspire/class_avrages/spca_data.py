import numpy as np
import os
from numpy.polynomial.legendre import leggauss
import scipy.special as sp
from aspire.class_avrages import mat_to_npy, mat_to_npy_vec
from lib.nufft_cims import py_nufft


class SamplePoints:
    def __init__(self, x, w):
        self.x = x
        self.w = w


class Precomp:
    def __init__(self, n_theta, n_r, resolution, freqs):
        self.n_r = n_r
        self.n_theta = n_theta
        self.resolution = resolution
        self.freqs = freqs


class SpcaData:
    def __init__(self, eigval, freqs, radial_freqs, coeff, mean, bandlimit, supp_num, eig_im, fn0):
        self.eigval = eigval
        self.freqs = freqs
        self.radial_freqs = radial_freqs
        self.coeff = coeff
        self.mean = mean
        self.c = bandlimit
        self.r = supp_num
        self.eig_im = eig_im
        self.fn0 = fn0

    def save(self, obj_dir):
        np.save(obj_dir + '/eigval.npy', self.eigval)
        np.save(obj_dir + '/freqs.npy', self.freqs)
        np.save(obj_dir + '/radial_freqs.npy', self.radial_freqs)
        np.save(obj_dir + '/coeff.npy', self.coeff)
        np.save(obj_dir + '/mean.npy', self.mean)
        np.save(obj_dir + '/c.npy', np.array([self.c]))
        np.save(obj_dir + '/r.npy', np.array([self.r]))
        np.save(obj_dir + '/eig_im.npy', self.eig_im)
        np.save(obj_dir + '/fn0.npy', self.fn0)

    def load(self, obj_dir, matlab=False):
        if matlab:
            files = os.listdir(obj_dir)
            vec_obj = ['c', 'r', 'eigval', 'mean', 'radial_freqs', 'freqs']
            curr_dir = os.getcwd()
            target_dir = os.path.join(curr_dir, obj_dir)
            os.chdir(target_dir)
            for f in files:
                if f[-1] == 't':
                    name = f[:-4]
                    if name in vec_obj:
                        curr_obj = mat_to_npy_vec(name)
                    else:
                        curr_obj = mat_to_npy(name)
                    np.save(name, curr_obj)
            os.chdir(curr_dir)

        self.eigval = np.load(obj_dir + '/eigval.npy')
        self.freqs = np.load(obj_dir + '/freqs.npy')
        self.radial_freqs = np.load(obj_dir + '/radial_freqs.npy')
        self.coeff = np.load(obj_dir + '/coeff.npy')
        self.mean = np.load(obj_dir + '/mean.npy')
        self.c = np.load(obj_dir + '/c.npy')[0]
        self.r = np.load(obj_dir + '/r.npy')[0]
        self.eig_im = np.load(obj_dir + '/eig_im.npy')
        self.fn0 = np.load(obj_dir + '/fn0.npy')


def compute_spca(images, noise_v_r, adaptive_support=False):
    num_images = images.shape[2]
    resolution = images.shape[0]

    if adaptive_support:
        # TODO debug this
        energy_thresh = 0.99
        # Estimate bandlimit and compact support size
        [bandlimit, support_size] = choose_support_v6(cfft2(images), energy_thresh)
        bandlimit = bandlimit * (0.5 / np.floor(resolution / 2.0))  # Rescaling between 0 and 0.5
    else:
        bandlimit = 0.5
        support_size = int(np.floor(resolution / 2.0))

    n_r = int(np.ceil(4 * bandlimit * support_size))
    basis, sample_points = precompute_fb(n_r, support_size, bandlimit)
    _, coeff, mean_coeff, spca_coeff, u, d = jobscript_ffbspca(images, support_size, noise_v_r, basis, sample_points)

    ang_freqs = []
    rad_freqs = []
    vec_d = []
    for i in range(len(d)):
        if len(d[i]) != 0:
            ang_freqs.extend(np.ones(len(d[i]), dtype='int') * i)
            rad_freqs.extend(np.arange(len(d[i])) + 1)
            vec_d.extend(d[i])

    ang_freqs = np.array(ang_freqs)
    rad_freqs = np.array(rad_freqs)
    d = np.array(vec_d)
    k = min(len(d), 400)  # keep the top 400 components
    sorted_indices = np.argsort(-d)
    sorted_indices = sorted_indices[:k]
    d = d[sorted_indices]
    ang_freqs = ang_freqs[sorted_indices]
    rad_freqs = rad_freqs[sorted_indices]

    s_coeff = np.zeros((len(d), num_images), dtype='complex128')
    for i in range(len(d)):
        s_coeff[i] = spca_coeff[ang_freqs[i]][rad_freqs[i] - 1]

    fn = ift_fb(support_size, bandlimit)

    eig_im = np.zeros((np.square(2 * support_size), len(d)), dtype='complex128')
    # might be able to do this faster
    for i in range(len(d)):
        tmp = fn[ang_freqs[i]]
        tmp = tmp.reshape((int(np.square(2 * support_size)), tmp.shape[2]), order='F')
        eig_im[:, i] = np.dot(tmp, u[ang_freqs[i]][:, rad_freqs[i] - 1])

    fn0 = fn[0].reshape((int(np.square(2 * support_size)), fn[0].shape[2]), order='F')

    spca_data = SpcaData(d, ang_freqs, rad_freqs, s_coeff, mean_coeff, bandlimit, support_size, eig_im, fn0)
    return spca_data


def lgwt(n, a, b):
    """
    get leggauss points in interval [a, b]
    :param n: number of points
    :param a: interval starting point
    :param b: interval end point
    :return: SamplePoints.x: sample points
             SamplePoints.w: weights
    """

    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return SamplePoints(x, w)


def choose_support_v6(proj_ctf_noisy, energy_threshold):
    # Determine sizes of the compact support in both real and Fourier space.
    # OUTPUTS:
    # c_limit: Size of support in Fourier space
    # R_limit: Size of support in real space
    # We scale the images in real space by L, so that the noise variance in
    # both real and Fourier domains is the same.
    # Based on code by Tejal from Oct 2015

    L = proj_ctf_noisy.data.shape[1]
    N = int(np.floor(L / 2))
    P = proj_ctf_noisy.data.shape[0]
    x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    r = np.sqrt(np.square(x) + np.square(y))
    r_flat = r.flatten()
    r_max = N

    img_f = proj_ctf_noisy.data.astype(np.float64)
    img = (icfft2(img_f)) * L
    mean_data = np.mean(img, axis=0)  # Remove mean from the data
    img = img - mean_data

    # Compute the variance of the noise in two different way. See below for the reason.
    img_corner = np.reshape(img, (P, L * L))
    img_corner = img_corner[:, r_flat > r_max]
    img_corner = img_corner.flatten()
    var_img = np.var(img_corner, ddof=1)

    imgf_corner = np.reshape(img_f, (P, L * L))
    imgf_corner = imgf_corner[:, r_flat > r_max]
    imgf_corner = imgf_corner.flatten()
    var_imgf = np.var(imgf_corner, ddof=1)

    noise_var = np.min([var_img, var_imgf])  # Note, theoretical img_f and
    # img should give the same variance but there is a small difference,
    # choose the smaller one so that you don't get a negative variance or power
    # spectrum in 46,47

    variance_map = np.var(img, axis=0, ddof=1)
    variance_map = variance_map.transpose()

    # Mean 2D variance radial function
    radial_var = np.zeros(N)
    for i in xrange(N):
        radial_var[i] = np.mean(variance_map[np.logical_and(r >= i, r < i + 1)])

    img_ps = np.square(np.abs(img_f))
    pspec = np.mean(img_ps, 0)
    pspec = pspec.transpose()
    radial_pspec = np.zeros(N)

    # Compute the radial power spectrum
    for i in xrange(N):
        radial_pspec[i] = np.mean(pspec[np.logical_and(r >= i, r < i + 1)])

    # Subtract the noise variance
    radial_pspec = radial_pspec - noise_var
    radial_var = radial_var - noise_var

    # compute the cumulative variance and power spectrum.
    c = np.linspace(0, 0.5, N)
    R = np.arange(0, N)
    cum_pspec = np.zeros(N)
    cum_var = np.zeros(N)

    for i in xrange(N):
        cum_pspec[i] = np.sum(np.multiply(radial_pspec[0:i + 1], c[0:i + 1]))
        cum_var[i] = np.sum(np.multiply(radial_var[0:i + 1], R[0:i + 1]))

    cum_pspec = cum_pspec / cum_pspec[-1]
    cum_var = cum_var / cum_var[-1]

    cidx = np.where(cum_pspec > energy_threshold)
    c_limit = c[cidx[0][0] - 1] * L
    Ridx = np.where(cum_var > energy_threshold)
    R_limit = R[Ridx[0][0] - 1]

    return c_limit, R_limit


def precompute_fb(n_r, support_size, bandlimit):
    sample_points = lgwt(n_r, 0, bandlimit)
    basis = bessel_ns_radial(bandlimit, support_size, sample_points.x)
    return basis, sample_points


def bessel_ns_radial(bandlimit, support_size, x):
    bessel = np.load('bessel.npy')
    bessel = bessel[bessel[:, 3] <= 2 * np.pi * bandlimit * support_size, :]
    angular_freqs = bessel[:, 0]
    max_ang_freq = int(np.max(angular_freqs))
    n_theta = int(np.ceil(16 * bandlimit * support_size))
    if not n_theta % 2:
        n_theta += 1

    radian_freqs = bessel[:, 1]
    r_ns = bessel[:, 2]
    phi_ns = np.zeros((len(x), len(angular_freqs)))
    phi = {}

    for i in range(len(angular_freqs)):
        r0 = x * r_ns[i] / bandlimit
        f = sp.jv(angular_freqs[i], r0)
        # probably the square and the sqrt not needed
        tmp = np.pi * np.square(sp.jv(angular_freqs[i] + 1, r_ns[i]))
        phi_ns[:, i] = f / (bandlimit * np.sqrt(tmp))

    for i in range(max_ang_freq + 1):
        phi[i] = phi_ns[:, angular_freqs == i]

    return Basis(phi, angular_freqs, radian_freqs, n_theta)


def jobscript_ffbspca(images, support_size, noise_var, basis, sample_points, num_threads=10):
    split_images = np.array_split(images, num_threads, axis=2)
    del images

    coeff = fbcoeff_nfft(split_images, support_size, basis, sample_points, num_threads)
    u, d, spca_coeff, mean_coeff = spca_whole(coeff, noise_var)
    return 0, coeff, mean_coeff, spca_coeff, u, d


def fbcoeff_nfft(split_images, support_size, basis, sample_points, num_threads):
    image_size = split_images[0].shape[0]
    orig = int(np.floor(image_size / 2))
    new_image_size = int(2 * support_size)

    # unpacking input
    phi_ns = basis.phi_ns
    angular_freqs = basis.angular_freqs
    max_angular_freqs = int(np.max(angular_freqs))
    n_theta = basis.n_theta
    x = sample_points.x
    w = sample_points.w
    w = w * x

    # sampling points in the fourier domain
    freqs = pft_freqs(x, n_theta)
    precomp = Precomp(n_theta, len(x), new_image_size, freqs)
    scale = 2 * np.pi / n_theta

    coeff_pos_k = []
    pos_k = []

    for i in range(num_threads):
        curr_images = split_images[i]
        start_pixel = orig - support_size
        end_pixel = orig + support_size
        curr_images = curr_images[start_pixel:end_pixel, start_pixel:end_pixel, :]
        tmp = cryo_pft_nfft(curr_images, precomp)
        pf_f = scale * np.fft.fft(tmp, axis=1)
        pos_k.append(pf_f[:, :max_angular_freqs + 1, :])

    pos_k = np.concatenate(pos_k, axis=2)

    for i in range(max_angular_freqs + 1):
        coeff_pos_k.append(np.einsum('ki, k, kj -> ij', phi_ns[i], w, pos_k[:, i, :]))

    return coeff_pos_k


def pft_freqs(x, n_theta):
    n_r = len(x)
    d_theta = 2 * np.pi / n_theta

    # sampling points in the fourier domain
    freqs = np.zeros((n_r * n_theta, 2))
    for i in range(n_theta):
        freqs[i * n_r:(i + 1) * n_r, 0] = x * np.sin(i * d_theta)
        freqs[i * n_r:(i + 1) * n_r, 1] = x * np.cos(i * d_theta)

    return freqs


def cryo_pft_nfft(projections, precomp):
    freqs = precomp.freqs
    m = len(freqs)

    n_theta = precomp.n_theta
    n_r = precomp.n_r
    num_projections = projections.shape[2]
    pf = np.zeros((m, num_projections), dtype='complex128')

    nufft_obj = py_nufft.factory('nufft')

    x = -2 * np.pi * freqs.T

    for i in range(num_projections):
        # use bigger epsilon
        pf[:, i] = nufft_obj.forward2d(projections[:, :, i].T, x, iflag=-1, eps=1e-15)[0]

    pf = pf.reshape((n_r, n_theta, num_projections), order='F')

    return pf


def spca_whole(coeff, var_hat):
    max_ang_freq = len(coeff) - 1
    n_p = coeff[0].shape[1]
    u = []
    d = []
    spca_coeff = []
    mean_coeff = np.mean(coeff[0], axis=1)
    lr = len(coeff)

    for i in range(max_ang_freq + 1):
        tmp = coeff[i]
        if i == 0:
            tmp = (tmp.T - mean_coeff).T
            lambda_var = float(lr) / n_p
        else:
            lambda_var = float(lr) / (2 * n_p)

        c1 = np.real(np.einsum('ij, kj -> ik', tmp, np.conj(tmp))) / n_p
        curr_d, curr_u = np.linalg.eig(c1)
        sorted_indices = np.argsort(-curr_d)
        curr_u = fix_signs(curr_u)
        curr_d = curr_d[sorted_indices]
        curr_u = curr_u[:, sorted_indices]

        if var_hat != 0:
            k = len(np.where(curr_d > var_hat * np.square(1 + np.sqrt(lambda_var)))[0])
            if k != 0:
                curr_d = curr_d[:k]
                curr_u = curr_u[:, :k]
                d.append(curr_d)
                u.append(curr_u)
                l_k = 0.5 * ((curr_d - (lambda_var + 1) * var_hat) + np.sqrt(
                    np.square((lambda_var + 1) * var_hat - curr_d) - 4 * lambda_var * np.square(var_hat)))
                snr_i = l_k / var_hat
                snr = (np.square(snr_i) - lambda_var) / (snr_i + lambda_var)
                weight = 1 / (1 + 1 / snr)
                spca_coeff.append(np.einsum('i, ji, jk -> ik', weight, curr_u, tmp))

        else:
            u.append(curr_u)
            d.append(curr_d)
            spca_coeff.append(np.einsum('ji, jk -> ik', curr_u, tmp))

    return u, d, spca_coeff, mean_coeff


def ift_fb(support_size, bandlimit):
    support_size = int(support_size)
    x, y = np.meshgrid(np.arange(-support_size, support_size), np.arange(-support_size, support_size))
    r = np.sqrt(np.square(x) + np.square(y))
    inside_circle = r <= support_size
    theta = np.arctan2(x, y)
    theta = theta[inside_circle]
    r = r[inside_circle]

    bessel = np.load('bessel.npy')
    bessel = bessel[bessel[:, 3] <= 2 * np.pi * bandlimit * support_size, :]
    k_max = int(np.max(bessel[:, 0]))
    fn = []

    computation1 = 2 * np.pi * bandlimit * r
    computation2 = np.square(computation1)
    c_sqrt_pi_2 = 2 * bandlimit * np.sqrt(np.pi)
    bessel_freqs = bessel[:, 0]
    bessel2 = bessel[:, 2]
    for i in range(k_max + 1):
        bessel_k = bessel2[bessel_freqs == i]
        tmp = np.zeros((2 * support_size, 2 * support_size, len(bessel_k)), dtype='complex128')

        f_r_base = c_sqrt_pi_2 * np.power(-1j, i) * sp.jv(i, computation1)
        f_theta = np.exp(1j * i * theta)

        tmp[inside_circle, :] = np.outer(f_r_base * f_theta, np.power(-1, np.arange(1, len(bessel_k) + 1)) * bessel_k)\
            / np.subtract.outer(computation2, np.square(bessel_k))

        fn.append(np.transpose(tmp, axes=(1, 0, 2)))

    return fn