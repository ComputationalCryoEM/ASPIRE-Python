import numpy as np
from ConverterModel.Converter import Converter
import matplotlib.pyplot as plt


def example():
    # file_name = 'aaa.npy'
    file_name = 'example_data_np_array.npy'
    beta = 0.5
    truncation_param = 10
    remove_mean = 1
    num_outliers = 0

    images = np.load(file_name)
    projections_denoise = denoise(images, beta, truncation_param, num_outliers, remove_mean)[0]

    plot_original_denoised(projections_denoise[:, :, 0], projections_denoise[:, :, 1])
    return 0


def plot_original_denoised(im_original, im_denoised):
    plt.subplot(1, 3, 1)
    plt.imshow(np.real(im_original), cmap='gray')
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(np.real(im_denoised), cmap='gray')
    plt.title('Denoised')

    plt.subplot(1, 3, 3)
    plt.imshow(np.real(im_original - im_denoised), cmap='gray')
    plt.title('Noise')
    plt.show()


def denoise(images, beta, truncation_param, num_outliers, remove_mean):

    # grids and sizes
    size_x, size_y, num_images = images.shape
    im_size = min(size_x, size_y)
    resolution = int(np.floor(im_size / 2))

    x_1d_grid = np.arange(-resolution, im_size - resolution)
    x, y = np.meshgrid(x_1d_grid, x_1d_grid)
    r = np.sqrt(np.square(x) + np.square(y))
    r_max = resolution
    r_smaller_r_max = r <= r_max

    # data handling
    projections = images[:im_size, :im_size, :]
    s_var_vec = np.var(projections[r_smaller_r_max, :], axis=0, ddof=1)
    n_var_vec = np.var(projections[~r_smaller_r_max, :], axis=0, ddof=1)
    s_var_vec = s_var_vec - n_var_vec
    a = s_var_vec/n_var_vec
    best_images = np.argsort(a)
    projections = projections[:, :, best_images[:100]]
    images = images[:, :, best_images[9000:10000]]
    pass

    # remove outliers
    # sorted_indices = np.argsort(n_var_vec)
    # images_to_keep = sorted_indices[np.arange(num_outliers, len(n_var_vec) - num_outliers)]
    # images_to_discard1 = np.union1d(sorted_indices[:num_outliers], sorted_indices[:-num_outliers])
    # sorted_images_to_keep = np.sort(images_to_keep)
    # n_var_vec = n_var_vec[sorted_images_to_keep]
    # s_var_vec = s_var_vec[sorted_images_to_keep]
    # projections = projections[:, :, sorted_images_to_keep]
    #
    # sorted_indices = np.argsort(s_var_vec)
    # images_to_keep = sorted_indices[np.arange(num_outliers, len(n_var_vec) - num_outliers)]
    # images_to_discard2 = np.union1d(sorted_indices[:num_outliers], sorted_indices[:-num_outliers])
    # sorted_images_to_keep = np.sort(images_to_keep)
    # n_var_vec = n_var_vec[sorted_images_to_keep]
    # s_var_vec = s_var_vec[sorted_images_to_keep]
    # projections = projections[:, :, sorted_images_to_keep]
    #
    # images_to_discard = np.union1d(images_to_discard1, images_to_discard2)
    # num_images = projections.shape[2]

    # estimate noise variance and signal power
    noise_var = np.mean(n_var_vec)
    signal_power = np.mean(s_var_vec)
    snr = 10 * np.log10(signal_power / noise_var - 1)

    # map images to PSWF expansion coefficients
    converter = Converter(im_size, truncation_param, beta)
    converter.init_direct()
    pswf_coeff = converter.direct_forward(projections)
    angular_frequency = converter.direct_model.angular_frequency.astype('int')
    psi = converter.direct_model.samples

    # Perform Preliminary de-noising of PSWF coefficients by singular value shrinkage (for every angular index)
    max_freq = np.max(angular_frequency)
    ang_freq_0 = angular_frequency == 0
    pswf_coeff_densvs = np.zeros(pswf_coeff.shape, dtype='complex128')
    ranks = np.zeros(max_freq + 1, dtype='int')  # unnecessary
    wtot = []
    psi_spca = []
    spca_coeff = []
    ang_freq_spca = []
    sdtot = []  # unnecessary

    if remove_mean:
        mu = np.mean(pswf_coeff[ang_freq_0], axis=1)
        pswf_coeff[ang_freq_0, :] = (pswf_coeff[ang_freq_0, :].T - mu).T
    else:
        mu = np.zeros(pswf_coeff[ang_freq_0, :].shape[0])

    for m in range(0, max_freq + 1):
        print('Performing preliminary de-noising of angular index: {} out of {}'.format(m, max_freq))
        indices = angular_frequency == m
        pswf_coeff_densvs[indices, :], rank, w, pc, coeff, sd_c = matrix_denoise(pswf_coeff[indices, :], noise_var)
        ranks[m] = rank
        wtot.extend(w.T)
        psi_spca.append(psi[:, indices].dot(pc))
        spca_coeff.append(coeff)
        ang_freq_spca.extend(m * np.ones(rank))
        sdtot.extend(sd_c.T)

    # replace prolates with steerable Principal Components
    mu = psi[:, ang_freq_0].dot(mu.reshape(len(mu), 1)).flatten()
    # mu = psi[:, ang_freq_0].dot(mu.reshape(len(mu), 1))
    psi = np.concatenate(psi_spca, axis=1)
    angular_frequency = np.array(ang_freq_spca, dtype='int')
    spca_coeff = np.concatenate(spca_coeff, axis=0)
    spca_coeff_densvs = (np.array(wtot) * spca_coeff.T).T

    # Form denoised images
    ang_freq_0 = angular_frequency == 0
    i_denoise_svs = (mu + (psi[:, ang_freq_0].dot(spca_coeff_densvs[ang_freq_0, :]) + 2 * np.real(psi[:, ~ang_freq_0].dot(spca_coeff_densvs[~ang_freq_0, :]))).T).T
    # i_denoise_svs = mu * np.ones((1, num_images))\
    #     + psi[:, ang_freq_0].dot(spca_coeff_densvs[ang_freq_0, :])\
    #     + 2 * np.real(psi[:, ~ang_freq_0].dot(spca_coeff_densvs[~ang_freq_0, :]))
    projections_denoise = np.zeros(projections.shape)
    projections_denoise[r_smaller_r_max, :] = np.real(i_denoise_svs)
    projections_denoise = np.transpose(projections_denoise, axes=(1, 0, 2))

    i_denoise_svs = (mu + (psi[:, ang_freq_0].dot(spca_coeff[ang_freq_0, :]) + 2 * np.real(psi[:, ~ang_freq_0].dot(spca_coeff[~ang_freq_0, :]))).T).T
    # i_denoise_svs = mu * np.ones((1, num_images))\
    #     + psi[:, ang_freq_0].dot(spca_coeff[ang_freq_0, :])\
    #     + 2 * np.real(psi[:, ~ang_freq_0].dot(spca_coeff[~ang_freq_0, :]))
    projections_spca = np.zeros(projections.shape)
    projections_spca[r_smaller_r_max, :] = np.real(i_denoise_svs)
    projections_spca = np.transpose(projections_spca, axes=(1, 0, 2))

    return projections_denoise, projections_spca, psi, angular_frequency, mu, spca_coeff


def matrix_denoise(mat, noise_var):

    size_x, size_y = mat.shape

    # define mat to be wide
    if size_x > size_y:
        mat = mat.T
        m = size_y
        n = size_x
        transpose_flag = True
    else:
        m = size_x
        n = size_y
        transpose_flag = False
    beta = m/n

    u, s, v = np.linalg.svd(mat, full_matrices=False)
    s = s[s > (1 + np.sqrt(beta)) * np.sqrt(n * noise_var)]
    r = len(s)
    u = u[:, :r]
    v = v[:r, :]
    sd_c = n * noise_var * np.sqrt((s ** 2 / n / noise_var - beta - 1) ** 2 - 4 * beta) / s
    w = sd_c / s

    x = np.dot(u * sd_c, v)

    eig_vectors = u
    coeff = (s * v.T).T

    if transpose_flag:
        x = x.T

    return x, r, w, eig_vectors, coeff, sd_c


def max_diff(a, b):
    return np.max(np.absolute(a - b))


example()
