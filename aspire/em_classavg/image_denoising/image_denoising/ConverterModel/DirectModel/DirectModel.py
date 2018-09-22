import numpy as np
import pycuda.gpuarray as gpuarray

import skcuda.linalg as linalg
import aspire.em_classavg.mask_images_kernel as mask_images_kernel
import aspire.em_classavg.config as config

class DirectModel:
    def __init__(self, resolution, truncation, beta, pswf2d, even):
        # find max alpha for each N

        linalg.init()  # TODO: is this the right place to do init???
        max_ns = []
        a = np.square(float(beta * resolution) / 2)
        m = 0
        alpha_all = []
        while True:
            alpha = pswf2d.alpha_all[m]

            lambda_var = a * np.square(np.absolute(alpha))
            gamma = np.sqrt(np.absolute(lambda_var / (1 - lambda_var)))

            n_end = np.where(gamma <= truncation)[0]

            if len(n_end) != 0:
                n_end = n_end[0]
                if n_end == 0:
                    break
                max_ns.extend([n_end])
                alpha_all.extend(alpha[:n_end])
                m += 1

        if even:
            x_1d_grid = range(-resolution, resolution)
        else:
            x_1d_grid = range(-resolution, resolution + 1)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_inside_the_circle = r_2d_grid <= resolution
        x = y_2d_grid[points_inside_the_circle]
        y = x_2d_grid[points_inside_the_circle]

        r_2d_grid_on_the_circle = np.sqrt(np.square(x) + np.square(y)) / resolution
        theta_2d_grid_on_the_circle = np.angle(x + 1j * y)
        self.samples = pswf2d.evaluate_all(r_2d_grid_on_the_circle, theta_2d_grid_on_the_circle, max_ns)

        self.resolution = resolution
        self.truncation = truncation
        self.beta = beta
        self.even = even

        self.angular_frequency = np.repeat(np.arange(len(max_ns)), max_ns).astype('float')
        self.radian_frequency = np.concatenate([range(1, l + 1) for l in max_ns]).astype('float')
        self.alpha_nn = np.array(alpha_all)
        self.samples = (self.beta / 2.0) * self.samples * self.alpha_nn
        self.samples_conj_transpose = self.samples.conj().transpose()

        self.image_height = len(x_1d_grid)

        if config.is_use_gpu:
            self.points_inside_the_circle = gpuarray.to_gpu(points_inside_the_circle).astype('complex64')
        else:
            self.points_inside_the_circle = points_inside_the_circle

        self.points_inside_the_circle_vec = points_inside_the_circle.reshape(self.image_height ** 2)

        self.non_neg_freq_inds = slice(0, len(self.angular_frequency))

        tmp = np.nonzero(self.angular_frequency == 0)[0]
        self.zero_freq_inds = slice(tmp[0], tmp[-1] + 1)

        tmp = np.nonzero(self.angular_frequency > 0)[0]
        self.pos_freq_inds = slice(tmp[0], tmp[-1] + 1)

    def get_points_inside_the_circle(self):
        if config.is_use_gpu:
            return self.points_inside_the_circle.get().astype('bool')
        else:
            return self.points_inside_the_circle

    def mask_points_inside_the_circle(self, images):
        if config.is_use_gpu:
            return self.__mask_points_inside_the_circle_gpu(images)
        else:
            return self.__mask_points_inside_the_circle_cpu(images)

    def __mask_points_inside_the_circle_cpu(self, images):
        return images * self.points_inside_the_circle

    def __mask_points_inside_the_circle_gpu(self, images):
        return mask_images_kernel.do_mask_gpu(images, self.points_inside_the_circle)

    def forward(self, images):
        images_shape = images.shape

        # if we got several images
        if len(images_shape) == 3:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], images_shape[2]), order='F')

        # else we got only one image
        else:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], 1), order='F')

        flattened_images = flattened_images[self.points_inside_the_circle_vec, :]
        coefficients = self.samples_conj_transpose.dot(flattened_images)
        return coefficients

    def backward(self, coefficients):
        # if we got only one vector
        if len(coefficients.shape) == 1:
            coefficients = coefficients.reshape((len(coefficients), 1))

        angular_is_zero = np.absolute(self.angular_frequency) == 0
        flatten_images = self.samples[:, angular_is_zero].dot(coefficients[angular_is_zero]) + \
                         2.0 * np.real(self.samples[:, ~angular_is_zero].dot(coefficients[~angular_is_zero]))

        n_images = int(flatten_images.shape[1])
        images = np.zeros((self.image_height, self.image_height, n_images)).astype('complex')
        images[self.get_points_inside_the_circle(), :] = flatten_images
        images = np.transpose(images, axes=(1, 0, 2))
        return images

    def get_samples_as_images(self):
        print("get_samples_as_images is not supported.")
        return -1

    def get_angular_frequency(self):
        return self.angular_frequency

    def get_num_prolates(self):
        # TODO: fix once Itay gives the samples in the correct way
        print("get_samples_as_images is not supported.")
        return -1

    def get_non_neg_freq_inds(self):
        return self.non_neg_freq_inds

    def get_zero_freq_inds(self):
        # first = next(x[0] for x in enumerate(self.angular_frequency) if x[1] == 0)
        # last = next(x[0] for x in enumerate(reversed(self.angular_frequency)) if x[1] == 0)
        return self.zero_freq_inds

    def get_pos_freq_inds(self):
        return self.pos_freq_inds

    def get_neg_freq_inds(self):
        ValueError('no negative frequencies')

class DirectModel_Full(DirectModel):

    def __init__(self, resolution, truncation, beta, pswf2d, even):
        super().__init__(resolution, truncation, beta, pswf2d, even)

        n_pos_freqs = np.size(np.where(self.angular_frequency > 0))
        tmp = np.arange(n_pos_freqs) + len(self.angular_frequency)  # we implicetely put the negative freqs at the end
        # self.unq_ang_freqs, self.Iunq_ang_freqs = np.unique(self.angular_frequency,return_inverse=True)
        self.neg_freq_inds = slice(tmp[0], tmp[-1] + 1)

        # images = data_utils.mat_to_npy('images')
        # PSWF_Nn_p_psis = data_utils.mat_to_npy('PSWF_Nn_p_psis').astype('complex64')
        # PSWF_Nn_p_psis = np.transpose(PSWF_Nn_p_psis, axes=(2, 0, 1)).copy()  # move to python convention
        # self.samples_as_images = PSWF_Nn_p_psis
        self.samples_as_images = self.calc_samples_as_images()
        if config.is_use_gpu:
            self.samples_as_images = gpuarray.to_gpu(self.samples_as_images)

    def get_neg_freq_inds(self):
        return self.neg_freq_inds

    def get_num_prolates(self):
        return self.non_neg_freq_inds.stop - self.non_neg_freq_inds.start + self.neg_freq_inds.stop - self.neg_freq_inds.start

    def get_angular_frequency(self):
        return np.concatenate((self.angular_frequency, -self.angular_frequency[self.pos_freq_inds]))

    def forward(self, images):
        if np.ndim(images) == 3:
            images = np.transpose(images, axes=(1, 2, 0))  # TODO: currently forward expexts images in matlab format
        coeffs_non_neg = super().forward(images)
        coeffs_non_neg = np.transpose(coeffs_non_neg)  # TODO: we want the data to be in rows

        n_images = len(coeffs_non_neg)
        coeffs = np.zeros((n_images, self.get_num_prolates())).astype('complex64')
        coeffs[:, self.non_neg_freq_inds] = coeffs_non_neg
        coeffs[:, self.neg_freq_inds] = np.conj(coeffs[:, self.pos_freq_inds])
        return coeffs

    def backward(self, coefficients):
        # first peal off the negative frequencies
        if coefficients.ndim == 1:
            coefficients_non_neg = np.delete(coefficients, self.neg_freq_inds)
        else:
            coefficients_non_neg = np.delete(coefficients, self.neg_freq_inds, axis=1)

        coefficients_non_neg = np.transpose(coefficients_non_neg)  # TODO: currently forward expects in matlab format
        images = super().backward(coefficients_non_neg)
        if np.ndim(images) == 3:
            images = np.transpose(images, axes=(2, 0, 1))  # TODO: currently forward expects images in matlab format
        return images

    def get_samples_as_images(self):

        return self.samples_as_images

    def calc_samples_as_images_old(self):

        self_samples = np.transpose(self.samples)  # TODO: fix once Itay fixes
        samples = np.zeros((self.image_height, self.image_height, self.get_num_prolates())).astype('complex64')
        samples_non_neg_freq = samples[self.get_points_inside_the_circle()]
        samples_non_neg_freq[:, self.non_neg_freq_inds] = self.samples
        samples_non_neg_freq[:, self.neg_freq_inds] = np.transpose(np.conj(self_samples[self.pos_freq_inds]))
        samples[self.get_points_inside_the_circle()] = samples_non_neg_freq

        samples = np.transpose(samples, axes=(1, 0, 2))  # python is row based but the mask assumes column base
        samples = np.transpose(samples, axes=(2, 0, 1))  # we want the first index to the image number

        return samples

    def calc_samples_as_images(self):

        self_samples = np.transpose(self.samples)
        samples = np.zeros((self.get_num_prolates(), self.image_height, self.image_height)).astype('complex64')
        samples_non_neg = np.zeros((self.non_neg_freq_inds.stop - self.non_neg_freq_inds.start, self.image_height, self.image_height)).astype('complex64')

        for counter, sample in enumerate(self_samples):
            samples_non_neg[counter, self.get_points_inside_the_circle()] = sample

        samples[self.non_neg_freq_inds] = samples_non_neg
        samples[self.neg_freq_inds] = np.conj(samples[self.pos_freq_inds])

        samples = np.transpose(samples, axes=(0, 2, 1)).copy()  # python is row based but the mask assumes column base

        return samples
