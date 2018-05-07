import numpy as np
import pycuda.gpuarray as gpuarray

import skcuda.linalg as linalg

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
        self.points_inside_the_circle = points_inside_the_circle
        self.points_inside_the_circle_gpu = gpuarray.to_gpu(self.points_inside_the_circle).astype('complex64')
        self.points_inside_the_circle_vec = points_inside_the_circle.reshape(self.image_height ** 2)

    def mask_points_inside_the_circle(self, images):
        return images * self.points_inside_the_circle

    def mask_points_inside_the_circle_gpu(self, images):

        images_masked = gpuarray.zeros_like(images)
        for i, image in enumerate(images):
            images_masked[i] = linalg.misc.multiply(image, self.points_inside_the_circle_gpu)

        # for i in len(images):
        #     images_masked[i] = linalg.misc.multiply(images[i],self.points_inside_the_circle_gpu)

        return images_masked

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
        images[self.points_inside_the_circle, :] = flatten_images
        images = np.transpose(images, axes=(1, 0, 2))
        return images

    def get_samples_as_images(self):
        print("get_samples_as_images is not supported.")
        return -1

    def get_angular_frequency(self):
        return self.angular_frequency

    def get_num_prolates(self):
        return np.shape(self.samples)[0]  # TODO: fix once Itay gives the samples in the correct way

    def get_non_neg_freq_inds(self):
        return np.arange(len(self.angular_frequency))

    def get_non_neg_freq_inds_gpu(self):
        return slice(0,len(self.angular_frequency))
        # return np.arange(len(self.angular_frequency))

    def get_zero_freq_inds_slice(self):
        # TODO: avoid computing this each time. Rather calc once the first and last and have this method just construct the slice
        tmp = np.nonzero(self.angular_frequency == 0)[0]
        return slice(tmp[0], tmp[-1] + 1)

    def get_pos_freq_inds_slice(self):
        # TODO: avoid computing this each time. Rather calc once the first and last and have this method just construct the slice
        tmp = np.nonzero(self.angular_frequency > 0)[0]
        return slice(tmp[0], tmp[-1] + 1)

    def get_neg_freq_inds_slice(self):
        ValueError('no negative frequencies')

    def get_zero_freq_inds(self):
        return np.nonzero(self.angular_frequency == 0)[0]

    def get_pos_freq_inds(self):
        return np.nonzero(self.angular_frequency > 0)[0]

    def get_neg_freq_inds(self):
        ValueError('no negative frequencies')

class DirectModel_Full(DirectModel):

    def __init__(self, resolution, truncation, beta, pswf2d, even):
        super().__init__(resolution, truncation, beta, pswf2d, even)

        n_pos_freqs = np.size(np.where(self.angular_frequency > 0))
        self.neg_freq_inds = np.arange(n_pos_freqs) + len(self.angular_frequency) # we put the negative freqs at the end
        # self.unq_ang_freqs, self.Iunq_ang_freqs = np.unique(self.angular_frequency,return_inverse=True)

        self.samples_as_images = self.calc_samples_as_images()
        self.samples_as_images_gpu = gpuarray.to_gpu(self.samples_as_images) #  TODO: keep either samples gpu or samples cpu. not both

    def get_neg_freq_inds_slice(self):
        return slice(self.neg_freq_inds[0], self.neg_freq_inds[-1] + 1)

    def get_neg_freq_inds(self):
        return self.neg_freq_inds

    def get_num_prolates(self):
        return len(self.get_zero_freq_inds()) + len(self.get_pos_freq_inds()) + len(self.neg_freq_inds)

    def get_angular_frequency(self):
        return np.concatenate((self.angular_frequency, -self.angular_frequency[self.get_pos_freq_inds()]))

    def forward(self, images):
        if np.ndim(images) == 3:
            images = np.transpose(images, axes=(1, 2, 0))  # TODO: currently forward expexts images in matlab format
        coeffs_non_neg = super().forward(images)
        coeffs_non_neg = np.transpose(coeffs_non_neg)  # TODO: we want the data to be in rows

        n_images = len(coeffs_non_neg)
        coeffs = np.zeros((n_images, self.get_num_prolates())).astype('complex')
        coeffs[:, self.get_non_neg_freq_inds()] = coeffs_non_neg
        coeffs[:, self.neg_freq_inds] = np.conj(coeffs[:, self.get_pos_freq_inds()])
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

    def get_samples_as_images_gpu(self):

        return self.samples_as_images_gpu

    def calc_samples_as_images_old(self):

        self_samples = np.transpose(self.samples)  # TODO: fix once Itay fixes
        samples = np.zeros((self.image_height, self.image_height, self.get_num_prolates())).astype('complex64')

        samples_non_neg_freq = samples[self.points_inside_the_circle]
        samples_non_neg_freq[:, self.get_non_neg_freq_inds()] = self.samples
        samples_non_neg_freq[:, self.get_neg_freq_inds()] = np.transpose(np.conj(self_samples[self.get_pos_freq_inds()]))
        samples[self.points_inside_the_circle] = samples_non_neg_freq

        samples = np.transpose(samples, axes=(1, 0, 2))  # python is row based but the mask assumes column base
        samples = np.transpose(samples, axes=(2, 0, 1))  # we want the first index to the image number

        return samples


    def calc_samples_as_images(self):

        self_samples = np.transpose(self.samples)
        samples = np.zeros((self.get_num_prolates(), self.image_height, self.image_height)).astype('complex64')
        samples_non_neg = np.zeros((len(self.get_non_neg_freq_inds()), self.image_height, self.image_height)).astype('complex64')

        for counter, sample in enumerate(self_samples):
            samples_non_neg[counter,self.points_inside_the_circle] = sample

        samples[self.get_non_neg_freq_inds()] = samples_non_neg
        samples[self.get_neg_freq_inds()] = np.conj(samples[self.get_pos_freq_inds()])

        samples = np.transpose(samples, axes=(0, 2, 1)).copy()  # python is row based but the mask assumes column base

        return samples

