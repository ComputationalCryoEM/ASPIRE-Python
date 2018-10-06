import numpy as np
from image_denoising.image_denoising.ConverterModel.FastModel.FastModel_init_utils import generate_pswf_quad, parameters_for_forward
from image_denoising.image_denoising.ConverterModel.FastModel.FastModel_forward_utils import forward


class FastModel:
    def __init__(self, resolution, truncation, beta, pswf2d, even):
        # find max alpha for each N
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
        points_inside_circle = r_2d_grid <= resolution

        self.resolution = resolution
        self.truncation = truncation
        self.beta = beta

        bandlimit = self.beta * np.pi * self.resolution
        self.bandlimit = bandlimit

        a, b, c, d, e, f = generate_pswf_quad(4 * self.resolution, 2 * bandlimit, 1e-16, 1e-16, 1e-16)

        self.pswf_radial_quad = pswf2d.evaluate_all(d, np.zeros(len(d)), max_ns)
        self.quad_rule_pts_x = a
        self.quad_rule_pts_y = b
        self.quad_rule_wts = c
        self.radial_quad_pts = d
        self.quad_rule_radial_wts = e
        self.num_angular_pts = f
        self.angular_frequency = np.repeat(np.arange(len(max_ns)), max_ns).astype('float')
        self.radian_frequency = np.concatenate([range(1, l + 1) for l in max_ns]).astype('float')
        self.alpha_nn = np.array(alpha_all)

        # pre computing variables for forward
        us_fft_pts, blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max =\
            parameters_for_forward(resolution, beta, self)
        self.us_fft_pts = us_fft_pts
        self.blk_r = blk_r
        self.num_angular_pts = num_angular_pts
        self.r_quad_indices = r_quad_indices
        self.numel_for_n = numel_for_n
        self.indices_for_n = indices_for_n
        self.n_max = n_max
        self.points_inside_circle = points_inside_circle
        self.image_height = len(x_1d_grid)
        self.points_inside_circle_vec = points_inside_circle.reshape(self.image_height ** 2)
        self.size_x = len(points_inside_circle)

    def forward(self, images):
        # start and finish are for the threads option in the future
        images_shape = images.shape
        start = 0

        # if we got several images
        if len(images_shape) == 3:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], images_shape[2]), order='F')
            finish = images_shape[2]

        # else we got only one image
        else:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], 1), order='F')
            finish = 1

        flattened_images = flattened_images[self.points_inside_circle_vec, :]

        coefficients = forward(flattened_images, self, start, finish)
        return coefficients

