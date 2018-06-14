import numpy as np
from ConverterModel.Converter import Converter
from scipy.misc import imresize
import time
import os


def test():
    data_path = os.path.join('test_data', 'example_data_np_array.npy')
    images = np.load(data_path)
    num_images = images.shape[2]
    bandlimit_ratio = 1.0
    truncation_parameter = 1
    resolutions = [64]
    images_multiplier = 100
    n = images_multiplier * num_images

    for resolution in resolutions:

        # testing with odd grid
        scaled_images = np.zeros((2 * resolution + 1, 2 * resolution + 1, num_images))
        for j in range(num_images):
            scaled_images[:, :, j] = imresize(images[:, :, j], (2 * resolution + 1, 2 * resolution + 1))
        scaled_images = np.repeat(scaled_images, images_multiplier, axis=2)

        print("testing images of size {}\n".format(scaled_images.shape[0]))

        # initializing models
        tic1 = time.clock()
        converter = Converter(scaled_images.shape[0], truncation_parameter, beta=bandlimit_ratio)
        tic2 = time.clock()
        converter.init_fast()
        tic3 = time.clock()
        converter.init_direct()
        tic4 = time.clock()
        print("finished initializing PSWF2D in {}".format(tic2 - tic1))
        print("finished initializing FastModel in {}".format(tic3 - tic2))
        print("finished initializing DirectModel in {}\n".format(tic4 - tic3))

        # forwarding images
        tic = time.clock()
        coefficients_fast = converter.fast_forward(scaled_images)
        toc = time.clock()
        t = toc - tic
        tpi = t/n
        print("finished fast forwarding {} images in {} seconds, average of {} seconds per image".format(n, t, tpi))

        tic = time.clock()
        coefficients_direct = converter.direct_forward(scaled_images)
        toc = time.clock()
        t = toc - tic
        tpi = t/n
        print("finished direct forwarding {} images in {} seconds, average of {} seconds per image\n".format(n, t, tpi))

        # test if coefficients are the same
        print("Maximum absolute difference between coefficients is {}\n".format(np.max(np.absolute(coefficients_fast - coefficients_direct))))

        # test reconstruction error
        tic = time.clock()
        reconstructed_images_direct = converter.direct_backward(coefficients_direct)
        reconstructed_images_fast = converter.direct_backward(coefficients_fast)
        toc = time.clock()
        t = toc - tic
        tpi = t / (2 * n)
        print("finished backward of {} images in {} seconds, average of {} seconds per image\n".format(2 * n, t, tpi))

        x_1d_grid = range(-resolution, resolution + 1)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_inside_the_circle = r_2d_grid <= resolution

        err_slow = reconstructed_images_direct - scaled_images
        e_slow = np.mean(np.square(np.absolute(err_slow)), axis=2)
        e_slow = np.sum(e_slow[points_inside_the_circle])

        err_fast = reconstructed_images_fast - scaled_images
        e_fast = np.mean(np.square(np.absolute(err_fast)), axis=2)
        e_fast = np.sum(e_fast[points_inside_the_circle])

        p = np.mean(np.square(np.absolute(scaled_images)), axis=2)
        p = np.sum(p[points_inside_the_circle])

        print("odd images with resolution {} fast coefficients reconstructed error: {}".format(resolution, e_fast / p))
        print("odd images with resolution {} direct coefficients reconstructed error: {}\n".format(resolution, e_slow / p))

        # testing with even grid
        scaled_images = np.zeros((2 * resolution, 2 * resolution, num_images))
        for j in range(num_images):
            scaled_images[:, :, j] = imresize(images[:, :, j], (2 * resolution, 2 * resolution))
        scaled_images = np.repeat(scaled_images, images_multiplier, axis=2)

        print("testing images of size {}\n".format(scaled_images.shape[0]))

        # initializing models
        tic1 = time.clock()
        converter = Converter(scaled_images.shape[0], truncation_parameter, beta=bandlimit_ratio)
        tic2 = time.clock()
        converter.init_fast()
        tic3 = time.clock()
        converter.init_direct()
        tic4 = time.clock()
        print("finished initializing PSWF2D in {}".format(tic2 - tic1))
        print("finished initializing FastModel in {}".format(tic3 - tic2))
        print("finished initializing DirectModel in {}\n".format(tic4 - tic3))

        # forwarding images
        tic = time.clock()
        coefficients_fast = converter.fast_forward(scaled_images)
        toc = time.clock()
        t = toc - tic
        tpi = t / n
        print("finished fast forwarding {} images in {} seconds, average of {} seconds per image".format(n, t, tpi))

        tic = time.clock()
        coefficients_direct = converter.direct_forward(scaled_images)
        toc = time.clock()
        t = toc - tic
        tpi = t / n
        print("finished direct forwarding {} images in {} seconds, average of {} seconds per image\n".format(n, t, tpi))

        # test if coefficients are the same
        print("Maximum absolute difference between coefficients is {}\n".format(np.max(np.absolute(coefficients_fast - coefficients_direct))))

        # test reconstruction error
        tic = time.clock()
        reconstructed_images_direct = converter.direct_backward(coefficients_direct)
        reconstructed_images_fast = converter.direct_backward(coefficients_fast)
        toc = time.clock()
        t = toc - tic
        tpi = t / (2 * n)
        print("finished backward of {} images in {} seconds, average of {} seconds per image\n".format(2 * n, t, tpi))

        x_1d_grid = range(-resolution, resolution)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_inside_the_circle = r_2d_grid <= resolution

        err_slow = reconstructed_images_direct - scaled_images
        e_slow = np.mean(np.square(np.absolute(err_slow)), axis=2)
        e_slow = np.sum(e_slow[points_inside_the_circle])

        err_fast = reconstructed_images_fast - scaled_images
        e_fast = np.mean(np.square(np.absolute(err_fast)), axis=2)
        e_fast = np.sum(e_fast[points_inside_the_circle])

        p = np.mean(np.square(np.absolute(scaled_images)), axis=2)
        p = np.sum(p[points_inside_the_circle])

        print("even images with resolution {} fast coefficients reconstructed error: {}".format(resolution, e_fast / p))
        print("even images with resolution {} direct coefficients reconstructed error: {}\n".format(resolution, e_slow / p))


test()
