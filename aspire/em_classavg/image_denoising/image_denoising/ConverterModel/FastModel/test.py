import numpy as np
from ConverterModel.Converter import Converter
from scipy.misc import imresize
import time
import os


def test():
    data_path = os.path.join(os.pardir, 'test_data', 'example_data_np_array.npy')
    images = np.load(data_path)
    num_images = images.shape[2]
    bandlimit_ratio = 1.0
    truncation_parameter = 1
    resolutions = [16, 32, 64]
    images_multiplier = 100
    n = images_multiplier * num_images

    for resolution in resolutions:

        # testing with odd grid
        scaled_images = np.zeros((2 * resolution + 1, 2 * resolution + 1, num_images))
        for j in range(num_images):
            scaled_images[:, :, j] = imresize(images[:, :, j], (2 * resolution + 1, 2 * resolution + 1))
        scaled_images = np.repeat(scaled_images, images_multiplier, axis=2)

        print("testing images of size {}".format(scaled_images.shape[0]))
        tic1 = time.clock()
        converter = Converter(scaled_images.shape[0], truncation_parameter, beta=bandlimit_ratio)
        tic2 = time.clock()
        converter.init_fast()
        tic3 = time.clock()
        converter.init_direct()
        print("finished initializing the model in {} sec, the PSWF2D took {} seconds".format(tic3 - tic1, tic2 - tic1))

        # test if coefficients are the same
        tic = time.clock()
        coefficients = converter.fast_forward(scaled_images)
        toc = time.clock()
        t = toc - tic
        tpi = t/n
        print("finished forwarding {} images in {} seconds, average of {} seconds per image".format(n, t, tpi))

        # test reconstruction error
        tic = time.clock()
        reconstructed_images = converter.direct_backward(coefficients)
        toc = time.clock()
        t = toc - tic
        tpi = t / n
        print("finished backward of {} images in {} seconds, average of {} seconds per image".format(n, t, tpi))

        x_1d_grid = range(-resolution, resolution + 1)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_inside_the_circle = r_2d_grid <= resolution

        err = reconstructed_images - scaled_images
        e = np.mean(np.square(np.absolute(err)), axis=2)
        e = np.sum(e[points_inside_the_circle])

        p = np.mean(np.square(np.absolute(scaled_images)), axis=2)
        p = np.sum(p[points_inside_the_circle])

        print("odd images with resolution {} fast coefficients reconstructed error: {}\n".format(resolution, e / p))

        # testing with even grid
        scaled_images = np.zeros((2 * resolution, 2 * resolution, num_images))
        for j in range(num_images):
            scaled_images[:, :, j] = imresize(images[:, :, j], (2 * resolution, 2 * resolution))
        scaled_images = np.repeat(scaled_images, images_multiplier, axis=2)

        print("testing images of size {}".format(scaled_images.shape[0]))
        tic1 = time.clock()
        converter = Converter(scaled_images.shape[0], truncation_parameter, beta=bandlimit_ratio)
        tic2 = time.clock()
        converter.init_fast()
        tic3 = time.clock()
        converter.init_direct()
        print("finished initializing the model in {} sec, the PSWF2D took {} seconds".format(tic3 - tic1, tic2 - tic1))
        # test if coefficients are the same
        tic = time.clock()
        coefficients = converter.fast_forward(scaled_images)
        toc = time.clock()
        t = toc - tic
        tpi = t/n
        print("finished forwarding {} images in {} seconds, average of {} seconds per image".format(n, t, tpi))

        # test reconstruction error
        tic = time.clock()
        reconstructed_images = converter.direct_backward(coefficients)
        toc = time.clock()
        t = toc - tic
        tpi = t / n
        print("finished backward of {} images in {} seconds, average of {} seconds per image".format(n, t, tpi))

        x_1d_grid = range(-resolution, resolution)
        x_2d_grid, y_2d_grid = np.meshgrid(x_1d_grid, x_1d_grid)
        r_2d_grid = np.sqrt(np.square(x_2d_grid) + np.square(y_2d_grid))
        points_inside_the_circle = r_2d_grid <= resolution

        err = reconstructed_images - scaled_images
        e = np.mean(np.square(np.absolute(err)), axis=2)
        e = np.sum(e[points_inside_the_circle])

        p = np.mean(np.square(np.absolute(scaled_images)), axis=2)
        p = np.sum(p[points_inside_the_circle])

        print("even images with resolution {} fast coefficients reconstructed error: {}\n".format(resolution, e / p))


test()
