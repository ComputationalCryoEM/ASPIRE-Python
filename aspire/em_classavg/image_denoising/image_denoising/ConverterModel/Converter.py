from aspire.em_classavg.image_denoising.image_denoising.PSWF2D.PSWF2DModel import PSWF2D
import numpy as np
from aspire.em_classavg.image_denoising.image_denoising.ConverterModel import DirectModel, DirectModel_Full


# from image_denoising.image_denoising.ConverterModel.FastModel.FastModel import FastModel

class Converter:
    def __init__(self, im_size, truncation, beta=1.0):
        # im_size is odd
        if im_size % 2:
            even = False
            resolution = int((im_size - 1) / 2)
        # im_size is even
        else:
            even = True
            resolution = int(im_size / 2)
        self.resolution = resolution
        self.truncation = truncation
        self.beta = beta
        self.even = even

        bandlimit = beta * np.pi * resolution
        self.pswf2d = PSWF2D(bandlimit)

        self.fast_model = None
        self.direct_model = None

    def init_direct(self, direct_type='full'):
        # self.direct_model = DirectModel(self.resolution, self.truncation, self.beta, self.pswf2d, self.even)
        if direct_type == 'full':
            self.direct_model = DirectModel_Full(self.resolution, self.truncation, self.beta, self.pswf2d, self.even)
        elif direct_type == 'orig':
            self.direct_model = DirectModel(self.resolution, self.truncation, self.beta, self.pswf2d, self.even)
        else:
            ValueError('no such supported type')

    def init_fast(self):
        self.fast_model = DirectModel(self.resolution, self.truncation, self.beta, self.pswf2d, self.even)

    def direct_forward(self, images):
        """

        :param images: (im_size, im_size) or (im_size, im_size, n) ndarray
            if images is of size (im_size, im_size) it will be viewed as n = 1
        :return: transformed images
        """
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.forward(images)

    def fast_forward(self, images):
        if self.fast_model is None:
            print("Fast model is not initialized. Use init_fast() to proceed.")
            return -1
        return self.fast_model.forward(images)

    def direct_backward(self, coefficients):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.backward(coefficients)

    def get_prolates_as_images(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_samples_as_images()

    def get_num_prolates(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_num_prolates()

    def get_angular_frequency(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_angular_frequency()

    def get_non_neg_freq_inds(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_non_neg_freq_inds()

    def get_zero_freq_inds(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_zero_freq_inds()

    def get_pos_freq_inds(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_pos_freq_inds()

    def get_neg_freq_inds(self):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.get_neg_freq_inds()

    def mask_points_inside_the_circle(self, images):
        if self.direct_model is None:
            print("Direct model is not initialized. Use init_direct() to proceed.")
            return -1
        return self.direct_model.mask_points_inside_the_circle(images)
