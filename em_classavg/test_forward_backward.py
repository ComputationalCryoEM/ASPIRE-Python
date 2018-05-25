import numpy as np
import matplotlib.pyplot as plt

from image_denoising.image_denoising.ConverterModel.Converter import Converter
import data_utils


def main():
    test_for_itay()
    # test_pswf_compat(True)
    test_pswf_compat(False)
    # test_forward_backward(True)
    # test_forward_backward(False)


def test_forward_backward(is_full=True):

    image = data_utils.mat_to_npy('image')
    image = np.transpose(image, axes=(1, 0))  # pswf shift x-y coordinates

    im_size = image.shape[-1]

    trunc_param = 10
    beta = 0.5
    converter = Converter(im_size, trunc_param, beta)
    if is_full:
        converter.init_direct('full')
    else:
        converter.init_direct('orig')

    coeffs = converter.direct_forward(image)
    image_out = converter.direct_backward(coeffs)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(np.real(image), cmap='gray')
    plt.title("original image")
    plt.subplot(122)
    plt.imshow(np.real(image_out), cmap='gray')
    plt.title("recon image")

    # plt.subplot(223)
    # plt.imshow(np.real(images[5]), cmap='gray')
    # plt.title("original image %d" % 5)
    # plt.subplot(224)
    # plt.imshow(np.real(images_out[5]), cmap='gray')
    # plt.title("recon image %d" % 5)

    plt.show()


def test_for_itay():

    im = data_utils.mat_to_npy('im')
    # images = np.transpose(images, axes=(1, 0))  # pswf shift x-y coordinates

    im_size = im.shape[-1]

    trunc_param = 10
    beta = 0.5
    converter = Converter(im_size, trunc_param, beta)
    converter.init_direct('orig')
    c_im_python = converter.direct_forward(im)

    c_im_matlab = data_utils.mat_to_npy('c_im_matlab')

    diff = np.min(np.abs(np.concatenate((c_im_python - c_im_matlab, c_im_python + c_im_matlab), axis=1)), axis=1)

    np.where(diff > 0)




def test_pswf_compat(is_full=True):

    image = data_utils.mat_to_npy('image')
    image = np.transpose(image, axes=(1, 0))  # pswf shift x-y coordinates

    im_size = image.shape[-1]

    trunc_param = 10
    beta = 0.5
    converter = Converter(im_size, trunc_param, beta)
    if is_full:
        converter.init_direct('full')
    else:
        converter.init_direct('orig')

    c_ims = converter.direct_forward(image)
    if is_full:
        image_out = converter.direct_backward(c_ims)[0]
    else:
        image_out = converter.direct_backward(c_ims)[:, :, 0]

    if is_full:
        c_ims_full = np.transpose(data_utils.mat_to_npy('c_ims_full'))
        c_ims_matlab = c_ims_full
    else:
        c_ims_not_full = data_utils.mat_to_npy('c_ims_not_full')
        c_ims_matlab = c_ims_not_full

    if is_full:
        image_matlab_out = converter.direct_backward(c_ims_matlab)[0]
    else:
        image_matlab_out = converter.direct_backward(c_ims_matlab)[:, :, 0]

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(np.real(image_out), cmap='gray')
    plt.title("python")
    plt.subplot(122)
    plt.imshow(np.real(image_matlab_out), cmap='gray')
    plt.title("matlab")

    plt.show()


if __name__ == "__main__":
    main()

