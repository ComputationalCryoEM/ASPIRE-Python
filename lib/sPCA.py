#!/opt/anaconda2/bin/python

import numpy
import mrcfile
import scipy.io


class SpcaData:
    def __init__(self):
        self.eigval = 0
        self.freqs = 0
        self.radial_freqs = 0
        self.coeff = 0
        self.mean = 0
        self.c = 0
        self.R = 0
        self.eig_im = 0
        self.fn0 = 0


def cfft2(x):
        return numpy.fft.fftshift(numpy.transpose(numpy.fft.fft2(numpy.transpose(numpy.fft.ifftshift(x)))))

def icfft2(x):
        return numpy.fft.fftshift(numpy.transpose(numpy.fft.ifft2(numpy.transpose(numpy.fft.ifftshift(x)))))


def choose_support_v6( proj_CTF_noisy, energy_threshold):
    # Determine sizes of the compact support in both real and Fourier space.
    # OUTPUTS:
    # c_limit: Size of support in Fourier space
    # R_limit: Size of support in real space
    # We scale the images in real space by L, so that the noise variance in
    # both real and Fourier domains is the same.
    # Based on code by Tejal from Oct 2015

    L = proj_CTF_noisy.data.shape[1]
    N = numpy.floor(L / 2.0)
    P = proj_CTF_noisy.data[0]
    x, y = numpy.meshgrid(numpy.arange(-N,N),numpy.arange(-N,N))
    r = numpy.sqrt(numpy.square(x) + numpy.square(y))
    r_max = N

    pass

def compute_sPCA(images, noise_v_r, adaptive_support = False):
    # images is mrcfile
    n=images.data.shape[0]
    L=images.data.shape[1]

    # if adaptive_support:
    #     energy_thresh = 0.99
    #     # Estimate bandlimit and compact support size
    #     [c, R] = choose_support_v6(cfft2(images), energy_thresh)
    #     c = c * (0.5 / numpy.floor(L / 2.0)) # Rescaling between 0 and 0.5
    # else:
    #     c = 0.5
    #     R = numpy.floor(L / 2.0)



if __name__ == "__main__":
    print "ok"
    # projs=mrcfile.open('/home/yoel/data/work/ASPIRE.py/tests/noisy_projs.mrcs')
    #print projs.data[1].shape
    #plt.imshow(projs.data[1], cmap='gray')
    #plt.show()
    #im=numpy.random.rand(4,4)
    #print center(numpy.asarray(im.shape))
    #proj=numpy.random.rand(5,5)
    # choose_support_v6(projs, 0.99)