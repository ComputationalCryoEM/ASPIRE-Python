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
    if len(x.shape) == 2:
        return numpy.fft.fftshift(numpy.transpose(numpy.fft.fft2(numpy.transpose(numpy.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y=numpy.fft.ifftshift(x,(1,2))
        y=numpy.transpose(y,(0,2,1))
        y=numpy.fft.fft2(y)
        y=numpy.transpose(y,(0,2,1))
        y=numpy.fft.fftshift(y,(1,2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")

def icfft2(x):
    if len(x.shape) == 2:
        return numpy.fft.fftshift(numpy.transpose(numpy.fft.ifft2(numpy.transpose(numpy.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = numpy.fft.ifftshift(x, (1, 2))
        y = numpy.transpose(y, (0, 2, 1))
        y = numpy.fft.ifft2(y)
        y = numpy.transpose(y, (0, 2, 1))
        y = numpy.fft.fftshift(y, (1, 2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def choose_support_v6( proj_CTF_noisy, energy_threshold):
    # Determine sizes of the compact support in both real and Fourier space.
    # OUTPUTS:
    # c_limit: Size of support in Fourier space
    # R_limit: Size of support in real space
    # We scale the images in real space by L, so that the noise variance in
    # both real and Fourier domains is the same.
    # Based on code by Tejal from Oct 2015

    L = proj_CTF_noisy.data.shape[1]
    N = int(numpy.floor(L / 2))
    P = proj_CTF_noisy.data.shape[0]
    x, y = numpy.meshgrid(numpy.arange(-N,N+1),numpy.arange(-N,N+1))
    r = numpy.sqrt(numpy.square(x) + numpy.square(y))
    r_flat=r.flatten()
    r_max = N

    img_f = proj_CTF_noisy.data.astype(numpy.float64)
    img = (icfft2(img_f)) * L
    mean_data = numpy.mean(img, axis=0) # Remove mean from the data
    img = img - mean_data

    # Compute the variance of the noise in two different way. See below for the reason.
    img_corner=numpy.reshape(img, (P, L*L))
    img_corner=img_corner[:,r_flat>r_max]
    img_corner=img_corner.flatten()
    var_img=numpy.var(img_corner,ddof=1)

    imgf_corner=numpy.reshape(img_f, (P, L*L))
    imgf_corner=imgf_corner[:,r_flat>r_max]
    imgf_corner=imgf_corner.flatten()
    var_imgf=numpy.var(imgf_corner,ddof=1)

    noise_var = numpy.min([var_img, var_imgf]) # Note, theoretical img_f and
        # img should give the same variance but there is a small difference,
        # choose the smaller one so that you don't get a negative variance or power
        # spectrum in 46,47

    variance_map = numpy.var(img, axis=0,ddof=1)
    variance_map = variance_map.transpose()

    # Mean 2D variance radial function
    radial_var = numpy.zeros(N)
    for i in xrange(N):
        radial_var[i] = numpy.mean(variance_map[numpy.logical_and(r >= i,r < i+1)])

    img_ps = numpy.square(numpy.abs(img_f))
    pspec = numpy.mean(img_ps, 0)
    pspec=pspec.transpose()
    radial_pspec = numpy.zeros(N)

    # Compute the radial power spectrum
    for i in xrange(N):
        radial_pspec[i] = numpy.mean(pspec[numpy.logical_and(r >= i, r < i+1)])

    # Subtract the noise variance
    radial_pspec = radial_pspec - noise_var
    radial_var = radial_var - noise_var

    # compute the cumulative variance and power spectrum.
    c = numpy.linspace(0, 0.5, N)
    R = numpy.arange(0,N)
    cum_pspec = numpy.zeros(N)
    cum_var = numpy.zeros(N)

    for i in xrange(N):
        cum_pspec[i] = numpy.sum(numpy.multiply(radial_pspec[0:i+1],c[0:i+1]))
        cum_var[i] = numpy.sum(numpy.multiply(radial_var[0:i+1],R[0:i+1]))

    cum_pspec = cum_pspec / cum_pspec[-1]
    cum_var = cum_var / cum_var[-1]

    cidx=numpy.where(cum_pspec > energy_threshold)
    c_limit = c[cidx[0][0] - 1] * L
    Ridx=numpy.where(cum_var > energy_threshold)
    R_limit = R[Ridx[0][0] - 1]

    return c_limit, R_limit

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
    projs=mrcfile.open('/home/yoel/data/work/ASPIRE.py/tests/noisy_projs.mrcs')
    #print projs.data[1].shape
    #plt.imshow(projs.data[1], cmap='gray')
    #plt.show()
    #im=numpy.random.rand(4,4)
    #print center(numpy.asarray(im.shape))
    #proj=numpy.random.rand(5,5)
    print choose_support_v6(projs, 0.99)