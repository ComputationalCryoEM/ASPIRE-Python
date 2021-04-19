import logging

import numpy as np
from scipy.linalg import qr
from tqdm import tqdm

from aspire.basis import FSPCABasis
from aspire.classification import Class2D
from aspire.covariance import RotCov2D
from aspire.utils import complex_type
from aspire.utils.random import rand

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


# copied for debugging/poc purposes
def icfft2(x):
    if len(x.shape) == 2:
        return np.fft.fftshift(
            np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x))))
        )
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, (1, 2))
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.ifft2(y)
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.fftshift(y, (1, 2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


# lol
def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


# copied for debugging/poc purposes
# very slow function compared to matlab
def rot_align(m, coeff, pairs):
    n_theta = 360.0
    p = pairs.shape[0]
    c = np.zeros((m + 1, p), dtype="complex128")
    m_list = np.arange(1, m + 1)

    for i in range(m + 1):
        c[i] = np.einsum(
            "ij, ij -> j", np.conj(coeff[i][:, pairs[:, 0]]), coeff[i][:, pairs[:, 1]]
        )

    c2 = np.flipud(np.conj(c[1:]))
    b = (2 * m + 1) * np.real(icfft(np.concatenate((c2, c), axis=0)))
    rot = np.argmax(b, axis=0)
    rot = (rot - m) * n_theta / (2 * m + 1)

    x_old = -np.ones(p)
    x_new = rot
    precision = 0.001
    num_iter = 0

    m_list_ang = m_list * np.pi / 180
    m_list_ang_1j = 1j * m_list_ang
    c_for_f_prime_1 = np.einsum("i, ij -> ji", m_list_ang, c[1:]).copy()
    c_for_f_prime_2 = np.einsum("i, ji -> ji", m_list_ang, c_for_f_prime_1).copy()

    diff = np.absolute(x_new - x_old)
    while np.max(diff) > precision:
        diff = np.absolute(x_new - x_old)
        indices = np.where(diff > precision)[0]
        x_old1 = x_new[indices]
        tmp = np.exp(np.outer(m_list_ang_1j, x_old1))

        delta = np.imag(
            np.einsum("ji, ij -> j", c_for_f_prime_1[indices], tmp)
        ) / np.real(np.einsum("ji, ij -> j", c_for_f_prime_2[indices], tmp))
        delta_bigger10 = np.where(np.abs(delta) > 10)[0]
        tmp_random = np.random.rand(len(delta))
        tmp_random = tmp_random[delta_bigger10]
        delta[delta_bigger10] = np.sign(delta_bigger10) * 10 * tmp_random
        x_new[indices] = x_old1 - delta
        num_iter += 1
        if num_iter > 100:
            break

    rot = x_new
    m_list = np.arange(m + 1)
    m_list_ang = m_list * np.pi / 180
    c = c * np.exp(1j * np.outer(m_list_ang, rot))
    corr = (np.real(c[0]) + 2 * np.sum(np.real(c[1:]), axis=0)) / 2

    return corr, rot


class RIRClass2D(Class2D):
    def __init__(
        self,
        src,
        pca_basis,
        fspca_components=400,
        alpha=1 / 3,
        sample_n=4000,
        bispectrum_componenents=300,
        n_nbor=100,
        dtype=None,
    ):
        """
        Constructor of an object for classifying 2D images using
        Rotationally Invariant Representation (RIR) algorithm.

        Z. Zhao, Y. Shkolnisky, A. Singer, Rotationally Invariant Image Representation
        for Viewing Direction Classification in Cryo-EM. (2014)

        :param src: Source instance
        :param basis: (Fast) Fourier Bessel Basis instance
        :param fspca_components: Components (top eigvals) to keep from full FSCPA, default truncates to  400.
        :param sample_n: A number and associated method used to confuse your enemies.
        :param alpha: Amplitude Power Scale, default 1/3 (eq 20 from  RIIR paper).
        :param dtype: optional dtype, otherwise taken from src.
        :return: RIRClass2D instance to be used to compute bispectrum-like rotationally invariant 2D classification.
        """
        super().__init__(src=src, dtype=dtype)

        self.pca_basis = pca_basis
        self.fb_basis = self.pca_basis.basis
        self.fspca_components = fspca_components
        self.sample_n = sample_n
        self.alpha = alpha
        self.bispectrum_componenents = bispectrum_componenents
        self.n_nbor = n_nbor
        # Type checks
        if self.dtype != self.fb_basis.dtype:
            logger.warning(
                f"RIRClass2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
            )

        # Sanity Checks
        assert hasattr(self.pca_basis, "calculate_bispectrum")

        # For now, only run with FSPCA
        assert isinstance(self.pca_basis, FSPCABasis)

    def classify(self):
        """
        Perform the 2D images classification.
        """

        ## Stage 1: Compute coef and reduce dimensionality.
        # Memioze/batch this later when result is working

        # Initial round of component truncation is before bispectrum.
        #  default of 400 components was taken from legacy code.
        # Instantiate a new compressed (truncated) basis.
        self.pca_basis = self.pca_basis.compress(self.fspca_components)

        # Expand into the compressed FSPCA space.
        fb_coef = self.fb_basis.evaluate_t(self.src.images(0, self.src.n))
        coef = self.pca_basis.expand(fb_coef)
        ## should be equiv, make a ut
        # coef = self.pca_basis.expand_from_image_basis(self.src.images(0, self.src.n))

        # Legacy code included a sanity check:
        non_zero_freqs = self.pca_basis.complex_angular_indices != 0
        # coef_norm = np.log(np.power(np.abs(coef[:,non_zero_freqs]), self.alpha)).all())
        # just stick to the paper (eq 20) for now , look at this more later.
        coef_normed = np.where(
            coef == 0, 0, coef / np.power(np.abs(coef), 1 - self.alpha)
        )  # should use an epsilon here...

        if not np.isfinite(coef_normed).all():
            raise ValueError("Coefs should be finite")

        ### Compute and reduce Bispectrum

        m = np.power(self.pca_basis.eigvals, self.alpha)
        m = m[
            self.pca_basis.complex_angular_indices != 0
        ]  # filter non_zero_freqs eq 18,19
        pm = m / np.sum(m)
        x = rand(len(m))
        m_mask = x < self.sample_n * pm

        M = None

        for i in tqdm(range(self.src.n)):
            B = self.pca_basis.calculate_bispectrum(
                coef_normed[i], filter_nonzero_freqs=True
            )

            #### Truncate Bispectrum (by sampling)
            #### Note, where is this written down? (and is it even needed?
            B = B[m_mask][:, m_mask]
            logger.info(f"Truncating Bispectrum to {B.shape} ({np.size(B)}) coefs.")

            # B is symmetric, take lower triangle of first two axis.
            tril = np.tri(B.shape[0], dtype=bool)
            B = B[tril, :]
            logger.info(f"Symmetry reduced Bispectrum to {np.size(B)} coefs.")
            # B is sparse and should have same sparsity for any image up to underflows...
            B = B.ravel()[np.flatnonzero(B)]
            logger.info(f"Sparse (nnz) reduced Bispectrum to {np.size(B)} coefs.")

            # Legacy code had bispect flattened as CSR and some other hacks.
            #   For now, we'll compute it densely then take nonzeros.
            if M is None:
                # Instanstiate M with B's nnz size
                M = np.empty((self.src.n, B.shape[0]), dtype=coef.dtype)
            M[i] = B

        ### Reduce dimensionality of Bispectrum sample with PCA
        logger.info(
            f"Computing PCA, returning {self.bispectrum_componenents} components."
        )
        # should add memory sanity check here... these can be crushingly large...

        M = M.T  # SVD will expect (n_img) samples as columns.
        u, s, v = self.pca_y(M, self.bispectrum_componenents)
        # ## # Check it looks something like a spectrum.
        # import matplotlib.pyplot as plt
        # plt.semilogy(s)
        # plt.show()
        ## Contruct coefficients
        coef_b = np.einsum("i, ij -> ij", s, np.conjugate(v))
        coef_b_r = np.conjugate(u.T).dot(np.conjugate(M))

        # normalize
        coef_b /= np.linalg.norm(coef_b, axis=0)
        coef_b_r /= np.linalg.norm(coef_b_r, axis=0)
        print(coef_b)
        print(coef_b_r)
        import matplotlib.pyplot as plt

        # plt.imshow(np.abs(coef_b))
        # plt.show()
        # plt.imshow(np.abs(coef_b_r))
        # plt.show()
        ## stage 2: Compute Nearest Neighbors
        classes, corr = self.nn_classification(coef_b, coef_b_r)
        print(classes)
        np.save(f"classes_raw_res{self.fb_basis.nres}_nimg{self.src.n}.npy", classes)
        np.save(f"corr_raw_res{self.fb_basis.nres}_nimg{self.src.n}.npy", corr)

        ## Stage 3: Align

        # translate some variables between this code and the legacy aspire aspire implementation (just trying to figure out of the old code ran...).
        freqs = self.pca_basis.complex_angular_indices
        coeff = coef.T
        n_im = self.src.n
        n_nbor = self.n_nbor

        ### COPIED FROM LEGACY CODE:
        # del coeff_b, concat_coeff
        max_freq = np.max(freqs)
        cell_coeff = []
        for i in range(max_freq + 1):
            cell_coeff.append(
                np.concatenate(
                    (coeff[freqs == i], np.conjugate(coeff[freqs == i])), axis=1
                )
            )

        # maybe pairs should also be transposed
        pairs = np.stack(
            (classes.flatten("F"), np.tile(np.arange(n_im), n_nbor)), axis=1
        )
        corr, rot = rot_align(max_freq, cell_coeff, pairs)

        rot = rot.reshape((n_im, n_nbor), order="F")
        classes = classes.reshape(
            (n_im, n_nbor), order="F"
        )  # this should already be in that shape
        corr = corr.reshape((n_im, n_nbor), order="F")
        id_corr = np.argsort(-corr, axis=1)
        for i in range(n_im):
            corr[i] = corr[i, id_corr[i]]
            classes[i] = classes[i, id_corr[i]]
            rot[i] = rot[i, id_corr[i]]

        class_refl = np.ceil((classes + 1.0) / n_im).astype("int")
        classes[classes >= n_im] = classes[classes >= n_im] - n_im
        rot[class_refl == 2] = np.mod(rot[class_refl == 2] + 180, 360)
        return classes, class_refl, rot, corr, 0

    def nn_classification(self, coeff_b, coeff_b_r, batch_size=2000):
        """
        Perform nearest neighbor classification and alignment.
        """
        # revisit
        n_im = self.src.n
        # Shouldn't have more neighbors than images
        if self.n_nbor >= n_im:
            logger.warning(
                f"Requested {self.n_nbor} self.n_nbors, but only {n_im} images. Setting self.n_nbors={n_im-1}."
            )
            self.n_nbor = n_im - 1

        # concat_coeff = np.concatenate((coeff_b, coeff_b_r), axis=0)
        concat_coeff = np.concatenate((coeff_b, coeff_b_r), axis=1)
        print(concat_coeff.shape)

        num_batches = (
            n_im + batch_size - 1
        ) // batch_size  # int(np.ceil(float(n_im) / batch_size))

        classes = np.zeros((n_im, self.n_nbor), dtype=int)
        for i in range(num_batches):
            start = i * batch_size
            finish = min((i + 1) * batch_size, n_im)
            corr = np.real(
                # I dont understand what they were doing here yet. (the indexing?
                np.dot(np.conjugate(coeff_b[:, start:finish]).T, concat_coeff)
            )
            classes[start:finish] = np.argsort(-corr, axis=1)[:, 1 : self.n_nbor + 1]

        return classes, corr

    def output(self):
        """
        Return class averages.
        """
        pass

    def pca_y(self, x, k, num_iters=2):
        """
        PCA using QR factorization.

        See:

        An algorithm for the principal component analysis of large data sets.
        Halko, Martinsson, Shkolnisky, Tygert , SIAM 2011.

        :param x: Data matrix
        :param k: Number of estimated Principal Components.
        :param num_iters: Number of dot product applications.
        :return: (left Singular Vectors, Singular Values, right Singular Vectors)
        """

        # TODO, move this out of this class, its a general method...
        # Note I should come back, read this paper, and understand this better, looks useful.

        m, n = x.shape

        def operator(mat):
            return x.dot(mat)

        def operator_transpose(mat):
            return np.conj(x.T).dot(mat)

        flag = False
        if m < n:
            flag = True
            operator_transpose, operator = operator, operator_transpose
            m, n = n, m

        ones = np.ones((n, k + 2))
        if x.dtype == np.dtype("complex"):
            h = operator(
                (2 * np.random.random((k + 2, n)).T - ones)
                + 1j * (2 * np.random.random((k + 2, n)).T - ones)
            )
        else:
            h = operator(2 * np.random.random((k + 2, n)).T - ones)

        f = [h]

        for i in range(num_iters):
            h = operator_transpose(h)
            h = operator(h)
            f.append(h)

        f = np.concatenate(f, axis=1)
        # f has e-16 error, q has e-13
        q, _, _ = qr(f, mode="economic", pivoting=True)
        b = np.conj(operator_transpose(q)).T
        u, s, v = np.linalg.svd(b, full_matrices=False)
        # not sure how to fix the signs but it seems like I dont need to
        # TODO use fix_svd, here and matlab
        # u, v = fix_svd(u, v)

        v = v.conj()
        u = np.dot(q, u)

        u = u[:, :k]
        v = v[:k]
        s = s[:k]

        if flag:
            u, v = v.T, u.T

        return u, s, v
