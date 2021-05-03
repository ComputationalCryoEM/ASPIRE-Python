import logging

import numpy as np
from scipy.linalg import qr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from aspire.basis import FSPCABasis
from aspire.classification import Class2D
from aspire.utils.random import rand

logger = logging.getLogger(__name__)


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

        # # Stage 1: Compute coef and reduce dimensionality.
        # Memioze/batch this later when result is working

        # Initial round of component truncation is before bispectrum.
        #  default of 400 components was taken from legacy code.
        # Instantiate a new compressed (truncated) basis.
        self.pca_basis = self.pca_basis.compress(self.fspca_components)

        # Expand into the compressed FSPCA space.
        fb_coef = self.fb_basis.evaluate_t(self.src.images(0, self.src.n))
        coef = self.pca_basis.expand(fb_coef)
        # # should be equiv, make a ut
        # coef = self.pca_basis.expand_from_image_basis(self.src.images(0, self.src.n))

        # Legacy code included a sanity check:
        # non_zero_freqs = self.pca_basis.complex_angular_indices != 0
        # coef_norm = np.log(np.power(np.abs(coef[:,non_zero_freqs]), self.alpha)).all())
        # just stick to the paper (eq 20) for now , look at this more later.
        coef_normed = np.where(
            coef == 0, 0, coef / np.power(np.abs(coef), 1 - self.alpha)
        )  # should use an epsilon here...

        if not np.isfinite(coef_normed).all():
            raise ValueError("Coefs should be finite")

        # ## Compute and reduce Bispectrum

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

            # ### Truncate Bispectrum (by sampling)
            # ### Note, where is this written down? (and is it even needed?
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

        # ## Reduce dimensionality of Bispectrum sample with PCA
        logger.info(
            f"Computing PCA, returning {self.bispectrum_componenents} components."
        )
        # should add memory sanity check here... these can be crushingly large...

        u, s, v = self.pca_y(M, self.bispectrum_componenents)        
        # u is shaped (features, n_img)
        print('u.shape', u.shape)
        print('s.shape', s.shape)
        print('v.shape', v.shape)

        # GBW
        # M_pca_basis = u.T @ M
        # coef_b = u @ M_pca_basis
        # # same as above, conjugated
        # coef_b_r = u @ (u.T @ np.conjugate(M))
        
       
        ### The following was from legacy code and I haven't figured it out.
        # ## # Check it looks something like a spectrum.
        # import matplotlib.pyplot as plt
        # plt.semilogy(s)
        # plt.show()
        # # Contruct coefficients
        coef_b = np.einsum("i, ij -> ij", s, np.conjugate(v))
        coef_b_r = np.conjugate(u.T).dot(np.conjugate(M))

        # normalize, XXX check axis
        coef_b /= np.linalg.norm(coef_b, axis=0)
        coef_b_r /= np.linalg.norm(coef_b_r, axis=0)

        # import matplotlib.pyplot as plt
        # plt.imshow(np.abs(coef_b))
        # plt.show()
        # plt.imshow(np.abs(coef_b_r))
        # plt.show()

        # # Stage 2: Compute Nearest Neighbors
        classes, refl, corr = self.nn_classification(coef_b, coef_b_r)
        # print(classes)

        # # # Stage 3: Align

        #angles = self.align(classes, refl, coef=fb_coef, basis=self.fb_basis)
        #angles = self.align(classes, refl, coef=coef)

        return self.legacy_align(classes, coef)

        #return classes, None, None, None, None

    def nn_classification(self, coeff_b, coeff_b_r):
        # Before we get clever lets just use a generally accepted implementation.
        
        n_img = coeff_b_r.shape[0]
        print('coeff_b', coeff_b.shape, coeff_b_r.shape)
        
        # Third party tools generally expecting:
        #   slow axis as n_data, fast axis n_features.        
        # Also most third party NN complain about complex...
        # X = np.abs(coeff_b)
        X = coeff_b.view(self.dtype)
        # We'll also want to consider the neighbors under reflection.
        #   These coefficients should be provided by coeff_b_r        
        # X_r = np.abs(coeff_b_r)
        #X_r = coeff_b_r.view(self.dtype)

        # We can compare both non-reflected and reflected representations as one large set by
        #   taking care later that we store refl=True where indices>=n_img
        #X_both = np.concatenate((X, X_r))
        X_both = X # ignore refl for now
        
        nbrs = NearestNeighbors(n_neighbors=self.n_nbor, algorithm='auto').fit(X_both)
        distances, indices = nbrs.kneighbors(X)
        print('indices.shape', indices.shape)
        # any refl?
        logger.info(f'Count indices>n_img: {np.sum(indices>=n_img)}')

        # # lets peek at the distribution of distances
        import matplotlib.pyplot as plt
        plt.hist(distances[:,1:].flatten(), bins='auto')
        plt.show()
        
        # There are two sets of vectors each n_img long.
        #   The second set represents reflected (gbw, unsure..)
        #   When a reflected coef vector is a nearest neighbor,
        #   we notate the original image index (indices modulus),
        #   and notate we'll need the reflection (refl).
        classes = indices % n_img
        refl = np.array(indices // n_img, dtype=bool)        
        corr = distances

        return classes, refl, corr
        
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

        for _ in range(num_iters):
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

    def align(self, classes, refl, coef=None, basis=None, epsilon=1e-10, max_iter=100):
        """
        For each class find the corresponding set of
        in plane rotation angles which align the images.

        :param classes: Integer array of image indices 
        representing images from the same viewing angle.
        :param refl: Bool array corresponding to whether
        the index is reflected.
        :param coef: Optionally supply coef, defaults
        to coefs computed from `basis`.
        :param basis: Optionally allow custom basis, defaults
        to FSPCA basis.        
        :param epsilon: Tolerance for stopping solver.
        :param max_iter: Max iteration before halting solver.

        """

        # As best I can tell the MATLAB code was using
        # Newton-Raphson to find an optimal-ish set of angles.


        # To make our equations we'll need steerable coefficients.
        if basis is None:
            basis = self.pca_basis
        if coef is None:
            coef = basis.evaluate_t(self.src.images(0, self.src.n))

        # For each class c, want equations like:
        #   ||image_c_i - image_c_0|| < delta
        # But I think we can just save some effort and use
        #   the basis coef, so we'll actually want
        #   ||coef_c_i - coef_c_0|| < eps

        # Lets stash a copy of the original image coef since we'll use them several times.        
        coef_0 = coef.copy()

        # Gather some shape information.
        # Initial image_0 should be fixed, so we ignore it in following arrays (n_class_members-1)
        # It would complicate the iterative loop (causing divs by 0).
        n_classes, n_class_members = classes.shape

        # Initial rotation can be zero.
        rotation_angles = np.zeros((n_classes, n_class_members-1),
                                   dtype=self.dtype)

        # We need to make a 3d array of coefs
        C = np.empty((n_classes, n_class_members, basis.count),
                     dtype=coef.dtype)

        # We'll need a place to store our errors,
        error = np.zeros((n_classes, n_class_members-1), dtype=self.dtype)

        # Now that we have some house keeping variables locally,
        #   make a util function to perform updates.
        # This should keep the Newton method looking cleaner.
        def _update_C(C, rotation_angles):
            for cls in tqdm(range(n_classes)):
                for member in range(n_class_members-1):
                    # Get the index wrt original set of images,
                    image_index = classes[cls][member+1]
                    #print(f'update_C cls {cls} member {member} image {image_index}')
                    #   and to rotate coeffs use `basis.rotate` method.
                    #C[cls,member] = basis.rotate(coef_0[image_index], rotation_angles[cls][member], refl[cls][member])
                    C[cls,member] = basis.rotate(coef_0[image_index], rotation_angles[cls][member])
            return C

        # and a similar utility for computing the error.
        def _update_error(C):
            for cls in tqdm(range(n_classes)):
                for member in range(n_class_members-1):
                    image_index = classes[cls][member+1]
                    #   and rotate the coeffs.
                    #error[cls, member] = np.linalg.norm((C[cls,member] - coef_0[cls]) ) #/ nrm
                    #error[cls, member] = np.sum(np.abs(C[cls,member] - coef_0[cls]))
                    error[cls, member] = np.mean(np.square(
                        basis.evaluate_to_image_basis(
                            C[cls,member] - coef_0[cls]).asnumpy()))
            return error

        # # Initialization
        # Each iteration densely assigns C,
        #   we can do an initial one now.
        itr = 0
        C = _update_C(C, rotation_angles)
        # Along with computing an initial error, 0 rotation angle
        error = _update_error(C)
        logger.info(f'Alignment iteration: {itr}    max(error): {np.max(error)}')

        # Make an intial guess for angle
        angle_diff = np.full_like(rotation_angles, np.pi) # ""h""
        rotation_angles[:,:] = np.pi
        
        # For Newton method we'll take f = error and just approx f' as a first difference using (f_2-f_1) / h
        # rotation_angles = rotation_angles_last - error / f'
        while itr < max_iter:
            itr += 1
            error_last = error.copy()

            C = _update_C(C, rotation_angles)
            error = _update_error(C)

            logger.info(f'Alignment iteration: {itr}    max(error): {np.max(error)}')

            if np.max(error) < epsilon:
                # convergence criteria met, exit loop
                # rotation_angles holds result
                break
            if not np.all(np.isfinite(error)):
                raise RuntimeError('Alignment has failed')

            first_diff = (error - error_last) / angle_diff
            print('error', error)
            print('error_last', error_last)
            print('angle_diff', angle_diff)
            print('first_diff', first_diff)
            angle_diff = error / first_diff # store for use in next iteration
            rotation_angles -= angle_diff
            # modulo circle
            rotation_angles %= 2*np.pi 

        else:
            logger.error(f"rotation_angles failed to converge in {itr} iterations."
                         f"Last error {error}")

        # Recall that we ignored the rotation angle of the first image,
        #   because it should remain zero.
        rotation_angles = np.pad(rotation_angles, [(0,0),(1,0)], mode='constant')
        
        return rotation_angles
        
                    
                                                    
                                              
    def legacy_align(self, classes, coef):
        # translate some variables between this code and the legacy aspire aspire implementation (just trying to figure out of the old code ran...).
        freqs = self.pca_basis.complex_angular_indices
        coeff = coef.T
        n_im = self.src.n
        n_nbor = self.n_nbor

        # ## COPIED FROM LEGACY CODE:
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

        # gbw, class_refl = np.ceil((classes + 1.0) / n_im).astype("int")
        # gbw, prefer bool
        class_refl = ((classes + 1) // n_im).astype(bool)
        classes[classes >= n_im] = classes[classes >= n_im] - n_im
        # gbw rot[class_refl == 2] = np.mod(rot[class_refl == 2] + 180, 360)
        rot[class_refl] = np.mod(rot[class_refl] + 180, 360)
        rot *= np.pi / 180.0  # gbw, radians
        return classes, class_refl, rot, corr, 0



                     
