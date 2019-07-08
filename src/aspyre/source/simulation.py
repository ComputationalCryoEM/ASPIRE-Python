import numpy as np
from scipy.linalg import qr, eigh

from aspyre.source import ImageSource
from aspyre.imaging import im_translate
from aspyre.imaging.threed import vol_project
from aspyre.utils import ensure
from aspyre.utils.matlab_compat import Random, m_reshape
from aspyre.utils.math import grid_3d, angles_to_rots
from aspyre.utils.matlab_compat import rand, randi, randn
from aspyre.utils.matrix import anorm, acorr, ainner, vol_to_vec, vec_to_vol, vecmat_to_volmat, make_symmat


class Simulation(ImageSource):
    def __init__(self, L=8, n=1024, states=None, filters=None, offsets=None, amplitudes=None, dtype='single', C=2,
                 rots=None):
        """
        A Cryo-EM simulation
        Other than the base class attributes, it has:

        :param C: The no. of distinct volumes
        :param rots: A 3-by-3-by-n array of rotation matrices corresponding to viewing directions
        """

        offsets = offsets or L / 16 * randn(2, n, seed=0)
        if amplitudes is None:
            min_, max_ = 2./3, 3./2
            amplitudes = min_ + rand(n, seed=0) * (max_ - min_)
        states = states or randi(C, n, seed=0)
        rots = rots or self._uniform_random_rotations(n, seed=0)

        super().__init__(
            L=L,
            n=n,
            states=states,
            filters=filters,
            offsets=offsets,
            amplitudes=amplitudes,
            rots=rots,
            dtype=dtype
        )

        self.C = C
        self.vols = self._gaussian_blob_vols(L=self.L, C=self.C, seed=0)

    def _uniform_random_rotations(self, n, seed=None):
        with Random(seed):
            angles = np.vstack((
                np.random.random((1, n)) * 2 * np.pi,
                np.arccos(2 * np.random.random((1, n)) - 1),
                np.random.random((1, n)) * 2 * np.pi
            ))
        return angles_to_rots(angles)

    def _gaussian_blob_vols(self, L=8, C=2, K=16, alpha=1, seed=None):
        """
        Generate Gaussian blob volumes
        :param L: The size of the volumes
        :param C: The number of volumes to generate
        :param K: The number of blobs
        :param alpha: A scale factor of the blob widths

        :return: A volume array of size L x L x L x C containing the C Gaussian blob volumes.
        """

        def gaussian_blobs(K, alpha):
            Q = np.zeros(shape=(3, 3, K)).astype(self.dtype)
            D = np.zeros(shape=(3, 3, K)).astype(self.dtype)
            mu = np.zeros(shape=(3, K)).astype(self.dtype)

            for k in range(K):
                V = randn(3, 3).astype(self.dtype) / np.sqrt(3)
                Q[:, :, k] = qr(V)[0]
                D[:, :, k] = alpha ** 2 / 16 * np.diag(np.sum(abs(V) ** 2, axis=0))
                mu[:, k] = 0.5 * randn(3) / np.sqrt(3)

            return Q, D, mu

        with Random(seed):
            vols = np.zeros(shape=(L, L, L, C)).astype(self.dtype)
            for k in range(C):
                Q, D, mu = gaussian_blobs(K, alpha)
                vols[:, :, :, k] = self.eval_gaussian_blobs(L, Q, D, mu)
            return vols

    def eval_gaussian_blobs(self, L, Q, D, mu):
        g = grid_3d(L)
        # Migration Note - Matlab (:) flattens in column-major order, so specify 'F' with flatten()
        coords = np.array([g['x'].flatten('F'), g['y'].flatten('F'), g['z'].flatten('F')])

        K = Q.shape[-1]
        vol = np.zeros(shape=(1, coords.shape[-1])).astype(self.dtype)

        for k in range(K):
            coords_k = coords - mu[:, k, np.newaxis]
            coords_k = Q[:, :, k] / np.sqrt(np.diag(D[:, :, k])) @ Q[:, :, k].T @ coords_k

            vol += np.exp(-0.5 * np.sum(np.abs(coords_k)**2, axis=0))

        vol = m_reshape(vol, g['x'].shape)

        return vol

    def clean_images(self, start=0, num=None):
        all_idx = np.arange(start, min(start+num, self.n))
        im = np.zeros((self.L, self.L, len(all_idx)))

        unique_states = np.unique(self.states[all_idx])
        for k in unique_states:
            vol_k = self.vols[:, :, :, k-1]
            idx_k = np.where(self.states[all_idx] == k)[0]
            rot = self.rots[:, :, all_idx[idx_k]]

            im_k = vol_project(vol_k, rot)
            im[:, :, idx_k] = im_k
        return im

    def _images(self, start=0, num=None):
        """
        Return images from the source.
        :param start: start index (0-indexed) of the start image to return
        :param num: No. of images to return. If None, *all* images are returned.
        :return: An ndarray of shape (L, L, num) where L = min(L, max_L), L being the size of each image.
        """
        end = self.n
        if num is not None:
            end = min(start + num, self.n)
        all_idx = np.arange(start, end)

        im = self.clean_images(start, num)
        im = self.filters(im, start, num)

        # Translations
        im = im_translate(im, self.offsets[:, all_idx])

        # Amplitudes
        im *= np.broadcast_to(self.amplitudes[all_idx], (self.L, self.L, len(all_idx)))

        return im

    def vol_coords(self, mean_vol=None, eig_vols=None):
        """
        Coordinates of simulation volumes in a given basis
        :param mean_vol: A mean volume in the form of an L-by-L-by-L array (default `mean_true`).
        :param eig_vols: A set of eigenvolumes in an L-by-L-by-L-by-K array (default `eigs`).
        :return:
        """
        if mean_vol is None:
            mean_vol = self.mean_true()
        if eig_vols is None:
            eig_vols = self.eigs()[0]

        vols = self.vols - np.expand_dims(mean_vol, 3)
        coords = vol_to_vec(eig_vols).T @ vol_to_vec(vols)
        res = vols - vec_to_vol(vol_to_vec(eig_vols) @ coords)
        res_norms = np.diag(anorm(res, (0, 1, 2)))
        res_inners = vol_to_vec(mean_vol).T @ vol_to_vec(res)

        return coords.squeeze(), res_norms, res_inners

    def mean_true(self):
        return np.mean(self.vols, 3)

    def covar_true(self):
        eigs_true, lamdbas_true = self.eigs()
        eigs_true = vol_to_vec(eigs_true)
        covar_true = eigs_true @ lamdbas_true @ eigs_true.T
        covar_true = vecmat_to_volmat(covar_true)

        return covar_true

    def eigs(self):
        """
        Eigendecomposition of volume covariance matrix of simulation
        :return: A 2-tuple:
            eigs_true: The eigenvectors of the volume covariance matrix in the form of an L-by-L-by-L-by-(C-1) array,
            where C is the number of distinct states in the simulation
            lambdas_true: The eigenvalues of the covariance matrix in the form of a (C-1)-by-(C-1) diagonal matrix.
        """
        C = self.C
        vols_c = self.vols - np.expand_dims(self.mean_true(), 3)
        p = np.ones(C) / C
        vols_c = vol_to_vec(vols_c)
        Q, R = qr(vols_c, mode='economic')

        # Rank is at most C-1, so remove last vector
        Q = Q[:, :-1]
        R = R[:-1, :]

        w, v = eigh(make_symmat(R @ np.diag(p) @ R.T))
        eigs_true = vec_to_vol(Q @ v)

        # Arrange in descending order (flip column order in eigenvector matrix)
        w = w[::-1]
        eigs_true = np.flip(eigs_true, axis=-1)

        return eigs_true, np.diag(w)

    # TODO: Too many eval_* methods doing similar things - encapsulate somehow?

    def eval_mean(self, mean_est):
        mean_true = self.mean_true()
        return self.eval_vol(mean_true, mean_est)

    def eval_vol(self, vol_true, vol_est):
        norm_true = anorm(vol_true)

        err = anorm(vol_true - vol_est)
        rel_err = err / norm_true
        corr = acorr(vol_true, vol_est)

        return {
            'err': err,
            'rel_err': rel_err,
            'corr': corr
        }

    def eval_covar(self, covar_est):
        covar_true = self.covar_true()
        return self.eval_volmat(covar_true, covar_est)

    def eval_volmat(self, volmat_true, volmat_est):
        """
        Evaluate volume matrix estimation accuracy
        :param volmat_true: The true volume matrices in the form of an L-by-L-by-L-by-L-by-L-by-L-by-K array.
        :param volmat_est: The estimated volume matrices in the same form.
        :return:
        """
        norm_true = anorm(volmat_true)

        err = anorm(volmat_true - volmat_est)
        rel_err = err / norm_true
        corr = acorr(volmat_true, volmat_est)

        return {
            'err': err,
            'rel_err': rel_err,
            'corr': corr
        }

    def eval_eigs(self, eigs_est, lambdas_est):
        """
        Evaluate covariance eigendecomposition accuracy
        :param eigs_est: The estimated volume eigenvectors in an L-by-L-by-L-by-K array.
        :param lambdas_est: The estimated eigenvalues in a K-by-K diagonal matrix (default `diag(ones(K, 1))`).
        :return:
        """
        eigs_true, lambdas_true = self.eigs()

        B = vol_to_vec(eigs_est).T @ vol_to_vec(eigs_true)
        norm_true = anorm(lambdas_true)
        norm_est = anorm(lambdas_est)

        inner = ainner(B @ lambdas_true, lambdas_est @ B)
        err = np.sqrt(norm_true ** 2 + norm_est ** 2 - 2 * inner)
        rel_err = err / norm_true
        corr = inner / (norm_true * norm_est)

        # TODO: Determine Principal Angles and return as a dict value

        return {
            'err': err,
            'rel_err': rel_err,
            'corr': corr
        }

    def eval_clustering(self, vol_idx):
        """
        Evaluate clustering estimation
        :param vol_idx: Indexes of the volumes determined (0-indexed)
        :return: Accuracy [0-1] in terms of proportion of correctly assigned labels
        """
        ensure(len(vol_idx) == self.n, f'Need {self.n} vol indexes to evaluate clustering')
        # Remember that `states` is 1-indexed while vol_idx is 0-indexed
        correctly_classified = np.sum(self.states-1 == vol_idx)

        return correctly_classified / self.n

    def eval_coords(self, mean_vol, eig_vols, coords_est):
        """
        Evaluate coordinate estimation
        :param mean_vol: A mean volume in the form of an L-by-L-by-L array.
        :param eig_vols: A set of eigenvolumes in an L-by-L-by-L-by-K array.
        :param coords_est: The estimated coordinates in the affine space defined centered at `mean_vol` and spanned
            by `eig_vols`.
        :return:
        """
        coords_true, res_norms, res_inners = self.vol_coords(mean_vol, eig_vols)

        # 0-indexed states vector
        states = self.states - 1

        coords_true = coords_true[states]
        res_norms = res_norms[states]
        res_inners = res_inners[states]

        mean_eigs_inners = np.asscalar(vol_to_vec(mean_vol).T @ vol_to_vec(eig_vols))
        coords_err = coords_true - coords_est

        err = np.hypot(res_norms, coords_err)
        mean_vol_norm2 = anorm(mean_vol) ** 2
        norm_true = np.sqrt(coords_true**2 + mean_vol_norm2 + 2*res_inners + 2*mean_eigs_inners * coords_true)
        norm_true = np.hypot(res_norms, norm_true)

        rel_err = err / norm_true
        inner = mean_vol_norm2 + mean_eigs_inners * (coords_true + coords_est) + coords_true * coords_est + res_inners
        norm_est = np.sqrt(coords_est**2 + mean_vol_norm2 + 2*mean_eigs_inners*coords_est)

        corr = inner / (norm_true * norm_est)

        return {
            'err': err,
            'rel_err': rel_err,
            'corr': corr
        }
