import logging

import numpy as np
from scipy.linalg import eigh, qr

from aspire.image import Image
from aspire.image.xform import NoiseAdder
from aspire.operators import ZeroFilter
from aspire.source import ImageSource
from aspire.utils import acorr, ainner, anorm, ensure, make_symmat, vecmat_to_volmat
from aspire.utils.coor_trans import grid_3d, uniform_random_angles
from aspire.utils.random import Random, rand, randi, randn
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class Simulation(ImageSource):
    def __init__(
        self,
        L=8,
        n=1024,
        vols=None,
        states=None,
        unique_filters=None,
        filter_indices=None,
        offsets=None,
        amplitudes=None,
        dtype=np.float32,
        C=2,
        angles=None,
        seed=0,
        memory=None,
        noise_filter=None,
    ):
        """
        A Cryo-EM simulation
        Other than the base class attributes, it has:

        :param C: The number of distinct volumes
        :param angles: A n-by-3 array of rotation angles
        """
        super().__init__(L=L, n=n, dtype=dtype, memory=memory)

        self.seed = seed

        # We need to keep track of the original resolution we were initialized with,
        # to be able to generate projections of volumes later, when we are asked to supply images.
        self._original_L = L

        if offsets is None:
            offsets = L / 16 * randn(2, n, seed=seed).astype(dtype).T

        if amplitudes is None:
            min_, max_ = 2.0 / 3, 3.0 / 2
            amplitudes = min_ + rand(n, seed=seed).astype(dtype) * (max_ - min_)

        if vols is None:
            self.vols = self._gaussian_blob_vols(L=self.L, C=C)
        else:
            assert isinstance(vols, Volume)
            self.vols = vols

        if self.vols.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" vols.dtype {self.vols.dtype} != self.dtype {self.dtype}."
                " In the future this will raise an error."
            )

        self.C = self.vols.n_vols

        if states is None:
            states = randi(self.C, n, seed=seed)
        self.states = states

        if angles is None:
            angles = uniform_random_angles(n, seed=seed, dtype=self.dtype)
        self.angles = angles

        if unique_filters is None:
            unique_filters = []
        self.unique_filters = unique_filters

        # Create filter indices and fill the metadata based on unique filters
        if unique_filters:
            if filter_indices is None:
                filter_indices = randi(len(unique_filters), n, seed=seed) - 1
            self.filter_indices = filter_indices

        self.offsets = offsets
        self.amplitudes = amplitudes

        self.noise_adder = None
        if noise_filter is not None and not isinstance(noise_filter, ZeroFilter):
            logger.info("Appending a NoiseAdder to generation pipeline")
            self.noise_adder = NoiseAdder(seed=self.seed, noise_filter=noise_filter)

    def _gaussian_blob_vols(self, L=8, C=2, K=16, alpha=1):
        """
        Generate Gaussian blob volumes
        :param L: The size of the volumes
        :param C: The number of volumes to generate
        :param K: The number of blobs
        :param alpha: A scale factor of the blob widths

        :return: A Volume instance containing C Gaussian blob volumes.
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

        with Random(self.seed):
            vols = np.zeros(shape=(C, L, L, L)).astype(self.dtype)
            for k in range(C):
                Q, D, mu = gaussian_blobs(K, alpha)
                vols[k] = self.eval_gaussian_blobs(L, Q, D, mu)
            return Volume(vols)

    def eval_gaussian_blobs(self, L, Q, D, mu):
        g = grid_3d(L, dtype=self.dtype)
        coords = np.array(
            [g["x"].flatten(), g["y"].flatten(), g["z"].flatten()], dtype=self.dtype
        )

        K = Q.shape[-1]
        vol = np.zeros(shape=(1, coords.shape[-1])).astype(self.dtype)

        for k in range(K):
            coords_k = coords - mu[:, k, np.newaxis]
            coords_k = (
                Q[:, :, k] / np.sqrt(np.diag(D[:, :, k])) @ Q[:, :, k].T @ coords_k
            )

            vol += np.exp(-0.5 * np.sum(np.abs(coords_k) ** 2, axis=0))

        vol = np.reshape(vol, g["x"].shape)

        return vol

    def projections(self, start=0, num=np.inf, indices=None):
        """
        Return projections of generated volumes, without applying filters/shifts/amplitudes/noise
        :param start: start index (0-indexed) of the start image to return
        :param num: Number of images to return. If None, *all* images are returned.
        :param indices: A numpy array of image indices. If specified, start and num are ignored.
        :return: An Image instance.
        """
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        im = np.zeros(
            (len(indices), self._original_L, self._original_L), dtype=self.dtype
        )

        states = self.states[indices]
        unique_states = np.unique(states)
        for k in unique_states:
            idx_k = np.where(states == k)[0]
            rot = self.rots[indices[idx_k], :, :]

            im_k = self.vols.project(vol_idx=k - 1, rot_matrices=rot)
            im[idx_k, :, :] = im_k.asnumpy()

        return Image(im)

    def clean_images(self, start=0, num=np.inf, indices=None):
        return self._images(start=start, num=num, indices=indices, enable_noise=False)

    def _images(self, start=0, num=np.inf, indices=None, enable_noise=True):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n), dtype=int)

        im = self.projections(start=start, num=num, indices=indices)

        im = self.eval_filters(im, start=start, num=num, indices=indices)
        im = im.shift(self.offsets[indices, :])

        im *= self.amplitudes[indices].reshape(len(indices), 1, 1).astype(self.dtype)

        if enable_noise and self.noise_adder is not None:
            im = self.noise_adder.forward(im, indices=indices)

        return im

    def vol_coords(self, mean_vol=None, eig_vols=None):
        """
        Coordinates of simulation volumes in a given basis
        :param mean_vol: A mean volume in the form of a Volume Instance (default `mean_true`).
        :param eig_vols: A set of k volumes in a Volume instance (default `eigs`).
        :return:
        """
        if mean_vol is None:
            mean_vol = self.mean_true()
        if eig_vols is None:
            eig_vols = self.eigs()[0]

        assert isinstance(mean_vol, Volume)
        assert isinstance(eig_vols, Volume)

        vols = self.vols - mean_vol  # note, broadcast

        V = vols.to_vec()
        EV = eig_vols.to_vec()

        coords = EV @ V.T

        res = vols - Volume.from_vec(coords.T @ EV)
        res_norms = anorm(res.asnumpy(), (1, 2, 3))
        res_inners = mean_vol.to_vec() @ res.to_vec().T

        return coords.squeeze(), res_norms, res_inners

    def mean_true(self):
        return Volume(np.mean(self.vols, 0))

    def covar_true(self):
        eigs_true, lamdbas_true = self.eigs()
        eigs_true = eigs_true.T.to_vec()

        covar_true = eigs_true.T @ lamdbas_true @ eigs_true
        covar_true = vecmat_to_volmat(covar_true)

        return covar_true

    def eigs(self):
        """
        Eigendecomposition of volume covariance matrix of simulation
        :return: A 2-tuple:
            eigs_true: The eigenvectors of the volume covariance matrix in the form of Volume instance.
            lambdas_true: The eigenvalues of the covariance matrix in the form of a (C-1)-by-(C-1) diagonal matrix.
        """
        C = self.C
        vols_c = self.vols - self.mean_true()

        p = np.ones(C) / C
        # RCOPT, we may be able to do better here if we dig in.
        Q, R = qr(vols_c.to_vec().T, mode="economic")

        # Rank is at most C-1, so remove last vector
        Q = Q[:, :-1]
        R = R[:-1, :]

        w, v = eigh(make_symmat(R @ np.diag(p) @ R.T))
        eigs_true = Volume.from_vec((Q @ v).T)

        # Arrange in descending order (flip column order in eigenvector matrix)
        w = w[::-1]
        eigs_true = eigs_true.flip()

        return eigs_true, np.diag(w)

    # TODO: Too many eval_* methods doing similar things - encapsulate somehow?

    def eval_mean(self, mean_est):
        mean_true = self.mean_true()
        return self.eval_vol(mean_true, mean_est)

    def eval_vol(self, vol_true, vol_est):
        norm_true = anorm(vol_true)

        err = anorm(vol_true - vol_est)
        rel_err = err / norm_true
        # RCOPT
        corr = acorr(vol_true.asnumpy(), vol_est.asnumpy())

        return {"err": err, "rel_err": rel_err, "corr": corr}

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

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def eval_eigs(self, eigs_est, lambdas_est):
        """
        Evaluate covariance eigendecomposition accuracy
        :param eigs_est: The estimated volume eigenvectors in an L-by-L-by-L-by-K array.
        :param lambdas_est: The estimated eigenvalues in a K-by-K diagonal matrix (default `diag(ones(K, 1))`).
        :return:
        """
        eigs_true, lambdas_true = self.eigs()

        B = eigs_est.to_vec() @ eigs_true.to_vec().T
        norm_true = anorm(lambdas_true)
        norm_est = anorm(lambdas_est)

        inner = ainner(B @ lambdas_true, lambdas_est @ B)
        err = np.sqrt(norm_true ** 2 + norm_est ** 2 - 2 * inner)
        rel_err = err / norm_true
        corr = inner / (norm_true * norm_est)

        # TODO: Determine Principal Angles and return as a dict value

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def eval_clustering(self, vol_idx):
        """
        Evaluate clustering estimation
        :param vol_idx: Indexes of the volumes determined (0-indexed)
        :return: Accuracy [0-1] in terms of proportion of correctly assigned labels
        """
        ensure(
            len(vol_idx) == self.n, f"Need {self.n} vol indexes to evaluate clustering"
        )
        # Remember that `states` is 1-indexed while vol_idx is 0-indexed
        correctly_classified = np.sum(self.states - 1 == vol_idx)

        return correctly_classified / self.n

    def eval_coords(self, mean_vol, eig_vols, coords_est):
        """
        Evaluate coordinate estimation
        :param mean_vol: A mean volume in the form of a Volume instance.
        :param eig_vols: A set of eigenvolumes in an Volume instance.
        :param coords_est: The estimated coordinates in the affine space defined centered at `mean_vol` and spanned
            by `eig_vols`.
        :return:
        """
        assert isinstance(mean_vol, Volume)
        assert isinstance(eig_vols, Volume)
        coords_true, res_norms, res_inners = self.vol_coords(mean_vol, eig_vols)

        # 0-indexed states vector
        states = self.states - 1

        coords_true = coords_true[states]
        res_norms = res_norms[states]
        res_inners = res_inners[:, states]

        mean_eigs_inners = (mean_vol.to_vec() @ eig_vols.to_vec().T).item()
        coords_err = coords_true - coords_est

        err = np.hypot(res_norms, coords_err)

        mean_vol_norm2 = anorm(mean_vol) ** 2
        norm_true = np.sqrt(
            coords_true ** 2
            + mean_vol_norm2
            + 2 * res_inners
            + 2 * mean_eigs_inners * coords_true
        )
        norm_true = np.hypot(res_norms, norm_true)

        rel_err = err / norm_true
        inner = (
            mean_vol_norm2
            + mean_eigs_inners * (coords_true + coords_est)
            + coords_true * coords_est
            + res_inners
        )
        norm_est = np.sqrt(
            coords_est ** 2 + mean_vol_norm2 + 2 * mean_eigs_inners * coords_est
        )

        corr = inner / (norm_true * norm_est)

        return {"err": err, "rel_err": rel_err, "corr": corr}
