import logging

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from aspire.abinitio import CLSymmetryC3C4
from aspire.utils import Rotation, anorm, cyclic_rotations
from aspire.utils.random import randn

logger = logging.getLogger(__name__)


class CLSymmetryCn(CLSymmetryC3C4):
    def __init__(
        self,
        src,
        symmetry=None,
        n_rad=None,
        n_theta=None,
        max_shift=0.15,
        shift_step=1,
        epsilon=1e-3,
        max_iters=1000,
        degree_res=1,
        n_points_sphere=500,
        seed=None,
    ):

        super().__init__(
            src,
            symmetry=symmetry,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            epsilon=epsilon,
            max_iters=max_iters,
            degree_res=degree_res,
            seed=seed,
        )

        self.n_points_sphere = n_points_sphere

    def _check_symmetry(self, symmetry):
        if symmetry is None:
            raise NotImplementedError(
                "Symmetry type not supplied. Please indicate C3 or C4 symmetry."
            )
        else:
            symmetry = symmetry.upper()
            if not symmetry[0] == "C":
                raise NotImplementedError(
                    f"Only Cn symmetry supported. {symmetry} was supplied."
                )
            self.order = int(symmetry[1])

    def estimate_relative_viewing_directions_cn(self):
        n_img = self.n_img
        # n_rad = self.n_rad
        n_theta = self.n_theta
        pf = self.pf
        max_shift_1d = self.max_shift
        shift_step = self.shift_step

        # Generate candidate rotation matrices and the common-line and
        # self-common-line indices induced by those rotations.
        Ris_tilde, R_theta_ijs = self.generate_cand_rots()
        # cijs_inds = self.compute_cls_inds(Ris_tilde, R_theta_ijs)
        scls_inds = self.compute_scls_inds(Ris_tilde)

        # Generate shift phases.
        r_max = pf.shape[0]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step
        )
        n_shifts = len(shifts)
        all_shift_phases = shift_phases.T

        # Transpose and reconstruct full polar Fourier for use in correlation.
        pf = pf.T
        pf_full = np.concatenate((pf, np.conj(pf)), axis=1)

        # Step 1: pre-calculate the likelihood with respect to the self-common-lines.
        scores_self_corrs = np.zeros((n_img, n_img), dtype=self.dtype)
        for i in range(n_img):
            pf_i = pf[i]
            pf_full_i = pf_full[i]

            # Generate shifted versions of image.
            pf_i_shifted = np.array(
                [pf_i * shift_phase for shift_phase in all_shift_phases]
            )
            pf_i_shifted = np.reshape(pf_i_shifted, (n_shifts * n_theta // 2, r_max))

            # Normalize each ray.
            pf_full_i /= norm(pf_full_i, axis=1)[..., np.newaxis]
            pf_i_shifted /= norm(pf_i_shifted, axis=1)[..., np.newaxis]

            # Compute correlation.
            corrs = pf_i_shifted @ np.conj(pf_full_i).T
            corrs = np.reshape(corrs, (n_shifts, n_theta // 2, n_theta))
            corrs_cands = np.array(
                [
                    np.max(
                        np.real(corrs[:, scls_inds_cand[:, 0], scls_inds_cand[:, 1]]),
                        axis=0,
                    )
                    for scls_inds_cand in scls_inds
                ]
            )

            scores_self_corrs[i] = np.mean(np.real(corrs_cands), axis=1)
        return scores_self_corrs

    def compute_scls_inds(self, Ri_cands):
        """
        Compute self-common-lines indices induced by candidate rotations.

        :param Ri_cands: An array of size n_candsx3x3 of candidate rotations.
        :return: An n_cands x (order-1)//2 x 2 array holding the indices of the (order-1)//2
            non-collinear pairs of self-common-lines for each candidate rotation.
        """
        order = self.order
        n_theta = self.n_theta
        n_scl_pairs = (order - 1) // 2
        n_cands = Ri_cands.shape[0]
        scls_inds = np.zeros((n_cands, n_scl_pairs, 2), dtype=np.uint16)
        rots_symm = cyclic_rotations(order, dtype=self.dtype).matrices

        for i_cand, Ri_cand in enumerate(Ri_cands):
            Riigs = Ri_cand.T @ rots_symm[1 : n_scl_pairs + 1] @ Ri_cand

            c1s = np.array((-Riigs[:, 1, 2], Riigs[:, 0, 2])).T
            c2s = np.array((-Riigs[:, 2, 1], Riigs[:, 2, 0])).T

            c1s_inds = self.cl_angles_to_ind(c1s, n_theta)
            c2s_inds = self.cl_angles_to_ind(c2s, n_theta)

            inds = np.where(c1s_inds >= (n_theta // 2))
            c1s_inds[inds] -= n_theta // 2
            c2s_inds[inds] += n_theta // 2
            c2s_inds[inds] = np.mod(c2s_inds[inds], n_theta)

            scls_inds[i_cand, :, 0] = c1s_inds
            scls_inds[i_cand, :, 1] = c2s_inds
        return scls_inds

    # TODO: cache
    def compute_cls_inds(self, Ris_tilde, R_theta_ijs):
        n_theta = self.n_theta
        n_points_sphere = self.n_points_sphere
        n_theta_ijs = R_theta_ijs.shape[0]
        cij_inds = np.zeros(
            (n_points_sphere, n_points_sphere, n_theta_ijs, 2), dtype=np.uint16
        )

        with tqdm(total=n_points_sphere) as pbar:
            for i in range(n_points_sphere):
                for j in range(n_points_sphere):
                    R_cands = Ris_tilde[i].T @ R_theta_ijs @ Ris_tilde[j]

                    c1s = np.array((-R_cands[:, 1, 2], R_cands[:, 0, 2])).T
                    c2s = np.array((-R_cands[:, 2, 1], R_cands[:, 2, 0])).T

                    c1s = self.cl_angles_to_ind(c1s, n_theta)
                    c2s = self.cl_angles_to_ind(c2s, n_theta)

                    inds = np.where(c1s >= n_theta // 2)
                    c1s[inds] -= n_theta // 2
                    c2s[inds] += n_theta // 2
                    c2s[inds] = np.mod(c2s[inds], n_theta)

                    cij_inds[i, j, :, 0] = c1s
                    cij_inds[i, j, :, 1] = c2s
                pbar.update()
        return cij_inds

    def generate_cand_rots(self):
        # Construct candidate rotations, Ris_tilde.
        vis = self.generate_cand_rots_third_rows(self.n_points_sphere)
        Ris_tilde = np.array([self._complete_third_row_to_rot(vi) for vi in vis])

        # Construct all in-plane rotations, R_theta_ijs
        theta_ij = np.arange(0, 360, self.degree_res) * np.pi / 180
        R_theta_ijs = Rotation.about_axis("z", theta_ij).matrices

        return Ris_tilde, R_theta_ijs

    def generate_cand_rots_third_rows(self, legacy=True):
        n_points_sphere = self.n_points_sphere

        if legacy:
            # Genereate random points on the sphere
            third_rows = randn(n_points_sphere, 3)
            third_rows /= anorm(third_rows, axes=(-1,))[:, np.newaxis]
        else:
            # Use Fibonocci sphere points
            third_rows = np.zeros((n_points_sphere, 3))
            phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

            for i in range(n_points_sphere):
                y = 1 - (i / float(n_points_sphere - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y

                theta = phi * i  # golden angle increment
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius

                third_rows[i] = x, y, z

        return third_rows
