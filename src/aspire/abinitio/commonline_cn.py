import logging

import numpy as np
from numpy.linalg import eigh, norm, svd
from tqdm import tqdm

from aspire.abinitio import CLSymmetryC3C4, SyncVotingMixin
from aspire.utils import (
    J_conjugate,
    Rotation,
    all_pairs,
    all_triplets,
    anorm,
    cyclic_rotations,
    pairs_to_linear,
)
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

    #    def estimate_relative_viewing_directions_cn(self):

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
