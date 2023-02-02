import logging

import numpy as np
from numpy.linalg import norm

from aspire.abinitio import CLSymmetryC3C4
from aspire.utils import J_conjugate, Rotation, anorm, cyclic_rotations, tqdm, trange
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
                "Symmetry type not supplied. Please indicate symmetry."
            )
        else:
            symmetry = symmetry.upper()
            if not symmetry[0] == "C":
                raise NotImplementedError(
                    f"Only Cn symmetry supported. {symmetry} was supplied."
                )
            self.order = int(symmetry[1:])

    def _estimate_relative_viewing_directions(self):
        n_img = self.n_img
        n_theta = self.n_theta
        pf = self.pf
        max_shift_1d = self.max_shift
        shift_step = self.shift_step

        # Generate candidate rotation matrices and the common-line and
        # self-common-line indices induced by those rotations.
        Ris_tilde, R_theta_ijs = self._generate_cand_rots()
        cijs_inds = self._compute_cls_inds(Ris_tilde, R_theta_ijs)
        scls_inds = self._compute_scls_inds(Ris_tilde)
        n_cands = len(Ris_tilde)
        n_theta_ijs = len(R_theta_ijs)

        # Generate shift phases.
        r_max = pf.shape[-1]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step
        )
        n_shifts = len(shifts)
        all_shift_phases = shift_phases.T

        # Reconstruct full polar Fourier for use in correlation.
        pf /= norm(pf, axis=2)[..., np.newaxis]  # Normalize each ray.
        pf_full = np.concatenate((pf, np.conj(pf)), axis=1)

        # Step 1: pre-calculate the likelihood with respect to the self-common-lines.
        scores_self_corrs = np.zeros((n_img, n_cands), dtype=self.dtype)
        logger.info("Computing likelihood wrt self common-lines.")
        for i in trange(n_img):
            pf_i = pf[i]
            pf_full_i = pf_full[i]

            # Generate shifted versions of image.
            pf_i_shifted = np.array(
                [pf_i * shift_phase for shift_phase in all_shift_phases]
            )
            pf_i_shifted = np.reshape(pf_i_shifted, (n_shifts * n_theta // 2, r_max))

            # Ignore dc-component.
            pf_full_i[:, 0] = 0
            pf_i_shifted[:, 0] = 0

            # Compute correlation of pf_i with itself over all shifts.
            corrs = pf_i_shifted @ np.conj(pf_full_i).T
            corrs = np.reshape(corrs, (n_shifts, n_theta // 2, n_theta))
            corrs_cands = np.max(
                np.real(corrs[:, scls_inds[:, :, 0], scls_inds[:, :, 1]]), axis=0
            )

            scores_self_corrs[i] = np.mean(np.real(corrs_cands), axis=1)

        # Remove candidates that are equator images. Equator candidates induce collinear
        # self common-lines, which always have perfect correlation.
        # TODO: Should the threshold be parameter-dependent instead of set to 10 degrees?
        cii_equators_inds = np.array(
            [
                ind
                for (ind, Ri_tilde) in enumerate(Ris_tilde)
                if abs(np.arccos(Ri_tilde[2, 2]) - np.pi / 2) < 10 * np.pi / 180
            ]
        )
        scores_self_corrs[:, cii_equators_inds] = 0

        # Step 2: Compute the likelihood for each pair of candidate matrices with respect
        # to the common-lines they induce.
        logger.info("Computing pairwise likelihood.")
        n_vijs = n_img * (n_img - 1) // 2
        vijs = np.zeros((n_vijs, 3, 3), dtype=self.dtype)
        viis_sync = np.zeros((n_img, 3, 3), dtype=self.dtype)
        rots_symm = cyclic_rotations(self.order, self.dtype).matrices
        c = 0

        # List of MeanOuterProductEstimator instances.
        # Used to keep a running mean of J-synchronized estimates for vii.
        mean_est = []
        for _ in range(n_img):
            mean_est.append(MeanOuterProductEstimator())

        with tqdm(total=n_vijs) as pbar:
            for i in range(n_img):
                pf_i = pf[i]

                # Generate shifted versions of the images.
                pf_i_shifted = np.array(
                    [pf_i * shift_phase for shift_phase in all_shift_phases]
                )
                pf_i_shifted = np.reshape(
                    pf_i_shifted, (n_shifts * n_theta // 2, r_max)
                )

                # Ignore dc-component.
                pf_i_shifted[:, 0] = 0

                for j in range(i + 1, n_img):
                    pf_full_j = pf_full[j]

                    # Ignore dc-component.
                    pf_full_j[:, 0] = 0

                    # Compute correlation.
                    corrs_ij = pf_i_shifted @ np.conj(pf_full_j).T

                    # Max out over shifts.
                    corrs_ij = np.max(
                        np.reshape(
                            np.real(corrs_ij), (n_shifts, n_theta // 2, n_theta)
                        ),
                        axis=0,
                    )

                    # Arrange correlation based on common lines induced by candidate rotations.
                    corrs = corrs_ij[cijs_inds[..., 0], cijs_inds[..., 1]]
                    corrs = np.reshape(
                        corrs, (-1, self.order, n_theta_ijs // self.order)
                    )
                    # Take the mean over all symmetric common lines.
                    corrs = np.mean(corrs, axis=1)
                    corrs = np.reshape(
                        corrs,
                        (
                            self.n_points_sphere,
                            self.n_points_sphere,
                            n_theta_ijs // self.order,
                        ),
                    )

                    # Self common-lines are invariant to n_theta_ijs (i.e., in-plane rotation angles) so max them out.
                    opt_theta_ij_ind_per_sphere_points = np.argmax(corrs, axis=-1)
                    corrs = np.max(corrs, axis=-1)

                    # Maximum likelihood while taking into consideration both cls and scls.
                    corrs = corrs * np.outer(scores_self_corrs[i], scores_self_corrs[j])

                    # Extract the optimal candidates.
                    opt_sphere_i, opt_sphere_j = np.unravel_index(
                        np.argmax(corrs), corrs.shape
                    )
                    opt_theta_ij = opt_theta_ij_ind_per_sphere_points[
                        opt_sphere_i, opt_sphere_j
                    ]

                    opt_Ri_tilde = Ris_tilde[opt_sphere_i]
                    opt_Rj_tilde = Ris_tilde[opt_sphere_j]
                    opt_R_theta_ij = R_theta_ijs[opt_theta_ij]

                    # Compute the estimate of vi*vi.T as given by j.
                    vii_j = np.mean(opt_Ri_tilde.T @ rots_symm @ opt_Ri_tilde, axis=0)
                    mean_est[i].push(vii_j)

                    # Compute the estimate of vj*vj.T as given by i.
                    vjj_i = np.mean(opt_Rj_tilde.T @ rots_symm @ opt_Rj_tilde, axis=0)
                    mean_est[j].push(vjj_i)

                    # Compute the estimate of vi*vj.T.
                    vijs[c] = np.mean(
                        opt_Ri_tilde.T @ rots_symm @ opt_R_theta_ij @ opt_Rj_tilde,
                        axis=0,
                    )

                    c += 1
                    pbar.update()

                viis_sync[i] = mean_est[i].synchronized_mean()

        return vijs, viis_sync

    def _compute_scls_inds(self, Ri_cands):
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
            c2s = np.array((Riigs[:, 2, 1], -Riigs[:, 2, 0])).T

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
    def _compute_cls_inds(self, Ris_tilde, R_theta_ijs):
        """
        Compute the common-lines indices induced by the candidate rotations.

        :param Ris_tilde: An array of size n_candsx3x3 of candidate rotations.
        :param R_theta_ijs: An array of size n_theta_ijsx3x3 of inplane rotations.
        :return: An array of size n_cands x n_cands x n_theta_ijs x 2 holding common-lines
            indices induced by the supplied rotations.
        """
        n_theta = self.n_theta
        n_points_sphere = self.n_points_sphere
        n_theta_ijs = R_theta_ijs.shape[0]
        cij_inds = np.zeros(
            (n_points_sphere, n_points_sphere, n_theta_ijs, 2), dtype=np.uint16
        )
        logger.info("Computing common-line indices induced by candidate rotations.")
        with tqdm(total=n_points_sphere) as pbar:
            for i in range(n_points_sphere):
                for j in range(n_points_sphere):
                    R_cands = Ris_tilde[i].T @ R_theta_ijs @ Ris_tilde[j]

                    c1s = np.array((-R_cands[:, 1, 2], R_cands[:, 0, 2])).T
                    c2s = np.array((R_cands[:, 2, 1], -R_cands[:, 2, 0])).T

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

    def _generate_cand_rots(self):
        logger.info("Generating candidate rotations.")
        # Construct candidate rotations, Ris_tilde.
        vis = self._generate_cand_rots_third_rows()
        Ris_tilde = np.array([self._complete_third_row_to_rot(vi) for vi in vis])

        # Construct all in-plane rotations, R_theta_ijs
        # The number of R_theta_ijs must be divisible by the symmetric order.
        n_theta_ij = 360 - (360 % self.order)
        theta_ij = np.arange(0, n_theta_ij, self.degree_res) * np.pi / 180
        R_theta_ijs = Rotation.about_axis("z", theta_ij).matrices

        return Ris_tilde, R_theta_ijs

    def _generate_cand_rots_third_rows(self, legacy=True):
        n_points_sphere = self.n_points_sphere
        if legacy:
            # Genereate random points on the sphere
            third_rows = randn(n_points_sphere, 3, seed=self.seed)
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


class MeanOuterProductEstimator:
    """
    Incrementally accumulate outer product entries of unknown conjugation.
    """

    # These arrays are small enough to just use doubles.
    # Then we can probably avoid numerical summing concerns without precomputing denom
    dtype = np.float64

    def __init__(self):
        # Create storage for J-synchronized estimates.
        self.V_estimates_sync = np.zeros((3, 3), dtype=self.dtype)
        self.count = 0

    def push(self, V):
        """
        Given V, accumulate entries into a running sum of J-synchronized entries.
        """

        # Accumulate synchronized entries to compute synchronized mean.
        if self.count == 0:
            self.V_estimates_sync += V
        else:
            if (np.sign(V) == np.sign(self.V_estimates_sync)).all():
                self.V_estimates_sync += V
            else:
                self.V_estimates_sync += J_conjugate(V)

        self.count += 1

    def synchronized_mean(self):
        """
        Calculate the mean of synchronized outer product estimates.
        """
        return self.V_estimates_sync / self.count
