import logging

import numpy as np
from numpy.linalg import norm

from aspire.abinitio import CLSymmetryC3C4
from aspire.operators import PolarFT
from aspire.utils import (
    J_conjugate,
    Rotation,
    all_pairs,
    anorm,
    best_rank1_approximation,
    cyclic_rotations,
    tqdm,
    trange,
)
from aspire.utils.random import Random, randn

logger = logging.getLogger(__name__)


class CLSymmetryCn(CLSymmetryC3C4):
    """
    Define a class to estimate 3D orientations using common lines methods for molecules with
    Cn cyclic symmetry, with n>4.
    """

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
        equator_threshold=10,
        seed=None,
        mask=True,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with Cn symmetry, n>4.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param symmetry: A string, 'Cn', indicating the symmetry type.
        :param n_rad: The number of points in the radial direction.
        :param n_theta: The number of points in the theta direction.
        :param max_shift: Maximum range for shifts as a proportion of resolution. Default = 0.15.
        :param shift_step: Resolution of shift estimation in pixels. Default = 1 pixel.
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        :param degree_res: Degree resolution for estimating in-plane rotations.
        :param n_points_sphere: The number of candidate rotations used to estimate viewing directions.
        :param equator_threshold: Threshold for removing candidate rotations within `equator_threshold`
            degrees of being an equator image. Default is 10 degrees.
        :param seed: Optional seed for RNG.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """

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
            mask=mask,
        )

        self.n_points_sphere = n_points_sphere
        self.equator_threshold = equator_threshold

    def _check_symmetry(self, symmetry):
        if symmetry is None:
            raise NotImplementedError(
                "Symmetry type not supplied. Please indicate symmetry."
            )
        else:
            symmetry = symmetry.upper()
            _sym_type = symmetry[0]
            _order = int(symmetry[1:])
            if not (_sym_type == "C" and _order > 4):
                raise NotImplementedError(
                    f"Only Cn symmetry, n > 4, supported. {symmetry} was supplied."
                )
            self.order = _order

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with Cn symmetry, n > 4.

        :return: Array of rotation matrices, size n_imgx3x3.
        """
        super().estimate_rotations()

    def _estimate_relative_viewing_directions(self):
        logger.info(f"Estimating relative viewing directions for {self.n_img} images.")
        pf = self.pf

        # Generate candidate rotation matrices and the common-line and
        # self-common-line indices induced by those rotations.
        Ris_tilde, R_theta_ijs = self.generate_candidate_rots(
            self.n_points_sphere,
            self.equator_threshold,
            self.order,
            self.degree_res,
            self.seed,
        )
        cijs_inds = self._compute_cls_inds(Ris_tilde, R_theta_ijs)
        scls_inds = self._compute_scls_inds(Ris_tilde)
        n_theta_ijs = len(R_theta_ijs)

        # Generate shift phases.
        r_max = pf.shape[-1]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, self.max_shift, self.shift_step
        )
        n_shifts = len(shifts)

        # Reconstruct full polar Fourier for use in correlation.
        pf /= norm(pf, axis=2)[..., np.newaxis]  # Normalize each ray.
        pf_full = PolarFT.half_to_full(pf)

        # Pre-compute shifted pf's.
        pf_shifted = (pf * shift_phases[:, None, None]).swapaxes(0, 1)
        pf_shifted = pf_shifted.reshape(
            (self.n_img, n_shifts * (self.n_theta // 2), r_max)
        )

        # Step 1: pre-calculate the likelihood with respect to the self-common-lines.
        scores_self_corrs = self._scl_likelihood(pf_shifted, pf_full, scls_inds)

        # Step 2: Compute the likelihood for each pair of candidate matrices with respect
        # to the common-lines they induce. Then compute estimates for viis and vijs.
        logger.info("Computing pairwise likelihood.")
        n_vijs = self.n_img * (self.n_img - 1) // 2
        vijs = np.zeros((n_vijs, 3, 3), dtype=self.dtype)
        viis = np.zeros((self.n_img, 3, 3), dtype=self.dtype)
        rots_symm = cyclic_rotations(self.order, self.dtype).matrices

        # List of MeanOuterProductEstimator instances.
        # Used to keep a running mean of J-synchronized estimates for vii.
        mean_est = []
        for _ in range(self.n_img):
            mean_est.append(MeanOuterProductEstimator())

        pairs = all_pairs(self.n_img)
        for ind, (i, j) in enumerate(tqdm(pairs)):
            # Compute correlation.
            corrs_ij = np.real(pf_shifted[i] @ np.conj(pf_full[j]).T)

            # Max out over shifts.
            corrs_ij = np.max(
                np.reshape(corrs_ij, (n_shifts, self.n_theta // 2, self.n_theta)),
                axis=0,
            )

            # Arrange correlation based on common lines induced by candidate rotations.
            corrs = corrs_ij[cijs_inds[..., 0], cijs_inds[..., 1]]
            corrs = np.reshape(
                corrs,
                (
                    self.n_points_sphere,
                    self.n_points_sphere,
                    self.order,
                    n_theta_ijs // self.order,
                ),
            )

            # Take the mean over all symmetry induced common lines.
            corrs = np.mean(corrs, axis=-2)

            # Compute maximum likelihood while taking into consideration both cls and scls.
            # Get indices of optimal candidates for Ri_tilde, Rj_tilde, and R_theta_ij.
            corrs = (
                corrs
                * np.outer(scores_self_corrs[i], scores_self_corrs[j])[..., np.newaxis]
            )
            opt_i, opt_j, opt_ij = np.unravel_index(np.argmax(corrs), corrs.shape)

            # Optimal candidate rotations.
            opt_Ri_tilde = Ris_tilde[opt_i]
            opt_Rj_tilde = Ris_tilde[opt_j]
            opt_R_theta_ij = R_theta_ijs[opt_ij]

            # Compute the estimate of vi*vi.T as given by j.
            vii_j = np.mean(opt_Ri_tilde.T @ rots_symm @ opt_Ri_tilde, axis=0)
            mean_est[i].push(vii_j)

            # Compute the estimate of vj*vj.T as given by i.
            vjj_i = np.mean(opt_Rj_tilde.T @ rots_symm @ opt_Rj_tilde, axis=0)
            mean_est[j].push(vjj_i)

            # Compute the estimate of vi*vj.T.
            vijs[ind] = np.mean(
                opt_Ri_tilde.T @ rots_symm @ opt_R_theta_ij @ opt_Rj_tilde,
                axis=0,
            )

        # There are conflicting methods betweeen the paper and the Matlab code for
        # finding the optimal estimates for viis. The paper suggests using SVD to find
        # the estimate for vii which is closest to rank 1. The Matlab code takes the
        # median over the stack of all estimates for vii. Here we have implemented
        # a method which J-synchronizes the estimates prior to taking the mean.
        # See issue #869 for more details.
        for i in range(self.n_img):
            viis[i] = mean_est[i].synchronized_mean()

        # As we are using a mean to get the estimates, viis, the estimate will not be rank-1
        # So we use SVD to find a close rank-1 approximation.
        viis_rank1 = best_rank1_approximation(viis)

        return vijs, viis_rank1

    def _scl_likelihood(self, pf_shifted, pf_full, scls_inds):
        scores_self_corrs = np.zeros((self.n_img, len(scls_inds)), dtype=self.dtype)
        logger.info("Computing likelihood wrt self common-lines.")
        for i in trange(self.n_img):
            # Compute correlation of pf[i] with itself over all shifts.
            corrs = np.real(pf_shifted[i] @ np.conj(pf_full[i]).T)
            # Reshape to (n_shifts, n_theta//2, n_theta).
            corrs = np.reshape(corrs, (-1, self.n_theta // 2, self.n_theta))
            corrs_cands = np.max(
                (corrs[:, scls_inds[:, :, 0], scls_inds[:, :, 1]]), axis=0
            )

            scores_self_corrs[i] = np.mean(corrs_cands, axis=1)

        return scores_self_corrs

    def _compute_scls_inds(self, Ris_tilde):
        """
        Compute self-common-lines indices induced by candidate rotations.

        :param Ris_tilde: An array of size n_candsx3x3 of candidate rotations.
        :return: An n_cands x (order-1)//2 x 2 array holding the indices of the (order-1)//2
            non-collinear pairs of self-common-lines for each candidate rotation.
        """
        n_scl_pairs = (self.order - 1) // 2
        n_cands = Ris_tilde.shape[0]
        scls_inds = np.zeros((n_cands, n_scl_pairs, 2), dtype=np.uint16)
        rots_symm = cyclic_rotations(self.order, dtype=self.dtype).matrices

        for i_cand, Ri_cand in enumerate(Ris_tilde):
            Riigs = Ri_cand.T @ rots_symm[1 : n_scl_pairs + 1] @ Ri_cand

            c1s, c2s = self.relative_rots_to_cl_indices(Riigs, self.n_theta)

            scls_inds[i_cand, :, 0] = c1s
            scls_inds[i_cand, :, 1] = c2s
        return scls_inds

    def _compute_cls_inds(self, Ris_tilde, R_theta_ijs):
        """
        Compute the common-lines indices induced by the candidate rotations.

        :param Ris_tilde: An array of size n_candsx3x3 of candidate rotations.
        :param R_theta_ijs: An array of size n_theta_ijsx3x3 of inplane rotations.
        :return: An array of size n_cands x n_cands x n_theta_ijs x 2 holding common-lines
            indices induced by the supplied rotations.
        """
        n_points_sphere = self.n_points_sphere
        n_theta_ijs = R_theta_ijs.shape[0]
        cij_inds = np.zeros(
            (n_points_sphere, n_points_sphere, n_theta_ijs, 2), dtype=np.uint16
        )
        logger.info("Computing common-line indices induced by candidate rotations.")
        for i in tqdm(range(n_points_sphere)):
            for j in range(n_points_sphere):
                R_cands = Ris_tilde[i].T @ R_theta_ijs @ Ris_tilde[j]

                c1s, c2s = self.relative_rots_to_cl_indices(R_cands, self.n_theta)

                cij_inds[i, j, :, 0] = c1s
                cij_inds[i, j, :, 1] = c2s
        return cij_inds

    @staticmethod
    def relative_rots_to_cl_indices(relative_rots, n_theta):
        """
        Given a set of relative rotations between pairs of images
        produce the common-line indices for each pair.

        :param relative_rots: The n x 3 x 3 relative rotations between pairs of images.
        :param n_theta: The theta resolution for common-line indices.

        :return: Common-line indices c1s, c2s each length n.
        """
        c1s = np.array((-relative_rots[:, 1, 2], relative_rots[:, 0, 2])).T
        c2s = np.array((relative_rots[:, 2, 1], -relative_rots[:, 2, 0])).T

        c1s = CLSymmetryC3C4.cl_angles_to_ind(c1s, n_theta)
        c2s = CLSymmetryC3C4.cl_angles_to_ind(c2s, n_theta)

        inds = np.where(c1s >= n_theta // 2)
        c1s[inds] -= n_theta // 2
        c2s[inds] += n_theta // 2
        c2s[inds] = np.mod(c2s[inds], n_theta)

        return c1s, c2s

    @staticmethod
    def generate_candidate_rots(n, equator_threshold, order, degree_res, seed):
        """
        Generate random rotations that exclude rotations inducing equator images
        for use as candidates in the CLSymmetryCn algorithm.

        :param n: Number of rotations to generate.
        :param equator_threshold: Angular distance from equator (in degrees).
        :param order: Cyclic order of underlying molecule.
        :param degree_res: Degree resolution for in-plane rotations.
        :param seed: Random seed.

        :returns: Candidate rotations, In-plane rotations
        """

        logger.info("Generating candidate rotations.")
        # Construct candidate rotations, Ris_tilde.
        Ris_tilde = np.zeros((n, 3, 3))
        counter = 0
        with Random(seed):
            while counter < n:
                third_row = randn(3)
                third_row /= anorm(third_row, axes=(-1,))
                Ri_tilde = CLSymmetryC3C4._complete_third_row_to_rot(third_row)

                # Exclude candidates that represent equator images. Equator candidates
                # induce collinear self-common-lines, which always have perfect correlation.
                angle_from_equator = abs(np.arccos(Ri_tilde[2, 2]) - np.pi / 2)
                if angle_from_equator >= equator_threshold * np.pi / 180:
                    Ris_tilde[counter] = Ri_tilde
                    counter += 1

        # Construct all in-plane rotations, R_theta_ijs
        # The number of R_theta_ijs must be divisible by the symmetric order.
        n_theta_ij = 360 - (360 % order)
        theta_ij = np.arange(0, n_theta_ij, degree_res) * np.pi / 180
        R_theta_ijs = Rotation.about_axis("z", theta_ij).matrices

        return Ris_tilde, R_theta_ijs


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
        # Method added Feb, 2023. (Josh Carmichael)
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
