import logging

import numpy as np
from numpy.linalg import norm

from aspire.abinitio import CLOrient3D
from aspire.operators import PolarFT
from aspire.utils import Rotation, trange

logger = logging.getLogger(__name__)


class CLSymmetryD2(CLOrient3D):
    """
    Define a class to estimate 3D orientations using common lines methods for
    molecules with D2 (dihedral) symmetry.

    The related publications are:
    E. Rosen and Y. Shkolnisky,
    Common lines ab-initio reconstruction of D2-symmetric molecules,
    SIAM Journal on Imaging Sciences, volume 13-4, p. 1898-1994, 2020
    """

    def __init__(
        self,
        src,
        n_rad=None,
        n_theta=None,
        max_shift=0.15,
        shift_step=1,
        grid_res=1200,
        inplane_res=5,
        eq_min_dist=7,
        seed=None,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with D2 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction of Fourier image.
        :param n_theta: The number of points in the theta direction of Fourier image.
        :param max_shift: Maximum range for shifts as a proportion of resolution. Default = 0.15.
        :param shift_step: Resolution of shift estimation in pixels. Default = 1 pixel.
        :param grid_res: Number of sampling points on sphere for projetion directions.
            These are generated using the Saaf - Kuijlaars algorithm. Default value is 1200.
        :param inplane_res: The sampling resolution of in-plane rotations for each
            projetion direction. Default value is 5.
        :param eq_min_dist: Width of strip around equator projection directions from
            which we DO NOT sample directions. Default value is 7.
        :param seed: Optional seed for RNG.
        """

        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
        )

        self.grid_res = grid_res
        self.inplane_res = inplane_res
        self.n_inplane_rots = int(360 / self.inplane_res)
        self.eq_min_dist = eq_min_dist
        self.seed = seed
        self._generate_gs()

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with D2 symmetry.

        :return: Array of rotation matrices, size n_imgx3x3.
        """
        self.compute_shifted_pf()
        self.generate_lookup_data()
        self.generate_scl_lookup_data()
        self.compute_scl_scores()

    def compute_shifted_pf(self):
        pf = self.pf

        # Generate shift phases.
        r_max = pf.shape[-1]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, self.max_shift, self.shift_step
        )
        self.n_shifts = len(shifts)

        # Reconstruct full polar Fourier for use in correlation.
        pf /= norm(pf, axis=2)[..., np.newaxis]  # Normalize each ray.
        self.pf_full = PolarFT.half_to_full(pf)

        # Pre-compute shifted pf's.
        pf_shifted = (pf * shift_phases[:, None, None]).swapaxes(0, 1)
        self.pf_shifted = pf_shifted.reshape(
            (self.n_img, self.n_shifts * (self.n_theta // 2), r_max)
        )

    def compute_cl_scores(self):
        """
        Run common lines Maximum likelihood procedure for a D2 molecule, to find
        the set of rotations Ri^TgkRj, k=1,2,3,4 for each pair of images i and j.
        """

    def compute_scl_scores(self):
        """
        Compute correlations for self-commonline candidates.
        """
        n_img = self.n_img
        n_theta = self.n_theta
        max_shift_1d = self.max_shift
        shift_step = self.shift_step
        n_eq = len(self.non_tv_eq_idx)
        n_inplane = self.n_inplane_rots

        # Run ML in parallel
        scl_matrix = np.concatenate((self.scl_idx_1, self.scl_idx_2))
        M = len(scl_matrix) // 3
        corrs_out = np.zeros((n_img, M), dtype=self.dtype)
        scl_idx = scl_matrix.reshape(M, 3)

        # Get non-equator indices to use with corrs matrix.
        non_eq_lin_idx = self.non_eq_idx.flatten()
        n_non_eq = len(non_eq_lin_idx)
        non_eq_idx = np.unravel_index(
            scl_idx[non_eq_lin_idx].flatten(), (n_theta // 2, n_theta)
        )

        for i in trange(n_img):
            pf_full_i = self.pf_full[i]
            pf_i_shifted = self.pf_shifted[i]

            # Compute max correlation over all shifts.
            corrs = np.real(pf_i_shifted @ np.conj(pf_full_i).T)
            corrs = np.reshape(corrs, (self.n_shifts, n_theta // 2, n_theta))
            corrs = np.max(corrs, axis=0)

            # Map correlations to probabilities (in the spirit of Maximum Likelihood).
            corrs = 0.5 * (corrs + 1)

            # Compute equator measures.
            eq_measures = self.all_eq_measures(corrs)

            # Handle the cases: Non-equator, Non-top-view equator, and Top view images.
            # 1. Non-equators: just take product of probabilities.
            prod_corrs = np.prod(corrs[non_eq_idx].reshape(n_non_eq, 3), axis=1)
            corrs_out[i, non_eq_lin_idx] = prod_corrs

            # 2. Non-topview equators: adjust scores by eq_measures
            for eq_idx in range(n_eq):
                for j in range(n_inplane):
                    # Take the correlations for the self common line candidate of the
                    # "equator rotation" `eq_idx` with respect to image i, and
                    # multiply by all scores from the function eq_measures (see
                    # documentation inside the function ). Then take maximum over
                    # all the scores.
                    scl_idx_list = np.unravel_index(
                        self.scl_idx_lists[0, eq_idx, j], (n_theta // 2, n_theta)
                    )
                    true_scls_corrs = corrs[scl_idx_list]
                    scls_cand_idx = self.scl_idx_lists[1, eq_idx, j]
                    eq_measures_j = eq_measures[scls_cand_idx]
                    measures_agg = true_scls_corrs * eq_measures_j
                    k = self.non_tv_eq_idx[eq_idx]
                    corrs_out[i, k * n_inplane + j] = np.max(measures_agg)

        self.scls_scores = corrs_out

    def all_eq_measures(self, corrs):
        """
        Compute a measure of how much an image from data is close to be an equator.
        """
        # First compute the eq measure (corrs(scl-k,scl+k) for k=1:90)
        # An eqautor image of a D2 molecule has the following property: If t_i is
        # the angle of one of the rays of the self common line then all the pairs of
        # rays of the form (t_i-k,t_i+k) for k=1:90 are identical. For each t_i we
        # average over correlations between the lines (t_i-k,t_i+k) for k=1:90
        # to measure the likelihood that the image is an equator and the ray (line)
        # with angle t_i is a self common line.
        # (This first loop can be done once outside this function and then pass
        # idx as an argument).
        idx = np.zeros((180, 90, 2))
        idx_1 = np.mod(np.vstack((-np.arange(1, 91), np.arange(1, 91))), 360)
        idx[0, :, :] = idx_1.T
        for k in range(1, 180):
            idx[k, :, :] = np.mod(idx_1.T + k, 360)
        idx = np.mod(idx, 360)

        idx_1 = idx[:, :, 0].flatten()
        idx_2 = idx[:, :, 1].flatten()

        # Make all Ri coordinates < 180 and compute linear indices for corrrelations
        bigger_than_180 = idx_1 >= 180
        idx_1[bigger_than_180] = idx_1[bigger_than_180] - 180
        idx_2[bigger_than_180] = (idx_2[bigger_than_180] + 180) % 360

        # Compute correlations.
        eq_corrs = corrs[idx_1.astype(int), idx_2.astype(int)]
        eq_corrs = eq_corrs.reshape(180, 90)
        corrs_mean = np.mean(eq_corrs, axis=1)

        # Now compute correlations for normals to scls.
        # An eqautor image of a D2 molecule has the additional following property:
        # The normal line to a self common line in 2D Fourier plane is real valued
        # and both of its rays have identical values. We use the correlation
        # between one Fourier ray of the normal to a self common line candidate t_i
        # with its anti-podal as an additional way to measure if the image is an
        # equator and t_i+0.5*pi is the normal to its self common line.
        r = 2

        normal_2_scl_idx = np.zeros((180, 2 * r + 1))
        normal_2_scl_idx_1 = np.mod(180 - np.arange(90 - r, 90 + r + 1), 360)
        normal_2_scl_idx[0, :] = normal_2_scl_idx_1
        for k in range(1, 180):
            normal_2_scl_idx[k, :] = np.mod(normal_2_scl_idx_1 + k, 360)

        # Make all Ri coordinates <=180 and compute linear indices for corrrelations
        bigger_than_180 = normal_2_scl_idx >= 180
        normal_2_scl_idx[bigger_than_180] = normal_2_scl_idx[bigger_than_180] - 180

        # Compute correlations for normals.
        normal_2_scl_idx = normal_2_scl_idx.flatten()
        normal_corrs = corrs[
            normal_2_scl_idx.astype(int), normal_2_scl_idx.astype(int) + 180
        ]
        normal_corrs = normal_corrs.reshape(180, 2 * r + 1)
        normal_corrs_max = np.max(normal_corrs, axis=1)

        return corrs_mean * normal_corrs_max

    def generate_lookup_data(self):
        """
        Generate candidate relative rotations and corresponding common line indices.
        """
        # Generate uniform grid on sphere with Saff-Kuijlaars and take one quarter
        # of sphere because of D2 symmetry redundancy.
        sphere_grid = self.saff_kuijlaars(self.grid_res)
        octant1_mask = np.all(sphere_grid > 0, axis=1)
        octant2_mask = (
            (sphere_grid[:, 0] > 0) & (sphere_grid[:, 1] > 0) & (sphere_grid[:, 2] < 0)
        )
        sphere_grid1 = sphere_grid[octant1_mask]
        sphere_grid2 = sphere_grid[octant2_mask]

        # Mark Equator Directions.
        # Common lines between projection directions which are perpendicular to
        # symmetry axes (equator images) have common line degeneracies. Two images
        # taken from directions on the same great circle which is perpendicular to
        # some symmetry axis only have 2 common lines instead of 4, and must be
        # treated separately.
        # We detect such directions by taking a strip of radius
        # eq_filter_angle about the 3 great circles perpendicular to the symmetry
        # axes of D2 (i.e to X,Y and Z axes).
        eq_idx1, eq_class1 = self.mark_equators(sphere_grid1, self.eq_min_dist)
        eq_idx2, eq_class2 = self.mark_equators(sphere_grid2, self.eq_min_dist)

        #  Mark Top View Directions.
        #  A Top view projection image is taken from the direction of one of the
        #  symmetry axes. Since all symmetry axes of D2 molecules are perpendicular
        #  this means that such an image is an equator with repect to both symmetry
        #  axes which are perpendicular to the direction of the symmetry axis from
        #  which the image was made, e.g. if the image was formed by projecting in
        #  the direction of the X (symmetry) axis, then it is an equator with
        #  respect to both Y and Z symmetry axes (it's direction is the
        #  interesection of 2 great circles perpendicular to Y and Z axes).
        #  Such images have severe degeneracies. A pair of Top View images (taken
        #  from different directions or a Top View and equator image only have a
        #  single common line. A top view and a regular non-equator image only have
        #  two common lines.

        # Remove top views from sphere grids and update equator indices and classes.
        self.sphere_grid1 = sphere_grid1[eq_class1 < 4]
        self.sphere_grid2 = sphere_grid2[eq_class2 < 4]
        self.eq_idx1 = eq_idx1[eq_class1 < 4]
        self.eq_idx2 = eq_idx2[eq_class2 < 4]
        self.eq_idx = np.concatenate((self.eq_idx1, self.eq_idx2))
        self.eq_class1 = eq_class1[eq_class1 < 4]
        self.eq_class2 = eq_class2[eq_class2 < 4]

        # Generate in-plane rotations for each grid point on the sphere.
        self.inplane_rotated_grid1 = self.generate_inplane_rots(
            self.sphere_grid1, self.inplane_res
        )
        self.inplane_rotated_grid2 = self.generate_inplane_rots(
            self.sphere_grid2, self.inplane_res
        )

        # Generate commmonline angles induced by all relative rotation candidates.
        cl_angles1 = self.generate_commonline_angles(
            self.inplane_rotated_grid1,
            self.inplane_rotated_grid1,
            self.eq_idx1,
            self.eq_idx1,
            self.eq_class1,
            self.eq_class1,
        )
        cl_angles2 = self.generate_commonline_angles(
            self.inplane_rotated_grid1,
            self.inplane_rotated_grid2,
            self.eq_idx1,
            self.eq_idx2,
            self.eq_class1,
            self.eq_class2,
        )

        # Generate commonline indices.
        self.cl_idx_1, self.cl_angles1 = self.generate_commonline_indices(cl_angles1)
        self.cl_idx_2, self.cl_angles2 = self.generate_commonline_indices(cl_angles2)

    def generate_scl_lookup_data(self):
        """
        Generate lookup data for self-commonlines.
        """
        # Get self-commonline angles.
        self.scl_angles1 = self.generate_scl_angles(
            self.inplane_rotated_grid1,
            self.eq_idx1,
            self.eq_class1,
        )
        self.scl_angles2 = self.generate_scl_angles(
            self.inplane_rotated_grid2,
            self.eq_idx2,
            self.eq_class2,
        )

        # Get self-commonline indices.
        self.scl_idx_1, self.scl_eq_lin_idx_lists_1 = self.generate_scl_indices(
            self.scl_angles1, self.eq_class1
        )
        self.scl_idx_2, self.scl_eq_lin_idx_lists_2 = self.generate_scl_indices(
            self.scl_angles2, self.eq_class2
        )
        self.scl_idx_lists = np.concatenate(
            (self.scl_eq_lin_idx_lists_1, self.scl_eq_lin_idx_lists_2), axis=1
        )

        # Compute non-equator indices.
        # Register non equator indices. Denote by C_ij the j'th in-plane rotation of
        # the i'th ML candidate, and arrange all candidates in a list with their in-plane
        # rotations in the order: C_11,...,C_1r,...,C_m1,...,C_mr where m is the
        # number of candidates and r is the number of in plane rotations. Here we
        # create a sub-list of only non equator candidates, i.e., if i_1,...,i_p are
        # non equators then we have the sub list is
        # C_(i_1)1,...,C(i_1)r,...C_(i_p)1,...,C_(i_p)r.
        n_non_eq = np.sum(self.eq_class1 == 0) + np.sum(self.eq_class2 == 0)
        non_eq_idx = np.zeros((n_non_eq, int(self.n_inplane_rots)))
        non_eq_idx[:, 0] = (
            np.hstack(
                (
                    np.where(self.eq_class1 == 0)[0],
                    len(self.eq_class1) + np.where(self.eq_class2 == 0)[0],
                )
            )
            * self.n_inplane_rots
        )
        for i in range(1, self.n_inplane_rots):
            non_eq_idx[:, i] = non_eq_idx[:, 0] + i

        self.non_eq_idx = non_eq_idx.astype(int)

        # Non-topview equator indices.
        non_tv_eq_idx = np.concatenate(
            (
                np.where(self.eq_class1 > 0)[0],
                len(self.eq_class1) + np.where(self.eq_class2 > 0)[0],
            )
        )

        self.non_tv_eq_idx = non_tv_eq_idx.astype(int)

    def generate_scl_angles(self, Ris, eq_idx, eq_class):
        """
        Generate self-commonline angles.

        :param Ris: Candidate rotation matrices, (n_sphere_grid, n_inplane_rots, 3, 3).
        :param eq_idx: Equator index mask for Ris.
        :param eq_class: Equator classification for Ris.
        """
        L = 360  # TODO: Maybe this should be self.n_theta

        # For each candidate rotation Ri we generate the set of 3 self-commonlines.
        scl_angles = np.zeros((*Ris.shape[:2], 3, 2), dtype=Ris.dtype)
        n_rots = len(Ris)
        for i in range(n_rots):
            Ri = Ris[i]
            # TODO: Reversing self.gs here to match matlab. Should use as is.
            for k, g in enumerate(self.gs[::-1][:3]):
                g_Ri = g * Ri
                Riis = np.transpose(Ri, axes=(0, 2, 1)) @ g_Ri

                scl_angles[i, :, k, 0] = np.arctan2(Riis[:, 2, 0], -Riis[:, 2, 1])
                scl_angles[i, :, k, 1] = np.arctan2(-Riis[:, 0, 2], Riis[:, 1, 2])

        # Prepare self commonline coordinates.
        scl_angles = scl_angles % (2 * np.pi)

        # Deal with non top view equators
        # A non-TV equator has only one self common line. However, we clasify an
        # equator as an image whose projection direction is at radial distance <
        # eq_filter_angle from the great circle perpendicural to a symmetry axis,
        # and not strictly zero distance. Thus in most cases we get 2 common lines
        # differing by a small difference in degrees. Actually the calculation above
        # gives us two NEARLY antipodal lines, so we first flip one of them by
        # adding 180 degrees to it. Then we aggregate all the rays within the range
        # between these two resulting lines to compute the score of this self common
        # line for this candidate. The scoring part is done in the ML function itself.
        # Furthermore, the line perpendicular to the self common line, though not
        # really a self common line, has the property that all its values are real
        # and both halves of the line (rays differing by pi, emanating from the
        # origin) have the same values, and so it 'behaves like' a self common
        # line which we also register here and exploit in the ML function.
        # We put the 'real' self common line at 2 first coordinates, the
        # candidate for perpendicular line is in 3rd coordinate.

        # If this is a self common line with respect to x-equator then the actual self
        # common line(s) is given by the self relative rotations given by the y and z
        # rotation (by 180 degrees) group members, i.e. Ri^TgyRj and Ri^TgzRj
        scl_angles[eq_class == 1] = scl_angles[eq_class == 1][:, :, [1, 2, 0]]
        scl_angles[eq_class == 1, :, 0] = scl_angles[eq_class == 1][:, :, 0, [1, 0]]

        # If this is a self common line with respect to y-equator then the actual self
        # common line(s) is given by the self relative rotations given by the x and z
        # rotation (by 180 degrees) group members, i.e. Ri^TgxRj and Ri^TgzRj
        scl_angles[eq_class == 2] = scl_angles[eq_class == 2][:, :, [0, 2, 1]]
        scl_angles[eq_class == 2, :, 0] = scl_angles[eq_class == 2][:, :, 0, [1, 0]]

        # If this is a self common line with respect to z-equator then the actual self
        # common line(s) is given by the self relative rotations given by the x and y
        # rotation (by 180 degrees) group members, i.e. Ri^TgxRj and Ri^TgyRj
        # No need to rearrange entries, the "real" common lines are already in
        # indices 1 and 2, but flip one common line to antipodal.
        scl_angles[eq_class == 3, :, 0] = scl_angles[eq_class == 3][:, :, 0, [1, 0]]

        # TODO: Maybe a cleaner way to do this.
        # Make sure angle range is <= 180 degrees.
        # p1 marks "equator" self-commonlines where both entries of the first
        # scl are greater than both entries of the second scl.
        p1 = scl_angles[eq_class > 0, :, 0] > scl_angles[eq_class > 0, :, 1]
        p1 = p1[:, :, 0] & p1[:, :, 1]
        # p2 marks "equator" self-commonlines where the angle range between the
        # first and second sets of self-commonlines is greater than 180.
        p2 = scl_angles[eq_class > 0, :, 0] - scl_angles[eq_class > 0, :, 1] < -np.pi
        p2 = p2[:, :, 0] | p2[:, :, 1]
        p = p1 | p2

        # Swap entries satisfying either of the above conditions.
        scl_angles[eq_class > 0] = (
            scl_angles[eq_class > 0][:, :, [1, 0, 2]] * p[:, :, None, None]
            + scl_angles[eq_class > 0] * ~p[:, :, None, None]
        )

        # Convert angles from radians to degrees (indices).
        scl_angles = np.round(scl_angles * 180 / np.pi) % L

        return scl_angles

    def generate_scl_indices(self, scl_angles, eq_class):
        L = 360

        # Create candidate common line linear indices lists for equators.
        # As indicated above for equator candidate, for each self common line we
        # don't get a single coordinate but a range of them. Here we register a
        # list of coordinates for each such self common line candidate.
        non_top_view_eq_idx = np.where(eq_class > 0)[0]
        n_eq = len(non_top_view_eq_idx)
        n_inplane_rots = scl_angles.shape[1]
        count_eq = 0

        # eq_lin_idx_lists[1,i,j] registers a list of linear indices of the j'th
        # in-plane rotation of the range for the (only) self common line of the i'th
        # candidate. eq_lin_idx_lists[2,i,j] registers the actual (integer) angle
        # of the self common line in the 2D Fourier space. Note that we need only
        # one number since each self common line has radial coordinates of the form
        # (theta, theta+180).
        eq_lin_idx_lists = np.empty((2, n_eq, n_inplane_rots), dtype=object)
        for i in non_top_view_eq_idx.tolist():
            for j in range(n_inplane_rots):
                idx1 = self.circ_seq(scl_angles[i, j, 0, 0], scl_angles[i, j, 1, 0], L)
                idx2 = self.circ_seq(scl_angles[i, j, 0, 1], scl_angles[i, j, 1, 1], L)

                # Adjust so idx1 is in [0, 180) range.
                idx1[idx1 >= 180] = (idx1[idx1 >= 180] - L // 2) % (L // 2)
                idx2[idx1 >= 180] = (idx2[idx1 >= 180] + L // 2) % L

                # register indices in list.
                eq_lin_idx_lists[0, count_eq, j] = np.ravel_multi_index(
                    (idx1, idx2), (L // 2, L)
                )
                eq_lin_idx_lists[1, count_eq, j] = idx1
            count_eq += 1

        scl_indices, _ = self.generate_commonline_indices(scl_angles)

        return scl_indices, eq_lin_idx_lists

    @staticmethod
    def circ_seq(n1, n2, L):
        """
        Make a circular sequence of integers between n1 and n2 modulo L.

        :param n1: First integer in sequence.
        :param n2: Last integer in sequence.
        :param L: Modulus of values in sequence.
        :return: Circular sequence modulo L.
        """
        if n2 < n1:
            n2 += L
        if n1 == n2:
            return np.array(n1).astype(int)

        seq = np.arange(n1, n2 + 1).astype(int) % L

        return seq

    @staticmethod
    def saff_kuijlaars(N):
        """
        Generates N vertices on the unit sphere that are approximately evenly distributed.

        This implements the recommended algorithm in spherical coordinates
        (theta, phi) according to "Distributing many points on a sphere"
        by E.B. Saff and A.B.J. Kuijlaars, Mathematical Intelligencer 19.1
        (1997) 5--11.

        :param N: Number of vertices to generate.

        :return: Nx3 array of vertices in cartesian coordinates.
        """
        k = np.arange(1, N + 1)
        h = -1 + 2 * (k - 1) / (N - 1)
        theta = np.arccos(h)
        phi = np.zeros(N)

        for i in range(1, N - 1):
            phi[i] = (phi[i - 1] + 3.6 / (np.sqrt(N * (1 - h[i] ** 2)))) % (2 * np.pi)

        # Spherical coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        mesh = np.column_stack((x, y, z))

        return mesh

    @staticmethod
    def mark_equators(sphere_grid, eq_filter_angle):
        """
        :param sphere_grid: Nx3 array of vertices in cartesian coordinates.
        :param eq_filter_angle: Angular distance from equator to be marked as
            an equator point.

        :returns:
            - eq_idx, a boolean mask for equator indices.
            - eq_class, n_rots length array of values indicating equator class.
        """
        # Project each vector onto xy, xz, yz planes and measure angular distance
        # from each plane.
        n_rots = len(sphere_grid)
        angular_dists = np.zeros((n_rots, 3), dtype=sphere_grid.dtype)

        # Distance from z-axis equator.
        proj_xy = sphere_grid.copy()
        proj_xy[:, 2] = 0
        proj_xy /= np.linalg.norm(proj_xy, axis=1)[:, None]
        angular_dists[:, 0] = np.sum(sphere_grid * proj_xy, axis=-1)

        # Distance from y-axis equator.
        proj_xz = sphere_grid.copy()
        proj_xz[:, 1] = 0
        proj_xz /= np.linalg.norm(proj_xz, axis=1)[:, None]
        angular_dists[:, 1] = np.sum(sphere_grid * proj_xz, axis=-1)

        # Distance from x-axis equator.
        proj_yz = sphere_grid.copy()
        proj_yz[:, 0] = 0
        proj_yz /= np.linalg.norm(proj_yz, axis=1)[:, None]
        angular_dists[:, 2] = np.sum(sphere_grid * proj_yz, axis=-1)

        # Mark all views close to an equator.
        eq_min_dist = np.cos(eq_filter_angle * np.pi / 180)
        n_eqs = np.sum(angular_dists > eq_min_dist, axis=1)
        eq_idx = n_eqs > 0

        # Classify equators.
        # 0 -> non-equator view
        # 1 -> z equator
        # 2 -> y equator
        # 3 -> x equator
        # 4 -> z top view
        # 5 -> y top view
        # 6 -> x top view
        eq_class = np.zeros(n_rots)
        top_view_idx = n_eqs > 1
        top_view_class = np.argmin(angular_dists[top_view_idx] > eq_min_dist)
        eq_class[top_view_idx] = top_view_class + 4
        eq_view_idx = n_eqs == 1
        eq_view_class = np.argmax(angular_dists[eq_view_idx] > eq_min_dist, axis=1)
        eq_class[eq_view_idx] = eq_view_class + 1

        return eq_idx, eq_class

    @staticmethod
    def generate_inplane_rots(sphere_grid, d_theta):
        """
        This function takes projection directions (points on the 2-sphere) and
        generates rotation matrices in SO(3). The projection direction
        is the 3rd column and columns 1 and 2 span the perpendicular plane.
        To properly discretize SO(3), for each projection direction we generate
        [2*pi/dtheta] "in-plane" rotations, of the plane
        perpendicular to this direction. This is done by generating one rotation
        for each direction and then multiplying on the right by a rotation about
        the Z-axis by k*dtheta degrees, k=0...2*pi/dtheta-1.

        :param sphere_grid: A set of points on the 2-sphere.
        :param d_theta: Resolution for in-plane rotations (in degrees)
        :returns: 4D array of rotations of size len(sphere_grid) x n_inplane_rots x 3 x 3.
        """
        dtype = sphere_grid.dtype
        # Generate one rotation for each point on the sphere.
        n_rots = len(sphere_grid)
        Ri2 = np.column_stack((-sphere_grid[:, 1], sphere_grid[:, 0], np.zeros(n_rots)))
        Ri2 /= np.linalg.norm(Ri2, axis=1)[:, None]
        Ri1 = np.cross(Ri2, sphere_grid)
        Ri1 /= np.linalg.norm(Ri1, axis=1)[:, None]

        rots_grid = np.zeros((n_rots, 3, 3), dtype=dtype)
        rots_grid[:, :, 0] = Ri1
        rots_grid[:, :, 1] = Ri2
        rots_grid[:, :, 2] = sphere_grid

        # Generate in-plane rotations.
        d_theta *= np.pi / 180
        # TODO: Negative signs to match matlab.
        inplane_rots = Rotation.about_axis(
            "z", np.arange(0, -2 * np.pi, -d_theta), dtype=dtype
        ).matrices
        n_inplane_rots = len(inplane_rots)

        # Generate in-plane rotations of rots_grid.
        inplane_rotated_grid = np.zeros((n_rots, n_inplane_rots, 3, 3), dtype=dtype)
        for i in range(n_rots):
            inplane_rotated_grid[i] = rots_grid[i] @ inplane_rots

        return inplane_rotated_grid

    def generate_commonline_angles(
        self, Ris, Rjs, Ri_eq_idx, Rj_eq_idx, Ri_eq_class, Rj_eq_class
    ):
        """
        Compute commonline angles induced by the 4 sets of relative rotations
        Rij = Ri.T @ g_m @ Rj, m = 0,1,2,3, where g_m is the identity and rotations
        about the three axes of symmetry of a D2 symmetric molecule.

        :param Ris: First set of candidate rotations.
        :param Rjs: Second set of candidate rotation.
        :param Ri_eq_idx: Equator index mask.
        :param Rj_eq_idx: Equator index mask.
        :param Ri_eq_class: Equator classification for Ris.
        :param Rj_eq_class: Equator classification for Rjs.

        :return: Commonline angles induced by relative rotation candidates.
        """
        n_rots_i = len(Ris)
        n_theta = Ris.shape[1]  # Same for Rjs, TODO: Don't call this n_theta

        # Generate upper triangular table of indicators of all pairs which are not
        # equators with respect to the same symmetry axis (named unique_pairs).
        eq_table = np.outer(Ri_eq_idx, Rj_eq_idx)
        in_same_class = (Ri_eq_class[:, None] - Rj_eq_class.T[None]) == 0
        eq2eq_Rij_table = np.triu(~(eq_table * in_same_class), 1)

        n_pairs = np.sum(eq2eq_Rij_table)
        idx = 0
        cl_angles = np.zeros((2 * n_pairs, n_theta, n_theta // 2, 4, 2))

        for i in range(n_rots_i):
            unique_pairs_i = np.where(eq2eq_Rij_table[i])[0]
            if len(unique_pairs_i) == 0:
                continue
            Ri = Ris[i]
            for j in unique_pairs_i:
                Rj = Rjs[j, : (n_theta // 2)]
                for k, g in enumerate(self.gs):
                    # Compute relative rotations candidates Rij = Ri.T @ gs @ Rj
                    g_Rj = g * Rj
                    Rijs = np.transpose(g_Rj, axes=(0, 2, 1)) @ Ri[:, None]

                    # Common line indices induced by Rijs
                    cl_angles[idx, :, :, k, 0] = np.arctan2(
                        Rijs[:, :, 2, 0], -Rijs[:, :, 2, 1]
                    )
                    cl_angles[idx, :, :, k, 1] = np.arctan2(
                        -Rijs[:, :, 0, 2], Rijs[:, :, 1, 2]
                    )
                    cl_angles[idx + n_pairs, :, :, k, 0] = np.arctan2(
                        Rijs[:, :, 0, 2], -Rijs[:, :, 1, 2]
                    )
                    cl_angles[idx + n_pairs, :, :, k, 1] = np.arctan2(
                        -Rijs[:, :, 2, 0], Rijs[:, :, 2, 1]
                    )

                idx += 1

        # Make all angles non-negative and convert to degrees.
        cl_angles = (cl_angles + 2 * np.pi) % (2 * np.pi)
        cl_angles = cl_angles * 180 / np.pi

        return cl_angles

    @staticmethod
    def generate_commonline_indices(cl_angles):
        # TODO: This is not accounting for n_theta other than 360!

        # Flatten the stack
        og_shape = cl_angles.shape
        cl_angles = np.reshape(cl_angles, (np.prod(og_shape[:-1]), 2))

        # Fourier ray index
        row_sub = np.round(cl_angles[:, 0]).astype("int") % 360
        col_sub = np.round(cl_angles[:, 1]).astype("int") % 360

        # Restrict Ri in-plane coordinates to <180 degrees.
        is_geq_than_pi = row_sub >= 180
        row_sub[is_geq_than_pi] = row_sub[is_geq_than_pi] - 180
        col_sub[is_geq_than_pi] = (col_sub[is_geq_than_pi] + 180) % 360

        # Convert to linear indices in 360*180 correlation matrix (same as cls_lookup in matlab)
        cl_idx = np.ravel_multi_index((row_sub, col_sub), dims=(180, 360))

        # Reshape cl_angles (to match matlab `cls`)
        cl_angles = cl_angles.reshape(og_shape)

        # Return as integer indices.
        return cl_idx, np.rint(cl_angles).astype(int)

    def _generate_gs(self):
        """
        Generate analogue to D2 rotation matrices, such that element-wise
        multiplication, `*`, by gs is equivalent to matrix multiplication,
        `@`, by a correspopnding rotation matrix.
        """
        gs = np.ones((4, 3, 3), dtype=self.dtype)
        gs[1, 1:3] = -gs[1, 1:3]
        gs[2, [0, 2]] = -gs[2, [0, 2]]
        gs[3, 0:2] = -gs[3, 0:2]

        self.gs = gs
