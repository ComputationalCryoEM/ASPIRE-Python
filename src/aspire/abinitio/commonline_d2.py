import logging

import numpy as np
import scipy.sparse.linalg as la
from numpy.linalg import norm

from aspire.abinitio import CLOrient3D
from aspire.operators import PolarFT
from aspire.utils import J_conjugate, Rotation, all_pairs, all_triplets, tqdm, trange
from aspire.utils.random import randn
from aspire.volume import DnSymmetryGroup

logger = logging.getLogger(__name__)


class CLSymmetryD2(CLOrient3D):
    """
    Define a class to estimate 3D orientations using common lines methods for
    molecules with D2 (dihedral) symmetry.

    Corresponding publication:
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
        epsilon=0.01,
        seed=None,
        mask=True,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with D2 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction of Fourier image.
        :param n_theta: The number of points in the theta direction of Fourier image.
        :param max_shift: Maximum range for shifts as a proportion of resolution. Default = 0.15.
        :param shift_step: Resolution of shift estimation in pixels. Default = 1 pixel.
        :param grid_res: Number of sampling points on sphere for projetion directions.
            These are generated using the Saaf-Kuijlaars algorithm. Default value is 1200.
        :param inplane_res: The sampling resolution of in-plane rotations for each
            projection direction. Default value is 5 degrees.
        :param eq_min_dist: Width of strip around equator projection directions from
            which we do not sample directions. Default value is 7 degrees.
        :param epsilon: Tolerance for J-synchronization power method.
        :param seed: Optional seed for RNG.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """

        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            mask=mask,
        )

        self.grid_res = grid_res
        self.inplane_res = inplane_res
        self.n_inplane_rots = int(360 / self.inplane_res)
        self.eq_min_dist = eq_min_dist
        self.seed = seed
        self.epsilon = epsilon

        self.triplets = all_triplets(self.n_img)
        self.pairs, self.pairs_to_linear = all_pairs(self.n_img, return_map=True)
        self.n_pairs = len(self.pairs)

        # D2 symmetry group.
        # Rearrange in order Identity, about_x, about_y, about_z.
        # This ordering is necessary for reproducing MATLAB code results.
        self.gs = DnSymmetryGroup(order=2, dtype=self.dtype).matrices[[0, 3, 2, 1]]

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with D2 symmetry. Sets the attribute
        self.rotations with an array of estimated rotation matrices, size src.nx3x3.
        """
        # Pre-compute phase-shifted polar Fourier.
        self._compute_shifted_pf()

        # Generate lookup data
        self._generate_lookup_data()
        self._generate_scl_lookup_data()

        # Compute self common-line scores.
        self._compute_scl_scores()

        # Compute common-lines and estimate relative rotations Rijs.
        self._compute_cl_scores()

        # Perform handedness synchronization.
        self.Rijs_sync = self._global_J_sync(self.Rijs_est)

        # Synchronize colors.
        self.colors, self.Rijs_rows = self._sync_colors(self.Rijs_sync)

        # Synchronize signs.
        Ris = self._sync_signs(self.Rijs_rows, self.colors)

        # Assign rotations.
        self.rotations = Ris

    #########################
    # Prepare Polar Fourier #
    #########################

    def _compute_shifted_pf(self):
        """
        Pre-compute shifted and full polar Fourier transforms.
        """
        logger.info("Preparing polar Fourier transform.")
        pf = self.pf

        # Generate shift phases.
        r_max = pf.shape[-1]
        max_shift_1d = np.ceil(2 * np.sqrt(2) * self.max_shift)
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, self.shift_step
        )
        self.n_shifts = len(shifts)

        # Reconstruct full polar Fourier for use in correlation.
        pf[:, :, 0] = 0  # Matching matlab convention to zero out the lowest frequency.
        pf /= norm(pf, axis=2)[..., np.newaxis]  # Normalize each ray.
        self.pf_full = PolarFT.half_to_full(pf)

        # Pre-compute shifted pf's.
        pf_shifted = pf[:, None] * shift_phases[None, :, None]
        self.pf_shifted = pf_shifted.reshape(
            (self.n_img, self.n_shifts * (self.n_theta // 2), r_max)
        )

    ###################################
    # Generate Commonline Lookup Data #
    ###################################

    def _generate_lookup_data(self):
        """
        Generate candidate relative rotations and corresponding common line indices.
        """
        logger.info("Generating commonline lookup data.")
        # Generate uniform grid on sphere with Saff-Kuijlaars and take one quarter
        # of sphere because of D2 symmetry redundancy.
        sphere_grid = self._saff_kuijlaars(self.grid_res)
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
        # `eq_min_dist` about the 3 great circles perpendicular to the symmetry
        # axes of D2 (i.e to X,Y and Z axes).
        eq_class1 = self._mark_equators(sphere_grid1, self.eq_min_dist)
        eq_class2 = self._mark_equators(sphere_grid2, self.eq_min_dist)

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
        self.eq_class1 = eq_class1[eq_class1 < 4]
        self.eq_class2 = eq_class2[eq_class2 < 4]

        # Generate in-plane rotations for each grid point on the sphere.
        self.inplane_rotated_grid1 = self._generate_inplane_rots(
            self.sphere_grid1, self.inplane_res
        )
        self.inplane_rotated_grid2 = self._generate_inplane_rots(
            self.sphere_grid2, self.inplane_res
        )

        # Generate commmonline angles induced by all relative rotation candidates.
        cl_angles1, self.eq2eq_Rij_table_11 = self._generate_commonline_angles(
            self.inplane_rotated_grid1,
            self.inplane_rotated_grid1,
            self.eq_class1,
            self.eq_class1,
        )
        cl_angles2, self.eq2eq_Rij_table_12 = self._generate_commonline_angles(
            self.inplane_rotated_grid1,
            self.inplane_rotated_grid2,
            self.eq_class1,
            self.eq_class2,
            same_octant=False,
        )

        # Generate commonline indices.
        self.cl_idx_1 = self._generate_commonline_indices(cl_angles1)
        self.cl_idx_2 = self._generate_commonline_indices(cl_angles2)
        self.cl_idx = np.hstack((self.cl_idx_1, self.cl_idx_2))

    def _generate_commonline_angles(
        self,
        Ris,
        Rjs,
        Ri_eq_class,
        Rj_eq_class,
        same_octant=True,
    ):
        """
        Compute commonline angles induced by the 4 sets of relative rotations
        Rij = Ri.T @ g_m @ Rj, m = 0,1,2,3, where g_m is the identity and rotations
        about the three axes of symmetry of a D2 symmetric molecule. Note, we only
        compute commonline angles between pairs of images which are not equator
        images with respect to the same axis of symmetry. To do this we build a
        table, `eq2eq_Rij_table`, which is `False` for pairs of images that are
        equator images with respect to the same axis of symmetry and `True` otherwise.

        :param Ris: First set of candidate rotations.
        :param Rjs: Second set of candidate rotation.
        :param Ri_eq_class: Equator classification for Ris.
        :param Rj_eq_class: Equator classification for Rjs.
        :param same_octant: True if both sets of candidates are in the same octant.

        :return: Commonline angles induced by relative rotation candidates.
        """
        n_rots_i = len(Ris)
        n_theta = Ris.shape[1]  # Same for Rjs, TODO: Don't call this n_theta

        # Generate upper triangular table of indicators of all pairs which are not
        # equators with respect to the same symmetry axis (named unique_pairs).
        eq_table = np.outer(Ri_eq_class > 0, Rj_eq_class > 0)
        in_same_class = (Ri_eq_class[:, None] - Rj_eq_class.T[None]) == 0
        eq2eq_Rij_table = ~(eq_table * in_same_class)

        # For candidates in the same octant only need upper triangle of table.
        if same_octant:
            eq2eq_Rij_table = np.triu(eq2eq_Rij_table, 1)

        n_pairs = np.count_nonzero(eq2eq_Rij_table)
        idx = 0
        cl_angles = np.zeros((2, n_pairs, n_theta, n_theta // 2, 4, 2))

        for i in range(n_rots_i):
            unique_pairs_i = np.nonzero(eq2eq_Rij_table[i])[0]
            if len(unique_pairs_i) == 0:
                continue
            Ri = Ris[i]
            for j in unique_pairs_i:
                Rj = Rjs[j, : n_theta // 2]

                # Compute relative rotations candidates Rij = Ri.T @ gs @ Rj
                Rijs = (
                    np.transpose(Ri, axes=(0, 2, 1))[:, None, None]
                    @ self.gs
                    @ Rj[:, None]
                )

                # Common line indices induced by Rijs
                cl_angles[0, idx, :, :, :, 0] = np.arctan2(
                    -Rijs[..., 0, 2], Rijs[..., 1, 2]
                )
                cl_angles[0, idx, :, :, :, 1] = np.arctan2(
                    Rijs[..., 2, 0], -Rijs[..., 2, 1]
                )
                cl_angles[1, idx, :, :, :, 0] = np.arctan2(
                    -Rijs[..., 2, 0], Rijs[..., 2, 1]
                )
                cl_angles[1, idx, :, :, :, 1] = np.arctan2(
                    Rijs[..., 0, 2], -Rijs[..., 1, 2]
                )

                idx += 1

        # Make all angles non-negative and convert to degrees.
        cl_angles = (cl_angles + 2 * np.pi) % (2 * np.pi)
        cl_angles = cl_angles * 180 / np.pi

        return cl_angles, eq2eq_Rij_table

    ########################################
    # Generate Self-Commonline Lookup Data #
    ########################################

    def _generate_scl_lookup_data(self):
        """
        Generate lookup data for self-commonlines.
        """
        logger.info("Generating self-commonline lookup data.")
        # Get self-commonline angles.
        self.scl_angles1 = self._generate_scl_angles(
            self.inplane_rotated_grid1,
            self.eq_class1,
        )
        self.scl_angles2 = self._generate_scl_angles(
            self.inplane_rotated_grid2,
            self.eq_class2,
        )

        # Get self-commonline indices.
        self.scl_idx_1, self.scl_eq_lin_idx_lists_1 = self._generate_scl_indices(
            self.scl_angles1, self.eq_class1
        )
        self.scl_idx_2, self.scl_eq_lin_idx_lists_2 = self._generate_scl_indices(
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
        n_non_eq = np.count_nonzero(self.eq_class1 == 0) + np.count_nonzero(
            self.eq_class2 == 0
        )
        non_eq_idx = np.zeros((n_non_eq, self.n_inplane_rots), dtype=int)
        non_eq_idx[:, 0] = (
            np.hstack(
                (
                    np.nonzero(self.eq_class1 == 0)[0],
                    len(self.eq_class1) + np.nonzero(self.eq_class2 == 0)[0],
                )
            )
            * self.n_inplane_rots
        )
        non_eq_idx[:, 1:] = non_eq_idx[:, [0]] + np.arange(1, self.n_inplane_rots)

        self.non_eq_idx = non_eq_idx

        # Non-topview equator indices.
        self.non_tv_eq_idx = np.concatenate(
            (
                np.nonzero(self.eq_class1 > 0)[0],
                len(self.eq_class1) + np.nonzero(self.eq_class2 > 0)[0],
            )
        )

        # Generate maps from scl indices to relative rotations.
        self._generate_scl_scores_idx_map()

    def _generate_scl_angles(self, Ris, eq_class):
        """
        Generate self-commonline angles. For each candidate rotation a pair of self-commonline
        angles are generated for each of the 3 self-commonlines induced by D2 symmetry.

        :param Ris: Candidate rotation matrices, (n_sphere_grid, n_inplane_rots, 3, 3).
        :param eq_idx: Equator index mask for Ris.
        :param eq_class: Equator classification for Ris.

        :return: `scl_angles` of shape (n_sphere_grid, n_inplane_rots, 3, 2).
        """

        # For each candidate rotation Ri we generate the set of 3 self-commonlines.
        scl_angles = np.zeros((*Ris.shape[:2], 3, 2), dtype=Ris.dtype)
        n_rots = len(Ris)
        for i in range(n_rots):
            Ri = Ris[i]
            for k, g in enumerate(self.gs[1:]):
                g_Ri = g @ Ri
                Riis = np.transpose(Ri, axes=(0, 2, 1)) @ g_Ri

                scl_angles[i, :, k, 0] = np.arctan2(Riis[:, 2, 0], -Riis[:, 2, 1])
                scl_angles[i, :, k, 1] = np.arctan2(-Riis[:, 0, 2], Riis[:, 1, 2])

        # Prepare self commonline coordinates.
        scl_angles = scl_angles % (2 * np.pi)

        # Deal with non top view equators
        # A non-TV equator has only one self common line. However, we clasify an
        # equator as an image whose projection direction is at radial distance <
        # `eq_min_dist` from the great circle perpendicular to a symmetry axis,
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

        # Make sure angle range is < 180 degrees.
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

        # Convert from radians [0,2*pi) to degrees [0, 360).
        return np.round(scl_angles * 180 / np.pi) % 360

    def _generate_scl_indices(self, scl_angles, eq_class):
        """
        Generate self-commonline indices. This includes a set of linear indices for
        all candidate rotations as well as lists of self-commonline index ranges for
        equator candidates.

        :param scl_angles: Self-commonline angles, shape (n_sphere_grid, n_inplane_rots, 3, 2).
        :param eq_class: Equator classification for the sphere_grid points represented
            by the first axis of `scl_angles`.

        :returns:
            - scl_indices, self-commonline linear indices.
            - eq_lin_idx_lists, a list containing a range of self-commonline
                indices for each equator candidate.
        """
        L = self.n_theta

        # Convert from angles to indices.
        scl_indices = self._generate_commonline_indices(scl_angles)
        scl_angles = np.mod(np.round(scl_angles / (2 * np.pi) * L), L).astype(int)

        # Create candidate common line linear indices lists for equators.
        # As indicated above for equator candidate, for each self common line we
        # don't get a single coordinate but a range of them. Here we register a
        # list of coordinates for each such self common line candidate.
        non_top_view_eq_idx = np.nonzero(eq_class > 0)[0]
        n_eq = len(non_top_view_eq_idx)
        n_inplane_rots = scl_angles.shape[1]
        count_eq = 0

        # eq_lin_idx_lists[0,i,j] registers a list of linear indices of the j'th
        # in-plane rotation of the range for the (only) self common line of the i'th
        # candidate. eq_lin_idx_lists[1,i,j] registers the actual (integer) angle
        # of the self common line in the 2D Fourier space. Note that we need only
        # one number since each self common line has radial coordinates of the form
        # (theta, theta+180).
        eq_lin_idx_lists = np.empty((2, n_eq, n_inplane_rots), dtype=object)
        for i in non_top_view_eq_idx.tolist():
            for j in range(n_inplane_rots):
                idx1 = self._circ_seq(scl_angles[i, j, 0, 0], scl_angles[i, j, 1, 0], L)
                idx2 = self._circ_seq(scl_angles[i, j, 0, 1], scl_angles[i, j, 1, 1], L)

                # Ensure idx1 and idx2 have same number of elements.
                # Might be off by one due to n_theta discretization.
                end = np.minimum(len(idx1), len(idx2))
                idx1, idx2 = idx1[:end], idx2[:end]

                # Adjust so idx1 is in [0, 180) range.
                is_geq_than_pi = idx1 >= L // 2
                idx1[is_geq_than_pi] = idx1[is_geq_than_pi] - L // 2
                idx2[is_geq_than_pi] = (idx2[is_geq_than_pi] + L // 2) % L

                # register indices in list.
                eq_lin_idx_lists[0, count_eq, j] = np.ravel_multi_index(
                    (idx1, idx2), (L // 2, L)
                )
                eq_lin_idx_lists[1, count_eq, j] = idx1
            count_eq += 1

        return scl_indices, eq_lin_idx_lists

    def _generate_scl_scores_idx_map(self):
        """
        Generates lookup tables for maximum likelihood scheme to estimate commonlines
        between images.

        This method creates two lookup tables (`oct1_ij_map` and `oct2_ij_map`)
        for pairs of candidate rotations (i, j) under the following conditions:

        1. Both rotations Ri and Rj are in octant 1.
        2. Ri is in octant 1 and Rj is in octant 2.

        For each pair of candidate rotations the tables give a map into the set of
        self-commonlines induced by those rotations. This table will be used later
        to incorporate a likelihood score for self-commonlines into the likelihood
        score for common lines for each pair of images.
        """
        # Calculate number of rotations in each octant.
        n_rot_1 = len(self.scl_idx_1) // (3 * self.n_inplane_rots)
        n_rot_2 = len(self.scl_idx_2) // (3 * self.n_inplane_rots)

        # First the map for i<j pairs for the rotations Ri and Rj in octant 1
        # which are not equator images with respect to the same axis of symmetry.
        n_pairs = np.count_nonzero(self.eq2eq_Rij_table_11)
        oct1_ij_map = np.zeros(
            (n_pairs, self.n_inplane_rots**2 // 2, 2), dtype=np.int64
        )

        # Create index arrays for i and j to cover all rotation combinations.
        i_idx = np.repeat(np.arange(self.n_inplane_rots), self.n_inplane_rots // 2)
        j_idx = np.tile(np.arange(self.n_inplane_rots // 2), self.n_inplane_rots)

        idx_vec = np.arange(n_rot_1)
        idx = 0

        for i in range(n_rot_1):
            unique_pairs_i = idx_vec[self.eq2eq_Rij_table_11[i]]
            if len(unique_pairs_i) == 0:
                continue
            i_idx_plus_offset = i_idx + (i * self.n_inplane_rots)

            for j in unique_pairs_i:
                j_idx_plus_offset = j_idx + (j * self.n_inplane_rots)
                oct1_ij_map[idx] = np.column_stack(
                    (i_idx_plus_offset, j_idx_plus_offset)
                )
                idx += 1

        # Now the map for i<j pairs for Ri in octant 1 and Rj in octant 2.
        n_pairs_12 = np.count_nonzero(self.eq2eq_Rij_table_12)
        oct2_ij_map = np.zeros(
            (n_pairs_12, self.n_inplane_rots**2 // 2, 2), dtype=np.int64
        )
        idx_vec = np.arange(n_rot_2)
        idx = 0

        for i in range(n_rot_1):
            unique_pairs_i = idx_vec[self.eq2eq_Rij_table_12[i] > 0]
            if len(unique_pairs_i) == 0:
                continue
            i_idx_plus_offset = i_idx + (i * self.n_inplane_rots)

            for j in unique_pairs_i:
                j_idx_plus_offset = j_idx + (j * self.n_inplane_rots)
                oct2_ij_map[idx] = np.column_stack(
                    (i_idx_plus_offset, j_idx_plus_offset)
                )
                idx += 1

        tmp1 = oct1_ij_map[:, :, 0].flatten()
        tmp2 = oct1_ij_map[:, :, 1].flatten()
        self.oct1_ij_map = np.column_stack((tmp1, tmp2))

        tmp1 = oct2_ij_map[:, :, 0].flatten()
        tmp2 = oct2_ij_map[:, :, 1].flatten()
        self.oct2_ij_map = np.column_stack((tmp1, tmp2))

    ##############################################
    # Compute Self-Commonline Correlation Scores #
    ##############################################

    def _compute_scl_scores(self):
        """
        Compute correlations for self-commonline candidates. For each image i
        we compute an auto-correlation table between all polar Fourier rays.
        We then use that table to apply a score to each non-topview candidate
        rotation which gives the likelihood that the self-commonlines induced
        by that candidate belong to the image i..
        """
        logger.info("Computing self-commonline correlation scores.")
        n_img = self.n_img
        n_theta = self.n_theta
        n_eq = len(self.non_tv_eq_idx)
        n_inplane = self.n_inplane_rots

        # Prepare self-commonline indices.
        scl_matrix = np.concatenate((self.scl_idx_1, self.scl_idx_2))
        M = len(scl_matrix) // 3
        scl_idx = scl_matrix.reshape(M, 3)

        # Get non-equator indices to use with corrs matrix.
        non_eq_lin_idx = self.non_eq_idx.flatten()
        n_non_eq = len(non_eq_lin_idx)
        non_eq_idx = np.unravel_index(
            scl_idx[non_eq_lin_idx].flatten(), (n_theta // 2, n_theta)
        )

        # Compute max correlation over all shifts.
        corrs = np.real(
            self.pf_shifted @ np.transpose(np.conj(self.pf_full), (0, 2, 1))
        )
        corrs = np.reshape(corrs, (self.n_img, self.n_shifts, n_theta // 2, n_theta))
        corrs = np.max(corrs, axis=1)

        # Map correlations to probabilities (in the spirit of Maximum Likelihood).
        corrs = 0.5 * (corrs + 1)

        # Compute equator measures.
        eq_measures = np.zeros((self.n_img, n_theta // 2), dtype=self.dtype)
        for i in range(self.n_img):
            eq_measures[i] = self._all_eq_measures(corrs[i])

        # Handle the cases: Non-equator, Non-top-view equator images.
        # 1. Non-equators: just take product of probabilities.
        corrs_out = np.zeros((n_img, M), dtype=self.dtype)
        prod_corrs = np.prod(
            corrs[:, non_eq_idx[0], non_eq_idx[1]].reshape(self.n_img, n_non_eq, 3),
            axis=2,
        )
        corrs_out[:, non_eq_lin_idx] = prod_corrs

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
                true_scls_corrs = corrs[:, scl_idx_list[0], scl_idx_list[1]]
                scls_cand_idx = self.scl_idx_lists[1, eq_idx, j]
                eq_measures_j = eq_measures[:, scls_cand_idx]
                measures_agg = true_scls_corrs[:, :, None] * eq_measures_j[:, None, :]
                k = self.non_tv_eq_idx[eq_idx]
                corrs_out[:, k * n_inplane + j] = np.max(measures_agg, axis=(-2, -1))

        self.scls_scores = corrs_out

    def _all_eq_measures(self, corrs):
        """
        Compute a measure indicating how likely an image is an equator image.

        :param corrs: Correlation table of shape (n_theta // 2, n_theta).

        :return: (n_theta // 2) likelihood scores.
        """
        # First compute the eq measure (corrs(scl-k,scl+k) for k=1:n_theta // 4)
        # An equator image of a D2 molecule has the following property: If t_i is
        # the angle of one of the rays of the self common line then all the pairs of
        # rays of the form (t_i-k,t_i+k) for k=1:n_theta // 4 are identical. For each t_i we
        # average over correlations between the lines (t_i-k,t_i+k) for k=1:n_theta // 4
        # to measure the likelihood that the image is an equator and the ray (line)
        # with angle t_i is a self common line.
        # (This first loop can be done once outside this function and then pass
        # idx as an argument).
        L = self.n_theta
        L_half = L // 2

        # Generate indices using broadcasting.
        t_i = np.arange(L_half)[:, None, None]
        k_vals = np.arange(1, L // 4 + 1)[None, :, None]
        neg_pos_k = np.array([-1, 1])[None, None, :]

        # Calculate indices, shape: (L//2, L//4, 2).
        idx = np.mod(t_i + k_vals * neg_pos_k, L)

        # Convert to Fourier ray indices.
        idx_1 = idx[:, :, 0].flatten()
        idx_2 = idx[:, :, 1].flatten()

        # Adjust idx_1 to be within [0, 180) and adjust idx_2 accordingly.
        is_geq_than_pi = idx_1 >= L_half
        idx_1[is_geq_than_pi] -= L_half
        idx_2[is_geq_than_pi] = (idx_2[is_geq_than_pi] + L_half) % L

        # Compute correlations
        eq_corrs = corrs[idx_1, idx_2].reshape(L_half, L // 4)
        corrs_mean = np.mean(eq_corrs, axis=1)

        # Now compute correlations for normals to scls.
        # An eqautor image of a D2 molecule has the additional following property:
        # The normal line to a self common line in 2D Fourier plane is real valued
        # and both of its rays have identical values. We use the correlation
        # between one Fourier ray of the normal to a self common line candidate t_i
        # with its anti-podal as an additional way to measure if the image is an
        # equator and t_i+0.5*pi is the normal to its self common line.
        r = np.ceil(2 * L / 360).astype(
            int
        )  # Search radius within 2 degrees of normal ray.

        # Generate indices for normal to scl index.
        normal_2_scl_idx_0 = (
            L_half - np.arange(L_half // 2 - r, L_half // 2 + r + 1)
        ) % L
        normal_2_scl_idx = (normal_2_scl_idx_0 + np.arange(L_half).reshape(-1, 1)) % L

        # Adjust indices to be within [0, 180) range.
        normal_2_scl_idx = np.where(
            normal_2_scl_idx >= L_half, normal_2_scl_idx - L_half, normal_2_scl_idx
        )

        # Compute correlations for normals.
        normal_corrs = corrs[normal_2_scl_idx, normal_2_scl_idx + L_half]
        normal_corrs_max = np.max(normal_corrs, axis=1)

        return corrs_mean * normal_corrs_max

    #########################################
    # Compute Commonline Correlation Scores #
    #########################################

    def _compute_cl_scores(self):
        """
        Run common lines Maximum likelihood procedure for a D2 molecule, to find
        the set of rotations Ri^TgkRj, k=1,2,3,4 for each pair of images i and j.
        """
        logger.info("Computing commonline correlation scores.")
        L = self.n_theta
        n_pairs = self.n_img * (self.n_img - 1) // 2

        # Map the self common line scores of each 2 candidate rotations R_i, R_j
        n_lookup_1 = len(self.scl_idx_1) // 3
        oct1_ij_map = np.vstack((self.oct1_ij_map, self.oct1_ij_map[:, [1, 0]]))
        oct2_ij_map = self.oct2_ij_map
        oct2_ij_map[:, 1] += n_lookup_1
        oct2_ij_map = np.vstack((oct2_ij_map, oct2_ij_map[:, [1, 0]]))
        ij_map = np.vstack((oct1_ij_map, oct2_ij_map))

        # Gather commonline indices and unravel to index into correlations.
        cl_idx = np.unravel_index(self.cl_idx, (L // 2, L))

        # Allocate output variables
        corrs_idx = np.zeros(n_pairs, dtype=np.int64)
        corrs_out = np.zeros(n_pairs, dtype=self.dtype)

        ij_idx = 0
        pbar = tqdm(
            desc="Searching for commonlines between pairs of images", total=n_pairs
        )

        # For each i'th image compute the correlation with all j'th images, j > i.
        for i in range(self.n_img - 1):
            pf_i = self.pf_shifted[i]
            scores_i = self.scls_scores[i]

            # Gather all pf_j in one array for vectorized computation
            pf_js = self.pf_full[i + 1 : self.n_img]
            n_pf_js = pf_js.shape[0]

            # Compute maximum correlation over all shifts for all pf_j
            corrs = np.real(pf_i @ np.conj(pf_js.transpose(0, 2, 1)))
            corrs = corrs.reshape(n_pf_js, self.n_shifts, L // 2, L)
            corrs = np.max(corrs, axis=1)  # Max over shifts

            # Take the product over symmetrically induced candidates. Eq. 4.5 in paper.
            prod_corrs = corrs[:, cl_idx[0], cl_idx[1]]
            prod_corrs = prod_corrs.reshape(n_pf_js, len(prod_corrs[0]) // 4, 4)
            prod_corrs = np.prod(prod_corrs, axis=2)

            # Incorporate scores of individual rotations from self-commonlines
            scores_js = self.scls_scores[i + 1 : self.n_img]
            scores_ij = scores_i[ij_map[:, 0]] * scores_js[:, ij_map[:, 1]]

            # Find maximum correlations and update results
            prod_corrs = prod_corrs * scores_ij
            max_indices = np.argmax(prod_corrs, axis=1)
            corrs_idx[ij_idx : ij_idx + len(max_indices)] = max_indices
            corrs_out[ij_idx : ij_idx + len(max_indices)] = prod_corrs[
                np.arange(len(max_indices)), max_indices
            ]

            ij_idx += len(max_indices)
            pbar.update(len(max_indices))

        pbar.close()

        # Get estimated relative viewing directions
        self.corrs_idx = corrs_idx
        self.Rijs_est = self._get_Rijs_from_lin_idx(corrs_idx)

    def _get_Rijs_from_lin_idx(self, lin_idx):
        """
        Restore map results from maximum-likelihood over commonlines to corresponding
        relative rotations.

        :param lin_idx: Set of linear indices corresponding to best estimate of Rijs.

        :return: Estimated Rijs.
        """
        Rijs_est = np.zeros((len(lin_idx), 4, 3, 3), dtype=self.dtype)
        n_cand_per_oct = len(self.cl_idx_1) // 4
        oct1_idx = lin_idx < n_cand_per_oct
        n_est_in_oct1 = np.count_nonzero(oct1_idx)
        if n_est_in_oct1 > 0:
            Rijs_est[oct1_idx] = self._get_Rijs_from_oct(lin_idx[oct1_idx], octant=1)
        if n_est_in_oct1 <= len(lin_idx):
            Rijs_est[~oct1_idx] = self._get_Rijs_from_oct(
                lin_idx[~oct1_idx] - n_cand_per_oct, octant=2
            )

        return Rijs_est

    def _get_Rijs_from_oct(self, lin_idx, octant=1):
        """
        Calculate estimated relative rotations Rijs from the linear indices of
        common-lines estimates from the search table. Rijs are generated from the
        rotation grids from which the common-lines table was generated.

        :param lin_idx: Set of linear indices corresponding to best estimate of Rijs.
        :param octant: Octant of rotation grid from which the Rj rotation was selected
             when generating the common-lines table.
        :return: Estimated Rijs.
        """
        if octant not in [1, 2]:
            raise ValueError("`octant` must be 1 or 2.")

        # Get pairs lookup table.
        if octant == 1:
            unique_pairs = self.eq2eq_Rij_table_11
        else:
            unique_pairs = self.eq2eq_Rij_table_12

        n_theta = self.n_inplane_rots
        n_lookup_pairs = np.count_nonzero(unique_pairs)
        n_rots = len(self.sphere_grid1)
        if octant == 1:
            n_rots2 = n_rots
        else:
            n_rots2 = len(self.sphere_grid2)

        # Map linear indices of chosen pairs of rotation candidates from ML to regular indices.
        p_idx, inplane_i, inplane_j = np.unravel_index(
            lin_idx, (2 * n_lookup_pairs, n_theta, n_theta // 2)
        )
        transpose_idx = p_idx >= n_lookup_pairs
        p_idx[transpose_idx] -= n_lookup_pairs
        s = self.inplane_rotated_grid1.shape
        inplane_rotated_grid = np.reshape(
            self.inplane_rotated_grid1, (np.prod(s[0:2]), 3, 3)
        )
        if octant == 1:
            s2 = s
            inplane_rotated_grid2 = inplane_rotated_grid
        else:
            s2 = self.inplane_rotated_grid2.shape
            inplane_rotated_grid2 = np.reshape(
                self.inplane_rotated_grid2, (np.prod(s2[0:2]), 3, 3)
            )

        # Convert linear indices of unique table to linear indices of index pairs table.
        idx_vec = np.arange(np.prod(unique_pairs.shape))
        unique_lin_idx = idx_vec[unique_pairs.flatten()]
        I, J = np.unravel_index(unique_lin_idx, (n_rots, n_rots2))
        est_idx = np.vstack((I[p_idx], J[p_idx]))

        # Assemble relative rotations Ri^TgRj using linear indices, where g is a group member of D2.
        Ris_lin_idx = np.ravel_multi_index((est_idx[0], inplane_i), s[:2])
        Rjs_lin_idx = np.ravel_multi_index((est_idx[1], inplane_j), s2[:2])
        Ris_t = np.transpose(inplane_rotated_grid[Ris_lin_idx], (0, 2, 1))
        Rjs = inplane_rotated_grid2[Rjs_lin_idx]
        Rijs_est = Ris_t[:, None] @ self.gs @ Rjs[:, None]

        Rijs_est[transpose_idx] = np.transpose(Rijs_est[transpose_idx], (0, 1, 3, 2))

        return Rijs_est

    ####################################
    # Perform Global J Synchronization #
    ####################################

    def _global_J_sync(self, Rijs):
        """
        Global J-synchronization of all third row outer products. Given n_pairsx4x3x3
        matrices Rijs, each of which might contain a spurious J, ie.
        Rij = J @ Ri.T @ gs @ Rj @ J instead of Rij = Ri.T @ gs @ Rj, we return Rijs
        that all have either a spurious J or not.

        :param Rijs: An (n-choose-2)x4 x3x3 array where each 3x3 slice holds an estimate
            for the corresponding outer-product Ri.T @ Rj. Each estimate might have a
            spurious J independently of other estimates.

        :return: Rijs, all of which have a spurious J or not.
        """
        logger.info("Performing global handedness synchronization.")
        # Find best J_configuration.
        J_list = self._J_configuration(Rijs)

        # Determine relative handedness of Rijs.
        sign_ij_J = self._J_sync_power_method(J_list)

        # Synchronize Rijs
        logger.info("Applying global handedness synchronization.")
        mask = sign_ij_J == 1
        Rijs[mask] = J_conjugate(Rijs[mask])

        return Rijs

    def _J_configuration(self, Rijs):
        """
        For each triplet of indices (i, j, k), consider the relative rotations
        tuples {Ri^TgmRj}, {Ri^TglRk} and {Rj^TgrRk}. Compute norms of the form
        ||Ri^TgmRj*Rj^TglRk-Ri^TglRk||, ||J*Ri^TgmRj*J*Rj^TglRk-Ri^TglRk||,
        ||Ri^TgmRj*J*Rj^TglRk*J-Ri^TglRk| and ||Ri^TgmRj*Rj^TglRk-J*Ri^TglRk*J||
        where gm,gl,gr are the varipus gorup members of Dn and J=diag([1,1-1]).
        The correct "J-configuration" is given for the smallest of these 4 norms.

        :param Rijs: (n-choose-2)x3x3 array of relative rotations.
        :return: List of n-choose-3 indices in {0,1,2,3} indicating
            which J-configuration for each triplet of Rijs, i<j<k.
        """
        J_list = np.zeros(len(self.triplets), dtype="int")
        R = Rijs
        trip_idx = 0
        pbar = tqdm(
            desc="Finding best J-configuration over triplets", total=len(self.triplets)
        )
        for i, j, k in self.triplets:
            ij = self.pairs_to_linear[i, j]
            jk = self.pairs_to_linear[j, k]
            ik = self.pairs_to_linear[i, k]
            Rij, Rjk, Rik = R[ij], R[jk], R[ik]
            Rjk_t = np.transpose(Rjk, (0, 2, 1))

            Rij_J = J_conjugate(Rij)
            Rjk_t_J = J_conjugate(Rjk_t)
            Rik_J = J_conjugate(Rik)

            final_votes = np.zeros(4)
            final_votes[0] = self._compare_rots(Rij, Rjk_t, Rik)
            final_votes[1] = self._compare_rots(Rij_J, Rjk_t, Rik)
            final_votes[2] = self._compare_rots(Rij, Rjk_t_J, Rik)
            final_votes[3] = self._compare_rots(Rij, Rjk_t, Rik_J)

            J_list[trip_idx] = np.argmin(final_votes)
            trip_idx += 1

            pbar.update()
        pbar.close()

        return J_list

    def _compare_rots(self, Rij, Rjk_t, Rik):
        """
        Compute norms for the 4 J-configurations and return indices
        corresponding to best configuration for the provided triplet
        of relative rotations.

        :param Rij: Relative rotation between i'th and j'th candidate rotations
            of shape (4, 3, 3).
        :param Rjk_t: Transpose of relative rotation between j'th and k'th candidate
            rotations of shape (4, 3, 3).
        :param Rik: Relative rotation between i'th and k'th candidate rotations
            of shape (4, 3, 3).
        :return: Score for this J-configuration of the given rotation triplet.
        """
        # We compute the four sets of 4^3 norms |Rik @ Rjk.T - Rij|
        # See equation (6.11) in publication.
        prod_arr = Rik[:, None] @ Rjk_t[None]
        diff_arr = prod_arr[:, :, None] - Rij
        diff_arr = diff_arr.reshape((64, 9))
        norm_arr = np.sum(diff_arr**2, axis=1)

        # For perfect estimates, 16 of the 64 norms will equal zero.
        # We sum over the smallest 16 values to get a vote for this J-configuration.
        m = np.sort(norm_arr)
        vote = np.sum(m[:16])

        return vote

    def _J_sync_power_method(self, J_list):
        """
        Calculate the leading eigenvector of the J-synchronization matrix
        using the power method.

        As the J-synchronization matrix is of size (n-choose-2)x(n-choose-2), we
        use the power method to compute the eigenvalues and eigenvectors,
        while constructing the matrix on-the-fly.

        :param Rijs: (n-choose-2)x3x3 array of estimates of relative orientation matrices.

        :return: An array of length n-choose-2 consisting of 1 or -1, where the sign
            of the i'th entry indicates whether the i'th relative orientation matrix
            will be J-conjugated.
        """

        # Set power method tolerance and maximum iterations.
        epsilon = self.epsilon
        max_iters = 100

        # Initialize candidate eigenvectors
        vec = randn(self.n_pairs, seed=self.seed)
        vec = vec / norm(vec)
        residual = 1
        itr = 0

        # Power method iterations
        logger.info(
            "Initiating power method to estimate J-synchronization matrix eigenvector."
        )
        while itr < max_iters and residual > epsilon:
            itr += 1
            vec_new = self._signs_times_v(J_list, vec)
            vec_new = vec_new / norm(vec_new)
            residual = norm(vec_new - vec)
            vec = vec_new
            logger.info(
                f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
            )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec)
        J_sync = np.sign(J_sync[0]) * J_sync  # Stabilize J_sync

        return J_sync

    def _signs_times_v(self, J_list, vec):
        """
        Multiplication of the J-synchronization matrix by a candidate eigenvector.

        The J-synchronization matrix is a matrix representation of the handedness graph,
        Gamma, whose set of nodes consists of the estimates Rijs and whose set of edges
        consists of the undirected edges between all triplets of estimates Rij, Rjk,
        and Rik, where i<j<k. The weight of an edge is set to +1 if its incident nodes
        agree in handednes and -1 if not.

        The J-synchronization matrix is of size (n-choose-2)x(n-choose-2), where each
        entry corresponds to the relative handedness of Rij and Rjk. The entry (ij, jk),
        where ij and jk are retrieved from the all_pairs indexing, is 1 if Rij and Rjk
        are of the same handedness and -1 if not. All other entries (ij, kl) hold a zero.

        Due to the large size of the J-synchronization matrix we construct it on the fly
        as follows. For each triplet of outer products Rij, Rjk, and Rik, the associated
        elements of the J-synchronization matrix are populated with +1 or -1 and
        multiplied by the corresponding elements of the current candidate eigenvector
        supplied by the power method. The new candidate eigenvector is updated for each
        triplet.

        :param J_list: n-choose-3 array of indices indicating the best signs configuration.
        :param vec: The current candidate eigenvector of length n-choose-2 from the power
            method.

        :return: New candidate eigenvector of length n-choose-2. The product of the J-sync
            matrix and vec.
        """
        new_vec = np.zeros_like(vec)
        signs_confs = np.array(
            [[1, 1, 1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]], dtype=int
        )
        trip_idx = 0
        for i in trange(self.n_img, desc="Computing signs_times_v"):
            for j in range(i + 1, self.n_img - 1):
                ij = self.pairs_to_linear[i, j]
                for k in range(j + 1, self.n_img):
                    ik = self.pairs_to_linear[i, k]
                    jk = self.pairs_to_linear[j, k]

                    best_i = J_list[trip_idx]
                    trip_idx += 1

                    s_ij_jk = signs_confs[best_i][0]
                    s_ik_jk = signs_confs[best_i][1]
                    s_ij_ik = signs_confs[best_i][2]

                    # Update multiplication
                    new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
                    new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
                    new_vec[ik] += s_ij_ik * vec[ij] + s_ik_jk * vec[jk]

        return new_vec

    ######################
    # Synchronize Colors #
    ######################

    def _sync_colors(self, Rijs):
        """
        At this point, we have obtained a hand-consistent set of 4-tuples of Rijs,
        with Rij = Ri.T @ g_s @ Rj, where s is an unknown permutation of (0, 1, 2, 3).
        Taking the average of the first Rij in the 4-tuple with the remaining 3
        results in a set of 3 outer products (+-)vi(m).T @ vj(m) of unknown ordering
        where vi(m) is the m'th row of the i'th rotation matrix, Ri.

        The color sync procedure partitions the set of 3-tuples of m'th row outer
        products into 3 sets of row-consistent outer products up to the sign of each.

        :param Rijs: Array of shape (n_pairs,4,3,3) consisting of the n_pairs of
            hand-consistent 4-tuples of Rijs.
        :returns:
            - cp, A color mapping vector of length (n_pairs * 3) which permutes
                the 3-tuples of `Rijs_rows` to be globally row-consistent.
            - Rijs_rows, An array of color synchronized rotations' rows outer products of
                shape (n_pairs, 3, 3, 3), where each Rijs_rows[ij] corresponds to a 3-tuple
                of m'th row outer product matrices, some of which having a spurious -1.
        """
        logger.info("Performing rotations' rows synchronization.")
        # Generate array of one rank matrices from which we can extract rows.
        # Matrices are of the form 0.5(Ri^TRj+Ri^TgkRj). Each such matrix can be
        # written in the form Qi^T*Ik*Qj where Ik is a 3x3 matrix with all zero
        # entries except for the entry a_kk, k in {1,2,3}.
        n_pairs = len(Rijs)
        Rijs_rows = np.zeros((n_pairs, 3, 3, 3), dtype=self.dtype)
        for layer in range(3):
            Rijs_rows[:, layer] = 0.5 * (Rijs[:, 0] + Rijs[:, layer + 1])

        # Partition the set of matrices Rijs_rows into 3 sets of matrices, where
        # each set there are only matrices Qi^T*Ik*Qj for a unique value of k in
        # {1,2,3}.
        # First determine for each pair of tuples of the form {Qi^T*Ik*Qj} and
        # {Qr^T*Il*Qj} where {i,j}\cap{r,l}==1, whether l==r.
        color_perms = self._match_colors(Rijs_rows)

        # Compute eigenvectors of color matrix. This is just a matrix of dimensions
        # 3(N choose 2)x3(N choose 2) where each entry corresponds to a pair of
        # matrices {Qi^T*Ir*Qj} and {Qr^T*Il*Qj} and equals \delta_rl.
        # The 2 leading eigenvectors span a linear subspace which contains a
        # vector which encodes the partition. All the entries of the vector are
        # either 1,0 or -1, where the number encodes which the index r in Ir.
        # This vector is a linear combination of the two leading eigen vectors,
        # and so we 'unmix' these vectors to retrieve it.
        color_mat = la.LinearOperator(
            (3 * n_pairs,) * 2, lambda v: self._mult_cmat_by_vec(color_perms, v)
        )

        # Seed eigs initial vector for iterative method.
        # scipy LinearOperator needs doubles for some architectures (arm).
        v0 = randn(3 * n_pairs, seed=self.seed).astype(np.float64, copy=False)

        v0 = v0 / norm(v0)
        vals, colors = la.eigs(color_mat, k=3, which="LR", v0=v0)
        vals = np.real(vals)
        colors = np.real(colors).astype(self.dtype, copy=False)
        colors = np.sign(colors[0]) * colors  # Stable eigs
        cp, _ = self._unmix_colors(colors[:, :2])

        return cp, Rijs_rows

    def _match_colors(self, Rijs_rows):
        """
        For each triplet of indices i < j < k, we consider the m'th row outer products stored
        as Rijs_rows, ie. Rijs_rows[ij], Rijs_rows[jk], and Rijs_rows[ik]. Recall that
        Rijs_rows[ij, n], n=0,1,2, corresponds to the 3x3 outer product vi_m.T @ vj_m, where
        vi_m is an unknown row of the rotation matrices Ri and Rj. For each triplet of these
        sets of row outer products this method finds a permutation sigma such that
        Rijs_rows[ij, sigma(n)], Rijs_rows[jk, sigma(n)], and Rijs_rows[ik, sigma(n)] all
        correspond to the same m'th row outer product.

        Framed as graph partioning problem we are coloring the vertices, Rijs_rows[ij, n],
        with three colors such that each color corresponds to the same row of the rotations
        Ris. This method returns the permutation that rearanges the elements of each triplet
        of Rijs to have matching color.

        :param Rijs_rows: An n_pairsx3x3x3 array of m'th row outer products for the pairs
            Ri, Rj, where Rijs_rows[:, i] is the m'th row outer product of unknown row m.
        :return: n_pairs length array corresponding to the permutation which color matches
            Rijs_rows.
        """
        Rijs_rows_t = np.transpose(Rijs_rows, (0, 1, 3, 2))
        trip_perms = np.array(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
            dtype="int",
        )
        inverse_perms = np.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [2, 0, 1],
                [1, 2, 0],
                [2, 1, 0],
            ],
            dtype="int",
        )

        m = np.zeros((6, 6), dtype=self.dtype)
        colors_i = np.zeros((len(self.triplets), 3), dtype=int)
        n_trip = len(self.triplets)
        votes = np.zeros((n_trip))
        trip_idx = 0

        # Compute relative color permutations. See Section 7.2 of paper.
        for i, j, k in self.triplets:
            ij = self.pairs_to_linear[i, j]
            jk = self.pairs_to_linear[j, k]
            ik = self.pairs_to_linear[i, k]

            # For r=1:3 compute 3*3 products v_{ji}(r)v_{ik}v_{kj}
            prod_arr = Rijs_rows[ik, None] @ Rijs_rows_t[jk, :, None]
            prod_arr_tmp = prod_arr.copy()
            prod_arr = Rijs_rows_t[ij, :, None] @ prod_arr_tmp.reshape((9, 3, 3))[None]
            prod_arr = np.transpose(
                prod_arr.reshape((3, 3, 3, 9), order="F"), (2, 1, 0, 3)
            )

            # Compare to v_{jj}(r)=v_{ji}v_{ij}.
            self_prods = Rijs_rows_t[ij] @ Rijs_rows[ij]
            self_prods = self_prods.reshape(3, 9)

            prod_arr1 = prod_arr.copy()
            prod_arr1 -= self_prods
            norms1 = np.sum(prod_arr1**2, axis=3)

            prod_arr2 = prod_arr.copy()
            prod_arr2 += self_prods
            norms2 = np.sum(prod_arr2**2, axis=3)

            # Compare to v_{jj}(r)=v_{jk}v_{kj}.
            self_prods = Rijs_rows[jk] @ Rijs_rows_t[jk]
            self_prods = self_prods.reshape(3, 9)

            prod_arr1 = prod_arr.copy()
            prod_arr1 -= self_prods[:, None, None]
            norms1 = norms1 + np.sum(prod_arr1**2, axis=3)

            prod_arr2 = prod_arr.copy()
            prod_arr2 += self_prods[:, None, None]
            norms2 = norms2 + np.sum(prod_arr2**2, axis=3)

            # For r=1:3 compute 3*3 products v_{ij}(r)v_{jk}v_{ki} and compare to
            # Compare to v_{ii}(r)=v_{ij}v_{ji}
            prod_arr = np.transpose(prod_arr_tmp, (0, 1, 3, 2))
            prod_arr = Rijs_rows[ij, :, None] @ prod_arr.reshape((9, 3, 3))[None]
            prod_arr = np.transpose(
                prod_arr.reshape((3, 3, 3, 9), order="F"), (1, 0, 2, 3)
            )

            # Compare to v_{ii}(r)=v_{ik}v_{ki}.
            self_prods = Rijs_rows[ik] @ Rijs_rows_t[ik]
            self_prods = self_prods.reshape(3, 9)

            prod_arr1 = prod_arr.copy()
            prod_arr1 -= self_prods[None, :, None]
            norms1 = norms1 + np.sum(prod_arr1**2, axis=3)

            prod_arr2 = prod_arr.copy()
            prod_arr2 += self_prods[None, :, None]
            norms2 = norms2 + np.sum(prod_arr2**2, axis=3)

            norms = np.minimum(norms1, norms2)

            for r in range(6):
                p1 = trip_perms[r]
                for s in range(6):
                    p2 = trip_perms[s]
                    m[r, s] = (
                        norms[p2[0], p1[0], 0]
                        + norms[p2[1], p1[1], 1]
                        + norms[p2[2], p1[2], 2]
                    )

            # In the event of duplicate min values min_idx is the first occurence
            # by column order to match matlab outputs.
            min_idx = np.unravel_index(np.argmin(m.T), m.shape)[::-1]
            votes[trip_idx] = m[min_idx]

            # Store permutation indices as digits of a base 10 number.
            colors_i[trip_idx, :2] = [
                100 * (min_idx[0] + 1),
                10 * (min_idx[1] + 1),
            ]

            # Calculate the relative permutation of Rik to Rij given
            # by (sigma_ik)\circ(sigma_ij)^-1
            inv_jk_perm = inverse_perms[min_idx[1]]
            rel_perm = trip_perms[min_idx[0]]
            rel_perm = rel_perm[inv_jk_perm]
            colors_i[trip_idx, 2] = (2 * (rel_perm[0] + 1) - 1) + (
                rel_perm[1] > rel_perm[2]
            )
            trip_idx += 1

        colors_i = np.sum(colors_i, axis=1)

        return colors_i

    def _mult_cmat_by_vec(self, c_perms, v):
        """
        Multiply color matrix by vector v "on the fly".

        :param c_perms: An (N over 3) vector. Each corresponds to a triplet of
            indices i<j<k and indicates the relative permutation of tuples
            {Qi^TIrQj},{Qj^TIlQk} and {Qk^TIlQi}. The color matrix can be
            completely reconstructed from this information, which is used
            here to execute a single multiplication of the matrix by the
            vector v instead of explicitely computing and storing the prohibitively
            large color matrix in memory.
        :param v: vector to be multiplied by 'color matrix'.
        """
        t_perms = np.array(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        )
        i_perms = np.array(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]]
        )
        out = np.zeros_like(v)
        trip_idx = 0
        for i in range(self.n_img):
            for j in range(i + 1, self.n_img - 1):
                ij_block = 3 * self.pairs_to_linear[i, j]
                for k in range(j + 1, self.n_img):
                    ik_block = 3 * self.pairs_to_linear[i, k]
                    jk_block = 3 * self.pairs_to_linear[j, k]

                    # Extract permutation indices from c_perms
                    n = c_perms[trip_idx]
                    trip_idx += 1
                    p_n1 = n // 100
                    p_n3 = n % 10
                    p_n2 = (n - p_n1 * 100 - p_n3) // 10

                    # Adjust for 0-based indexing. (Take this out by computing c_perms with 0-base)
                    p_n1 = (p_n1 - 1).astype("int")
                    p_n2 = (p_n2 - 1).astype("int")
                    p_n3 = (p_n3 - 1).astype("int")

                    # Multiply vector by color matrix

                    # Upper triangular part
                    p = t_perms[p_n1] + ik_block
                    out[ij_block] = out[ij_block] - v[p[1]] - v[p[2]] + v[p[0]]
                    out[ij_block + 1] = out[ij_block + 1] - v[p[0]] - v[p[2]] + v[p[1]]
                    out[ij_block + 2] = out[ij_block + 2] - v[p[0]] - v[p[1]] + v[p[2]]

                    p = t_perms[p_n2] + jk_block
                    out[ij_block] = out[ij_block] - v[p[1]] - v[p[2]] + v[p[0]]
                    out[ij_block + 1] = out[ij_block + 1] - v[p[0]] - v[p[2]] + v[p[1]]
                    out[ij_block + 2] = out[ij_block + 2] - v[p[0]] - v[p[1]] + v[p[2]]

                    p = i_perms[p_n3] + jk_block
                    out[ik_block] = out[ik_block] - v[p[1]] - v[p[2]] + v[p[0]]
                    out[ik_block + 1] = out[ik_block + 1] - v[p[0]] - v[p[2]] + v[p[1]]
                    out[ik_block + 2] = out[ik_block + 2] - v[p[0]] - v[p[1]] + v[p[2]]

                    # Lower triangular part
                    p = i_perms[p_n1] + ij_block
                    out[ik_block] = out[ik_block] - v[p[1]] - v[p[2]] + v[p[0]]
                    out[ik_block + 1] = out[ik_block + 1] - v[p[0]] - v[p[2]] + v[p[1]]
                    out[ik_block + 2] = out[ik_block + 2] - v[p[0]] - v[p[1]] + v[p[2]]

                    p = i_perms[p_n2] + ij_block
                    out[jk_block] = out[jk_block] - v[p[1]] - v[p[2]] + v[p[0]]
                    out[jk_block + 1] = out[jk_block + 1] - v[p[0]] - v[p[2]] + v[p[1]]
                    out[jk_block + 2] = out[jk_block + 2] - v[p[0]] - v[p[1]] + v[p[2]]

                    p = t_perms[p_n3] + ik_block
                    out[jk_block] = out[jk_block] - v[p[1]] - v[p[2]] + v[p[0]]
                    out[jk_block + 1] = out[jk_block + 1] - v[p[0]] - v[p[2]] + v[p[1]]
                    out[jk_block + 2] = out[jk_block + 2] - v[p[0]] - v[p[1]] + v[p[2]]
        return out

    def _unmix_colors(self, color_vecs):
        """
        The 'color vector' which partitions the rank 1 3x3 matrices into 3 sets
        is one of 2 leading orthogonal eigenvectors of the color matrix.
        SVD retrieves two orthogonal linear combinations of these vectors which
        can be 'unmixed' to retrieve the color vector by finding a suitable
        2D rotation of these vectors (see Section 7.3 of D2 paper for details).
        """
        n_p = color_vecs.shape[0] // 3
        d_theta = self.n_theta // self.n_theta
        max_t = self.n_theta // d_theta + 1

        def R_theta(theta):
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                dtype=self.dtype,
            )
            return R

        s = float("inf")
        scores = np.zeros(max_t, dtype=self.dtype)
        idx = 0
        for t in np.arange(0, max_t, 0.5):
            unmix_ev = color_vecs @ R_theta(np.pi * t / 180)
            s1 = unmix_ev[:, 0].reshape(n_p, 3)
            p11 = (-s1).argsort(axis=1)  # descending argsort
            s1 = np.take_along_axis(s1, p11, axis=1)
            score11 = np.sum((s1[:, 0] + s1[:, 2]) ** 2 + s1[:, 1] ** 2)

            s2 = abs(unmix_ev[:, 1].reshape(n_p, 3))
            p12 = (-s2).argsort(axis=1)  # descending argsort
            s2 = np.take_along_axis(s2, p12, axis=1)
            score12 = np.sum(
                (s2[:, 0] - 2 * s2[:, 1]) ** 2
                + (s2[:, 0] - 2 * s2[:, 2]) ** 2
                + (s2[:, 1] - s2[:, 2]) ** 2
            )  # Matlab comment: Is this an error??? + instead of - in the first 2 members

            s1 = abs(unmix_ev[:, 0].reshape(n_p, 3))
            p12 = (-s1).argsort(axis=1)  # descending argsort
            s1 = np.take_along_axis(s1, p11, axis=1)
            score22 = np.sum(
                (s1[:, 0] - 2 * s1[:, 1]) ** 2
                + (s1[:, 0] - 2 * s1[:, 2]) ** 2
                + (s1[:, 1] - s1[:, 2]) ** 2
            )

            s2 = unmix_ev[:, 1].reshape(n_p, 3)
            p22 = (-s2).argsort(axis=1)  # descending argsort
            s2 = np.take_along_axis(s2, p12, axis=1)
            score21 = np.sum((s2[:, 0] + s2[:, 2]) ** 2 + s2[:, 1] ** 2)

            score_vecs = [score11 + score12, score21 + score22]
            which_vec = np.argmin([score11 + score12, score21 + score22])
            scores[idx] = score_vecs[which_vec]
            if scores[idx] < s:
                s = scores[idx]
                if which_vec == 0:
                    p = p11
                else:
                    p = p22
                best_unmix = unmix_ev[:, which_vec]

            # Assign integers between 1:3 to permutations
            colors = np.zeros((n_p, 3), dtype=int)
            for i in range(n_p):
                p_i = p[i]
                p_i_sqr = p_i[p_i]
                if np.sum((p_i_sqr - [0, 1, 2]) ** 2) == 0:  # non-cyclic permutation
                    colors[i] = p_i
                else:
                    colors[i] = p_i_sqr
            colors = colors.flatten()

        return colors, best_unmix

    #####################
    # Synchronize Signs #
    #####################

    def _sync_signs(self, rr, c_vec):
        """
        This function executes the final stage of the algorithm, Signs
        synchroniztion. At this point, we have rotation rows
        rr[ij, m] = sij_m * vi_m.T @ vj_m, where vi_m, vj_m are the m'th rows
        of rotation matrices Ri and Rj and sij_m is an unknown sign. This method
        uses the permutation vector, `c_vec`, to partition the rotation row
        outer products and constructs a symmetric block matrix, H, with ij'th block
        sij * vi.T @ vj. The signs sij are then adjusted so that H is rank-1. This
        matrix is then factored to extract the rows of each rotation matrix. At the
        end all rows of the rotations Ri are exctracted and the matrices Ri are assembled.

        :param rr: Array of color synchronized rotations' rows outer products of
            shape (n_pairs, 3, 3, 3), where each rr[ij] corresponds to a 3-tuple
            of m'th row outer product matrices, some of which having a spurious -1.
        :param c_vec: A color mapping vector of length (n_pairs * 3) which permutes
            the 3-tuples of `rr` to be globally row-consistent.
        :return: n_img x 3 x 3 array of rotation matrices.
        """
        logger.info("Performing signs synchronization.")
        c_mat, c_mat_5d, c_mat_4d = self._construct_color_mats(rr, c_vec)

        sync_signs2 = self._compute_signs(c_mat_5d, c_mat_4d)

        rows_arr = self._estimate_rows(sync_signs2, c_mat_5d)

        signs = self._compute_signs_adjustment(rows_arr)

        rots = self._extract_rotations(c_mat, signs)

        return rots

    def _construct_color_mats(self, rr, c_vec):
        """
        Construct the partitioned row synchronized color matrices, `c_mat`, where
        c_mat[m] contains the 3x3 blocks sij*vi_m.T @ vj_m, where vi_m is the m'th
        row of the i'th rotation Ri and sij is the unknown sign.

        :param rr: Non-partitioned rotation row matrices.
        :param c_vec: Color partition vector.
        :return: Partitioned row synchronized color matrices.
        """
        # Partition the union of tuples {0.5*(Ri^TRj+Ri^TgkRj), k=1:3} according
        # to the color partition established in color synchronization procedure.
        # The partition is stored in two different arrays each with the purpose
        # of a computational speed up for two different computations performed
        # later (space considerations are of little concern since arrays are ~
        # o(N^2) which doesn't pose a constraint for inputs on the scale of 10^3-10^4.
        c_mat_5d = np.zeros((self.n_img, self.n_img, 3, 3, 3), dtype=self.dtype)
        c_mat_4d = np.zeros((self.n_pairs, 3, 3, 3), dtype=self.dtype)
        c_vec = c_vec.reshape(self.n_pairs, 3)
        for i in range(self.n_img - 1):
            for j in range(i + 1, self.n_img):
                ij = self.pairs_to_linear[i, j]
                c_mat_5d[i, j, c_vec[ij]] = rr[ij]
                c_mat_5d[j, i, c_vec[ij]] = rr[ij].transpose(0, 2, 1)
                c_mat_4d[ij, c_vec[ij]] = rr[ij]

        # Compute estimates for the tuples {0.5*(Ri^TRi+Ri^TgkRi), k=1:3} for
        # i=1:N. For 1<=i,j<=N and c=1,2,3 write Qij^c=0.5*(Ri^TRj+Ri^TgmRj).
        # For each i in {1:N} and each k in {1,2,3} the estimator is the
        # average over all j~=i of Qij^c*(Qij^c)^T.
        # Since in practice the result of the average is not really rank 1, we
        # compute the best rank approximation to this average.
        for i in range(self.n_img):
            for c in range(3):
                Rijs = c_mat_5d[i, :, c]
                Rijs = np.delete(Rijs, i, axis=0)
                Rii_est = Rijs @ np.transpose(Rijs, (0, 2, 1))
                Rii = np.mean(Rii_est, axis=0)
                U, _, _ = np.linalg.svd(Rii)
                c_mat_5d[i, i, c] = np.outer(U[:, 0], U[:, 0])

        # Construct the 3Nx3N row synchroniztion matrices (as done for C_2), one
        # for all first rows of the matrices Ri, one for all second rows and one
        # for all third rows. The ij'th block of the k'th matrix is Qij^c.
        # In C_2 one such matrix is constructed for the 3rd rows
        # and is rank 1 by construction. In practice, thus far, for each c and
        # (i,j) we either have Qij^c or -Qij^c independently.
        c_mat = np.zeros((3, self.n_img, 3, self.n_img, 3), dtype=self.dtype)
        for i in range(self.n_img - 1):
            for j in range(i + 1, self.n_img):
                ij = self.pairs_to_linear[i, j]
                c_mat[c_vec[ij], i, :, j, :] = rr[ij]

        c_mat = c_mat + c_mat.transpose(0, 3, 4, 1, 2)

        for c in range(3):
            for i in range(self.n_img):
                c_mat[c, i, :, i, :] = c_mat_5d[i, i, c]

        return c_mat, c_mat_5d, c_mat_4d

    def _compute_signs(self, c_mat_5d, c_mat_4d):
        """
        Compute signs for adjusting `c_mat` to be composed of all rank-1 3x3 blocks.
        """
        # To decompose cMat as a rank 1 matrix we need to adjust the signs of the
        # Qij^c so that sign(Qij^c*Qjk^c) = sign(Qik^c) for all c=1,2,3 and (i,j).
        # In practice we compare the sign of the sum of the entries of Qij^c*Qjk^c
        # to the sum of entries of Qik^c.

        # For computational comfort the signs for each c=1,2,3 are stored in a
        # Nx(N over 2) array, where the ij'th column corresponds to the signs of
        # Qij^c * Qjk^c for k~=i,j. The entries in the k=i,j rows of the ij'th
        # column are zero, the value zero is arbitrary, since these entries are
        # not used by the algorithm, and only exist for comfort (of storage and
        # access).
        signs = np.zeros((3, self.n_pairs, self.n_img), dtype=self.dtype)
        for c in range(3):
            for p in range(self.n_pairs):
                i, j = self.pairs[p]
                idx_mask = np.full(self.n_img, True)
                idx_mask[[i, j]] = False
                signs[c, p, idx_mask] = self._calc_Rij_prods(c_mat_5d, i, j, c)

        # Now compute the signs of Qij^c.
        est_signs = np.sign(np.sum(c_mat_4d, axis=(-2, -1)))
        signs = np.transpose(signs, (0, 2, 1))
        for c in range(3):
            signs[c] = est_signs[:, c] * signs[c]

        # Qik^c can be compared with Qir^c*Qrk^c for each r~=i,k, that is,
        # N-2 options. Another way to look at this, is that the r'th image
        # participates in all comparisons of the form sign(Qir^c*Qrk^c)~sign(Qik)
        # for r~=i,k for each c=1,2,3 (see Section 8 in D2 paper).
        # For each image r construct a 3Nx3N matrix. If
        # sign(Qir^c*Qrk^c)~sign(Qik)=1, its ik'th 3x3 block is set to Qik,
        # otherwise, it is set to -Qik.
        sync_signs2 = np.arange(self.n_img).reshape((1, 1, self.n_img, 1))
        sync_signs2 = np.tile(sync_signs2, (3, self.n_img, 1, self.n_img))
        for c in range(3):
            for r in range(self.n_img):
                # Fill signs for synchroniztion for the r'th image.
                # Go over all i,j~=r.
                i_idx = np.concatenate(
                    (np.arange(0, r), np.arange(r + 1, self.n_img))
                )  # i~=r
                for i in i_idx:
                    if i <= r:
                        j_idx = np.concatenate(
                            (np.arange(i + 1, r), np.arange(r + 1, self.n_img))
                        )
                    else:
                        j_idx = np.arange(i + 1, self.n_img)
                    for j in j_idx:
                        ij = self.pairs_to_linear[i, j]
                        sync_signs2[c, r, j, i] = (
                            j + 0.5 * (1 - signs[c, r, ij]) * self.n_img
                        )
                        sync_signs2[c, r, i, j] = (
                            i + 0.5 * (1 - signs[c, r, ij]) * self.n_img
                        )
                        # The function (1-x)/2 maps 1->0 and -1->1

        return sync_signs2

    def _estimate_rows(self, sync_signs2, c_mat_5d):
        """
        Construct 3N x 3N matrix of rank-1 3x3 blocks of sij*vi_m.T @ vj_m,
        the leading eigenvectors of which correspond to estimates for the rows
        of the rotations Ri, up to signs.
        """
        c_mat_5d_mp = np.concatenate((c_mat_5d, -c_mat_5d), axis=1)
        rows_arr = np.zeros((3, self.n_img, 3 * self.n_img), dtype=self.dtype)
        svals = np.zeros((3, 2, self.n_img), dtype=self.dtype)

        logger.info("Constructing and decomposing N sign synchronization matrices...")
        for c in range(3):
            for r in range(self.n_img):
                # Image r used for signs.
                c_mat_eff = self._fill_sign_sync_matrix_c(
                    c_mat_5d_mp, sync_signs2, c, r
                )

                # Construct (3*N)x(3*N) rank 1 matrices from Qik
                c_mat_for_svd = np.zeros(
                    (3 * self.n_img, 3 * self.n_img), dtype=self.dtype
                )
                for i in range(self.n_img):
                    row_3Nx3 = c_mat_eff[i]
                    row_3Nx3 = row_3Nx3.reshape(3 * self.n_img, 3)
                    c_mat_for_svd[:, 3 * i : 3 * i + 3] = row_3Nx3

                c_mat_for_svd = c_mat_for_svd + c_mat_for_svd.T

                # Extract leading eigenvector of rank 1 matrix. For each r and c
                # this gives an estimate for the c'th row of the rotation Rr, up
                # to sign +/-.
                for i in range(self.n_img):
                    c_mat_for_svd[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = c_mat_eff[
                        i, i
                    ]
                U, S, _ = np.linalg.svd(c_mat_for_svd)
                svals[c, :, r] = S[:2]
                rows_arr[c, r] = U[:, 0]

        return rows_arr

    def _compute_signs_adjustment(self, rows_arr):
        """
        Compute signs adjustment vector.
        """
        # Sync signs according to results for each image. Dot products between
        # signed row estimates are used to construct an (N over 2)x(N over 2)
        # sign synchronization matrix S. If (v_i)k and (v_j)k are the i'th and
        # j'th estimates for the c'th row of Rk, then the entry (i,k),(k,j) entry
        # of S is <(v_i)k,(v_j)k>, where the rows and columns of S are indexed by
        # double indexes (i,j), 1<=i<j<=(N over 2).
        pairs_map = np.zeros((self.n_pairs, 2 * (self.n_img - 2)), dtype=int)
        for i in range(self.n_img):
            for j in range(i + 1, self.n_img):
                ij = self.pairs_to_linear[i, j]
                pairs_map[ij] = np.concatenate(
                    (
                        self.pairs_to_linear[:i, i],
                        self.pairs_to_linear[i, np.r_[i + 1 : j, j + 1 : self.n_img]],
                        self.pairs_to_linear[np.r_[:i, i + 1 : j], j],
                        self.pairs_to_linear[j, j + 1 :],
                    )
                )

        signs = np.zeros((3, self.n_pairs), dtype=self.dtype)
        s_out = np.zeros((3, 3), dtype=self.dtype)

        logger.info("Constructing and decomposing 3 sign synchroniztion matrices.")
        # The matrix S requires space on order of O(N^4). Instead of storing it
        # in memory we compute its SVD using the function smat which multiplies
        # (N over 2)x1 vectors by S.
        for c in range(3):
            # Prepare data for smat to act on vectors.
            sign_mat = np.zeros((self.n_pairs, 2 * (self.n_img - 2)), dtype=int)
            for i in range(self.n_img - 1):
                for j in range(i + 1, self.n_img):
                    ij = self.pairs_to_linear[i, j]
                    sij = rows_arr[c, j, 3 * i : 3 * i + 3]
                    sji = rows_arr[c, i, 3 * j : 3 * j + 3]
                    siks = rows_arr[
                        c, np.r_[:i, i + 1 : j, j + 1 : self.n_img], 3 * i : 3 * i + 3
                    ]
                    sjks = rows_arr[
                        c, np.r_[:i, i + 1 : j, j + 1 : self.n_img], 3 * j : 3 * j + 3
                    ]
                    sign_mat[ij] = np.concatenate(
                        (np.sign(siks @ sij), np.sign(sjks @ sji))
                    )

            smat = la.LinearOperator(
                shape=(self.n_pairs, self.n_pairs),
                matvec=lambda v, s=sign_mat: self._mult_smat_by_vec(v, s, pairs_map),
                rmatvec=lambda v, s=sign_mat: self._mult_smat_by_vec(v, s, pairs_map),
            )
            U, S, _ = la.svds(smat, k=3, which="LM")
            U = np.sign(U[0]) * U  # Stable svds
            signs[c] = U[:, -1]  # svds returns in ascending order
            s_out[c] = S[::-1]

        return np.sign(signs)

    def _extract_rotations(self, c_mat, signs):
        """
        Adjust the signs of each block of `c_mat` then extract the rotation
        rows and construct the estimated rotations.

        :param c_mat: The color synchronization matrix.
        :param signs: The signs adjustment matrix.
        :return: Estimated rotations.
        """
        # Adjust the signs of Qij^c in the matrices cMat(:,:,c) for all c=1,2,3
        # and 1<=i<j<=N according to the results of the signs from the last stage.
        logger.info("Constructing and decomposing 3 row synchroniztion matrices.")
        for c in range(3):
            idx = 0
            for i in range(self.n_img - 1):
                for j in range(i + 1, self.n_img):
                    c_mat[c, j, :, i, :] *= signs[c, idx]
                    c_mat[c, i, :, j, :] *= signs[c, idx]
                    idx += 1

        # cMat(:,:,c) are now rank 1. Decompose using SVD and take leading eigenvector.
        c_mat = c_mat.reshape(3, 3 * self.n_img, 3 * self.n_img)
        U1, _, _ = la.svds(c_mat[0], k=3, which="LM")
        U2, _, _ = la.svds(c_mat[1], k=3, which="LM")
        U3, _, _ = la.svds(c_mat[2], k=3, which="LM")

        # Stabilize and take leading eigenvector.
        U1 = np.sign(U1[0, -1]) * U1[:, -1]
        U2 = np.sign(U2[0, -1]) * U2[:, -1]
        U3 = np.sign(U3[0, -1]) * U3[:, -1]

        # The c'th row of the rotation Rj is Uc(3*j-2:3*j,1)/norm(Uc(3*j-2:3*j,1)),
        # (Rows must be normalized to length 1).
        logger.info("Assembeling rows to rotations matrices.")
        rot = np.zeros((self.n_img, 3, 3), dtype=self.dtype)
        rot[:, 0] = U1.reshape(self.n_img, 3)
        rot[:, 1] = U2.reshape(self.n_img, 3)
        rot[:, 2] = U3.reshape(self.n_img, 3)
        rot /= np.linalg.norm(rot, axis=-1)[:, :, None]

        # Ensure we have rotations.
        not_a_rot = np.argwhere(np.linalg.det(rot) < 0)
        rot[not_a_rot, 2] *= -1

        return rot

    def _fill_sign_sync_matrix_c(self, c_mat_5d_mp, sync_signs2, c, img):
        c_mat_eff = np.zeros((self.n_img, self.n_img, 3, 3), dtype=self.dtype)
        for r in range(self.n_img):
            c_mat_eff[:, r] = c_mat_5d_mp[r, sync_signs2[c, img, :, r], c]
        return c_mat_eff

    def _calc_Rij_prods(self, c_mat_5d, i, j, c):
        Rik = np.delete(c_mat_5d[i, :, c], [i, j], axis=0)
        Rkj = np.delete(c_mat_5d[:, j, c], [i, j], axis=0)
        Rij = Rik @ Rkj

        # In case we get a zero score arbitrarily choose sign +1.
        ij_signs = np.sum(Rij, axis=(-2, -1))
        zeros_idx = ij_signs == 0
        if np.count_nonzero(zeros_idx) > 0:
            ij_signs[zeros_idx] = 1

        return np.sign(ij_signs)

    def _mult_smat_by_vec(self, v, sign_mat, pairs_map):
        """
        Multiplies the signs sync matrix by a vector.
        """
        v_out = np.zeros_like(v)
        for i in range(self.n_img):
            for j in range(i + 1, self.n_img):
                ij = self.pairs_to_linear[i, j]
                v_out[ij] = sign_mat[ij] @ v[pairs_map[ij]]
        return v_out

    ####################
    # Helper Functions #
    ####################

    @staticmethod
    def _circ_seq(n1, n2, L):
        """
        For integers 0 <= n1, n2 < L, make a circular sequence of integers between
        n1 and n2 modulo L.

        :param n1: First integer in sequence.
        :param n2: Last integer in sequence.
        :param L: Modulus of values in sequence.
        :return: Circular sequence modulo L.
        """
        if min(n1, n2) < 0 or max(n1, n2) >= L:
            raise ValueError(
                f"n1 and n2 must both be in [0, {L}). Found n1={n1}, n2={n2}."
            )
        if n2 < n1:
            n2 += L
        if n1 == n2:
            return np.array([n1]).astype(int) % L

        seq = np.arange(n1, n2 + 1).astype(int) % L

        return seq

    @staticmethod
    def _saff_kuijlaars(N):
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
    def _mark_equators(sphere_grid, eq_filter_angle):
        """
        This method categorizes a set of 3D unit vectors into equator and non-equator
        vectors determined by the parameter `eq_filter_angle`, returned as `eq_idx`.
        It further categorizes the vectors into the classes non_equator, z-equator,
        y-equator, x-equator, z-top_view, y-top_view, and x-top_view, which are labeled
        respectively with the values 0 - 6 and returned as `eq_class`.

        :param sphere_grid: Nx3 array of vertices in cartesian coordinates.
        :param eq_filter_angle: Angular distance from equator to be marked as
            an equator point.

        :return: eq_class, n_rots length array of values indicating equator class.
        """
        # Project each vector onto xy, xz, yz planes and measure angular distance
        # from each plane.
        n_rots = len(sphere_grid)
        angular_dists = np.zeros((n_rots, 3), dtype=sphere_grid.dtype)

        # For each grid point get the distance from the z, y, and x-axis equators.
        for i in range(3):
            proj_along_axis = sphere_grid.copy()
            proj_along_axis[:, 2 - i] = 0
            proj_along_axis /= np.linalg.norm(proj_along_axis, axis=1)[:, None]
            angular_dists[:, i] = np.sum(sphere_grid * proj_along_axis, axis=-1)

        # Mark all views close to an equator.
        eq_min_dist = np.cos(eq_filter_angle * np.pi / 180)
        n_eqs = np.count_nonzero(angular_dists > eq_min_dist, axis=1)

        # Classify equators.
        # 0 -> non-equator view
        # 1 -> z equator
        # 2 -> y equator
        # 3 -> x equator
        # 4 -> z top view
        # 5 -> y top view
        # 6 -> x top view
        eq_class = np.zeros(n_rots)

        # Grid points which are equator points with respect to 2 equators are considered top views.
        # For example, a grid point that is close to both the x and y equator is a z top view.
        top_view_idx = n_eqs > 1
        top_view_class = np.argmin(angular_dists[top_view_idx] > eq_min_dist, axis=1)
        eq_class[top_view_idx] = top_view_class + 4

        # Assign grid points which are equator points with respect to only 1 equator.
        eq_view_idx = n_eqs == 1
        eq_view_class = np.argmax(angular_dists[eq_view_idx] > eq_min_dist, axis=1)
        eq_class[eq_view_idx] = eq_view_class + 1

        return eq_class

    @staticmethod
    def _generate_inplane_rots(sphere_grid, d_theta):
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
        # Negative signs to match matlab.
        inplane_rots = Rotation.about_axis(
            "z", np.arange(0, -2 * np.pi, -d_theta), dtype=dtype
        ).matrices
        n_inplane_rots = len(inplane_rots)

        # Generate in-plane rotations of rots_grid.
        inplane_rotated_grid = np.zeros((n_rots, n_inplane_rots, 3, 3), dtype=dtype)
        for i in range(n_rots):
            inplane_rotated_grid[i] = rots_grid[i] @ inplane_rots

        return inplane_rotated_grid

    def _generate_commonline_indices(self, cl_angles):
        """
        Converts a multi-dimensional stack of pairs of commonline angles in [0, 360) degrees
        into a flattened stack of polar Fourier linear indices, with the convention that
        each linear index corresponds to an unraveled index in [0, n_theta // 2) x [0, n_theta).

        :param cl_angles: A multi-dimensional stack of commonline angles in degrees, shape (..., 2).
        :return: cl_idx, a 1D array of linear indices.
        """
        L = self.n_theta

        # Flatten the stack
        og_shape = cl_angles.shape
        cl_angles = np.reshape(cl_angles, (np.prod(og_shape[:-1]), 2))

        # Fourier ray index
        row_sub = np.round(cl_angles[:, 0] * L / 360).astype("int") % L
        col_sub = np.round(cl_angles[:, 1] * L / 360).astype("int") % L

        # Restrict Ri in-plane coordinates to <180 degrees.
        is_geq_than_pi = row_sub >= L // 2
        row_sub[is_geq_than_pi] = row_sub[is_geq_than_pi] - L // 2
        col_sub[is_geq_than_pi] = (col_sub[is_geq_than_pi] + (L // 2)) % L

        # Convert to linear indices in 180x360 correlation matrix.
        cl_idx = np.ravel_multi_index((row_sub, col_sub), dims=(L // 2, L))

        return cl_idx
