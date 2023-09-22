import logging

import numpy as np

from aspire.abinitio import CLOrient3D
from aspire.utils import Rotation

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
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
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
        self.eq_min_dist = eq_min_dist
        self.seed = seed

    def estimate_rotations(self):
        """
        Estimate rotation matrices for molecules with C3 or C4 symmetry.

        :return: Array of rotation matrices, size n_imgx3x3.
        """
        pass

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
        eq_mask1, top_view_mask1 = self.mark_equators(sphere_grid1, self.eq_min_dist)
        eq_mask2, top_view_mask2 = self.mark_equators(sphere_grid2, self.eq_min_dist)

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

        # Remove top views from sphere grids and update equator masks.
        sphere_grid1 = sphere_grid1[~top_view_mask1]
        sphere_grid2 = sphere_grid2[~top_view_mask2]
        eq_mask1 = eq_mask1[~top_view_mask1]
        eq_mask2 = eq_mask2[~top_view_mask2]

        # Generate in-plane rotations for each grid point on the sphere.

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

        :return: Indices of points on sphere whose distance from one of
            the equators is < eq_filter angle.
        """
        # Project each vector onto xy, xz, yz planes and measure angular distance
        # from each plane.
        eq_min_dist = np.cos(eq_filter_angle * np.pi / 180)

        # Mask for z-axis equator views.
        proj_xy = sphere_grid.copy()
        proj_xy[:, 2] = 0
        proj_xy /= np.linalg.norm(proj_xy, axis=1)[:, None]
        ang_dists_xy = np.sum(sphere_grid * proj_xy, axis=-1)
        z_eq_mask = ang_dists_xy > eq_min_dist

        # Mask for y-axis equator views.
        proj_xz = sphere_grid.copy()
        proj_xz[:, 1] = 0
        proj_xz /= np.linalg.norm(proj_xz, axis=1)[:, None]
        ang_dists_xz = np.sum(sphere_grid * proj_xz, axis=-1)
        y_eq_mask = ang_dists_xz > eq_min_dist

        # Mask for x-axis equator views.
        proj_yz = sphere_grid.copy()
        proj_yz[:, 0] = 0
        proj_yz /= np.linalg.norm(proj_yz, axis=1)[:, None]
        ang_dists_yz = np.sum(sphere_grid * proj_yz, axis=-1)
        x_eq_mask = ang_dists_yz > eq_min_dist

        # Mask for all views close to an equator.
        eq_mask = z_eq_mask | y_eq_mask | x_eq_mask

        # Top view masks.
        # A top view is a view along an axis of symmetry (ie. x, y, or z).
        # A top view is also at the intersection of the two equator views
        # perpendicular to the axis of symmetry.
        z_top_view_mask = y_eq_mask & x_eq_mask
        y_top_view_mask = z_eq_mask & x_eq_mask
        x_top_view_mask = z_eq_mask & y_eq_mask
        top_view_mask = z_top_view_mask | y_top_view_mask | x_top_view_mask

        return eq_mask, top_view_mask

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
        Ri2 = np.column_stack((-sphere_grid[:, 2], sphere_grid[:, 1], np.zeros(n_rots)))
        Ri2 /= np.linalg.norm(Ri2, axis=1)[:, None]
        Ri1 = np.cross(Ri2, sphere_grid)
        Ri1 /= np.linalg.norm(Ri1, axis=1)[:, None]

        rots_grid = np.zeros((n_rots, 3, 3), dtype=dtype)
        rots_grid[:, :, 0] = Ri1
        rots_grid[:, :, 1] = Ri2
        rots_grid[:, :, 2] = sphere_grid

        # Generate in-plane rotations.
        d_theta *= np.pi / 180
        inplane_rots = Rotation.about_axis(
            "z", np.arange(0, 2 * np.pi, d_theta), dtype=dtype
        ).matrices
        n_inplane_rots = len(inplane_rots)

        # Generate in-plane rotations of rots_grid.
        inplane_rotated_grid = np.zeros((n_rots, n_inplane_rots, 3, 3), dtype=dtype)
        for i in range(n_rots):
            inplane_rotated_grid[i] = rots_grid[i] @ inplane_rots

        return inplane_rotated_grid
