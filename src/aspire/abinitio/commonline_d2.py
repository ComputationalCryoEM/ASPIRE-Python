import logging

import numpy as np

from aspire.abinitio import CLOrient3D

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
        n_rots = len(sphere_grid)
        angular_dists = np.zeros(3, n_rots, dtype=sphere_grid.dtype)

        proj_xy = sphere_grid.copy()
        proj_xy[:, 2] = 0
        proj_xy /= np.linalg.norm(proj_xy, axis=1)[:, None]
        angular_dists[0] = np.sum(sphere_grid * proj_xy, axis=-1)

        proj_xz = sphere_grid.copy()
        proj_xz[:, 1] = 0
        proj_xz /= np.linalg.norm(proj_xz, axis=1)[:, None]
        angular_dists[1] = np.sum(sphere_grid * proj_xz, axis=-1)

        proj_yz = sphere_grid.copy()
        proj_yz[:, 0] = 0
        proj_yz /= np.linalg.norm(proj_yz, axis=1)[:, None]
        angular_dists[2] = np.sum(sphere_grid * proj_yz, axis=-1)

        # Mark points close to equator (within eq_filter_angle).
        eq_min_dist = np.cos(eq_filter_angle * np.pi / 180)
        n_eqs_close = np.sum(angular_dists > eq_min_dist, axis=0)
        eq_mask = n_eqs_close > 0

        # Classify equators.
        # 1 -> z equator
        # 2 -> y equator
        # 3 -> x equator
        # 4 -> z top view, ie. both x and y equator
        # 5 -> y top view, ie. both x and z equator
        # 6 -> x top view, ie. both y and z equator
        eq_class = np.zeros(n_rots)
        top_view_mask = n_eqs_close > 1
