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
            These are generated using the Saaf - Kuijlaars algoithm. Default value is 1200.
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
        
