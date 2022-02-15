import logging

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CLSymmetryC3C4(CLOrient3D):
    """
    Define a class to estimate 3D orientations using common lines methods for molecules with
    C3 and C4 cyclical symmetry.

    The related publications are listed below:
    G. Pragier and Y. Shkolnisky,
    A Common Lines Approach for Abinitio Modeling of Cyclically Symmetric Molecules,
    Inverse Problems, 35, 124005, (2019).

    Y. Shkolnisky, and A. Singer,
    Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

    A. Singer, R. R. Coifman, F. J. Sigworth, D. W. Chester, Y. Shkolnisky,
    Detecting Consistent Common Lines in Cryo-EM by Voting,
    Journal of Structural Biology, 169, 312-322 (2010).
    """

    def __init__(self, src, n_symm=None, n_rad=None, n_theta=None):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_symm: The symmetry order of the molecule. 3 or 4.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """

        super().__init__(src, n_rad=n_rad, n_theta=n_theta)

        self.n_symm = n_symm

    def orientation_estimation(self):
        """
        Estimate rotation matrices for symmetric molecules.
        """
        pass

    ###########################################
    # Primary Methods                         #
    ###########################################

    def compute_third_row_outer_prod_c34(self, max_shift_1d):
        """
        Compute the outer products of the third rows of the rotation matrices Rij and Rii.

        A pre-computed common line matrix is required as input.
        """

        pf = self.pf
        n_symm = self.n_symm
        max_shift = self.max_shift
        shift_step = self.shift_step
        n_theta = self.n_theta

        # Step 1: Detect a single pair of common-lines between each pair of images

        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        if n_symm == 3:
            is_handle_equator_ims = False
        else:
            is_handle_equator_ims = True

        sclmatrix = self.self_clmatrix_c3_c4(
            pf, n_symm, max_shift, shift_step, is_handle_equator_ims
        )

        # Step 3: Calculate self-relative-rotations
        Riis = self.estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta)

        # Step 4: Calculate relative rotations
        Rijs = self.estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta)

        # Step 5: Inner J-synchronization
        vijs, viis = self.local_sync_J_c3_c4(n_symm, Rijs, Riis)

        return vijs, viis

    def global_sync_J(self, vijs, viis):
        # return vijs, viis
        pass

    def estimate_third_rows(self, vijs, viis, n_symm):
        # return vis
        pass

    def estimate_inplane_rotations(
        self, pf, vis, inplane_rot_res, max_shift, shift_step
    ):
        # return rots
        pass

    #################################################
    # Secondary methods                             #
    #################################################

    def self_clmatrix_c3_c4(
        self, pf, n_symm, max_shift, shift_step, is_handle_equator_ims
    ):
        # return sclmatrix
        pass

    def estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta):
        # return Riis
        pass

    def estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta):
        # return Rijs
        pass

    def local_sync_J_c3_c4(n_symm, Rijs, Riis):
        # return vijs, viis
        pass
