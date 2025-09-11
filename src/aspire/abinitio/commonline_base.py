import logging
import math

import numpy as np
import scipy.sparse as sparse

from aspire.image import Image
from aspire.operators import PolarFT
from aspire.utils import Rotation, fuzzy_mask
from aspire.utils.random import choice

from .commonline_utils import _generate_shift_phase_and_filter

logger = logging.getLogger(__name__)


class CLOrient3D:
    """
    Define a base class for estimating 3D orientations using common lines methods
    """

    def __init__(
        self,
        src,
        n_rad=None,
        n_theta=360,
        n_check=None,
        hist_bin_width=3,
        full_width=6,
        max_shift=0.15,
        shift_step=1,
        offsets_max_shift=None,
        offsets_shift_step=None,
        offsets_max_memory=10000,
        offsets_equations_factor=1,
        mask=True,
    ):
        """
        Initialize an object for estimating 3D orientations using common lines.

        :param src: The source object of 2D denoised or class-averaged images.
        :param n_rad: The number of points in the radial direction. If None,
            n_rad will default to the ceiling of half the resolution of the source.
        :param n_theta: The number of points in the theta direction. This value must be even.
            Default is 360.
        :param n_check: For each image/projection find its common-lines with
            n_check images. If n_check is less than the total number of images,
            a random subset of n_check images is used.
        :param max_shift: Determines maximum range for shifts for
            common-line detection as a proportion of the
            resolution. Default is 0.15.
        :param shift_step: Resolution of shift estimation for
            common-line detection in pixels.  Default is 1 pixel.
        :param offsets_max_shift: Determines maximum range for shifts
            for 2D offset estimation as a proportion of the
            resolution. Default `None` inherits from `max_shift`.
        :param offsets_shift_step: Resolution of shift estimation for
            2D offset estimation in pixels.  Default `None` inherits
            from `shift_step`.
        :param offsets_equations_factor: The factor to rescale the
            number of shift equations (=1 in default)
        :param offsets_max_memory: If there are N images and N_check
            selected to check for common lines, then the exact system
            of equations solved for the shifts is of size 2N x
            N(N_check-1)/2 (2N unknowns and N(N_check-1)/2 equations).
            This may be too big if N is large. The algorithm will use
            `equations_factor` times the total number of equations if
            the resulting total number of memory requirements is less
            than `offsets_max_memory` (in megabytes); otherwise it will reduce
            the number of equations by approximation to fit in
            `offsets_max_memory`.  For more information see the
            references in `estimate_shifts`.  Defaults to 10GB.
        :param hist_bin_width: Bin width in smoothing histogram (degrees).
        :param full_width: Selection width around smoothed histogram peak (degrees).
            `adaptive` will attempt to automatically find the smallest number of
            `hist_bin_width`s required to find at least one valid image index.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """
        self.src = src
        # Note dtype is inferred from self.src
        self.dtype = self.src.dtype
        self.n_img = src.n
        self.n_res = self.src.L
        self.n_rad = n_rad
        self.n_theta = n_theta
        self.n_check = n_check
        self.hist_bin_width = hist_bin_width
        if str(full_width).lower() == "adaptive":
            full_width = -1
        self.full_width = int(full_width)
        self.max_shift = math.ceil(max_shift * self.n_res)
        self.shift_step = shift_step
        self.offsets_max_shift = self.max_shift
        if offsets_max_shift is not None:
            self.offsets_max_shift = math.ceil(offsets_max_shift * self.n_res)
        self.offsets_shift_step = offsets_shift_step or self.shift_step
        self.offsets_equations_factor = offsets_equations_factor
        self.offsets_max_memory = int(offsets_max_memory)
        self.mask = mask
        self._pf = None

        # Outputs
        self.rotations = None
        self.shifts = None

        self._build()

    def _build(self):
        """
        Build the internal data structure for orientation estimation
        """
        if self.n_rad is None:
            self.n_rad = math.ceil(0.5 * self.n_res)
        if self.n_check is None:
            self.n_check = self.n_img
        if not (0 < self.n_check <= self.n_img):
            msg = "n_check must be in (0, n_img]"
            logger.error(msg)
            raise NotImplementedError(msg)
        if self.n_theta % 2 == 1:
            msg = "n_theta must be even"
            logger.error(msg)
            raise NotImplementedError(msg)

    @property
    def pf(self):
        if self._pf is None:
            self._prepare_pf()
        return self._pf

    def _prepare_pf(self):
        """
        Prepare the polar Fourier transform used for correlations.
        """
        imgs = self.src.images[:]

        if self.mask:
            # For best results and to reproduce MATLAB:
            #   Set risetime=2
            #   Always compute mask (erf) in doubles.
            fuzz_mask = fuzzy_mask((self.n_res, self.n_res), np.float64, risetime=2)
            #   Apply mask in doubles (allow imgs to upcast as needed)
            imgs = imgs * fuzz_mask
            #   Cast to desired type
            imgs = Image(imgs.asnumpy().astype(self.dtype, copy=False))

        # Obtain coefficients of polar Fourier transform for input 2D images
        pft = PolarFT(
            (self.n_res, self.n_res), self.n_rad, self.n_theta, dtype=self.dtype
        )
        pf = pft.transform(imgs)

        # We remove the DC the component. pf has size (n_img) x (n_theta/2) x (n_rad-1),
        # with pf[:, :, 0] containing low frequency content and pf[:, :, -1] containing
        # high frequency content.
        self._pf = pf[:, :, 1:]

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images

        Subclasses should implement this function.
        """
        raise NotImplementedError("subclasses should implement this")

    @property
    def rotations(self):
        """
        Returns estimated rotations.

        Computes if `rotations` is None.

        :return: Estimated rotations
        """
        if self._rotations is None:
            self.estimate_rotations()
        else:
            logger.info("Using existing estimated `rotations`.")
        return self._rotations

    @rotations.setter
    def rotations(self, value):
        self._rotations = value

    @property
    def shifts(self):
        """
        Returns estimated shifts.

        Computes if `shifts` is None.

        :return: Estimated shifts
        """
        if self._shifts is None:
            self.estimate_shifts()
        else:
            logger.info("Using existing estimated `shifts`.")
        return self._shifts

    @shifts.setter
    def shifts(self, value):
        self._shifts = value

    def estimate_shifts(self, equations_factor=1, max_memory=4000):
        """
        Estimate 2D shifts in images

        This function computes 2D shifts in x, y of images by solving the least-squares
        equations to `Ax = b`. `A` on the left-hand side is a sparse matrix representing
        precomputed coefficients of shift equations; and on the right-side, `b` is
        estimated 1D shifts along the theta direction between two Fourier rays (one in
        image i and the other in image j).  Each row of shift equations contains four
        unknowns, shifts in x, y for a pair of images. The detailed implementation
        can be found in the book chapter as below:
        Y. Shkolnisky and A. Singer,
        Center of Mass Operators for Cryo-EM - Theory and Implementation,
        Modeling Nanoscale Imaging in Electron Microscopy,
        T. Vogt, W. Dahmen, and P. Binev (Eds.)
        Nanostructure Science and Technology Series,
        Springer, 2012, pp. 147–177
        """

        # Generate approximated shift equations from estimated rotations
        shift_equations, shift_b = self._get_shift_equations_approx()

        # Solve the linear equation, optionally printing numerical debug details.
        show = False
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            show = True

        # Estimate shifts.
        est_shifts = sparse.linalg.lsqr(
            shift_equations, shift_b, atol=1e-8, btol=1e-8, iter_lim=100, show=show
        )[0]
        est_shifts = est_shifts.reshape((self.n_img, 2))
        # Convert (XY) axes and negate estimated shift orientations
        self.shifts = -est_shifts[:, ::-1]

        return self.shifts

    def estimate(self, **kwargs):
        """
        Estimate orientation and shifts for all 2D images.

        :return: (rotations, shifts)
        """

        self.estimate_rotations(**kwargs)
        self.estimate_shifts(**kwargs)

        return self.rotations, self.shifts

    def _get_shift_equations_approx(self):
        """
        Generate approximated shift equations from estimated rotations

        The function computes the common lines from the estimated rotations,
        and then, for each common line, estimates the 1D shift between its two
        Fourier rays (one in image i and the other in image j). Using the common
        lines and the 1D shifts, shift equations are generated randomly based
        on a memory factor and represented by sparse matrix.

        This function processes the (Fourier transformed) images exactly as the
        `build_clmatrix` function.

        :return; The left and right-hand side of shift equations
        """

        n_theta_half = self.n_theta // 2
        n_img = self.n_img

        # `estimate_shifts()` requires that rotations have already been estimated.
        rotations = Rotation(self.rotations)

        pf = self.pf.copy()

        # Estimate number of equations that will be used to calculate the shifts
        n_equations = self._estimate_num_shift_equations(n_img)

        # Allocate local variables for estimating 2D shifts based on the estimated number
        # of equations. The shift equations are represented using a sparse matrix,
        # since each row in the system contains four non-zeros (as it involves
        # exactly four unknowns). The variables below are used to construct
        # this sparse system. The k'th non-zero element of the equations matrix
        # is stored at index (shift_i(k),shift_j(k)).
        shift_i = np.zeros((n_equations, 4), dtype=self.dtype)
        shift_j = np.zeros((n_equations, 4), dtype=self.dtype)
        shift_eq = np.zeros((n_equations, 4), dtype=self.dtype)
        shift_b = np.zeros(n_equations, dtype=self.dtype)

        # Prepare the shift phases to try and generate filter for common-line detection
        # The shift phases are pre-defined in a range of max_shift that can be
        # applied to maximize the common line calculation. The common-line filter
        # is also applied to the radial direction for easier detection.
        r_max = pf.shape[2]
        _, shift_phases, h = _generate_shift_phase_and_filter(
            r_max, self.offsets_max_shift, self.offsets_shift_step, self.dtype
        )

        d_theta = np.pi / n_theta_half

        # Generate two index lists for [i, j] pairs of images
        idx_i, idx_j = self._generate_index_pairs(n_equations)

        # Go through all shift equations in the size of n_equations
        # Iterate over the common lines pairs and for each pair find the 1D
        # relative shift between the two Fourier lines in the pair.
        for shift_eq_idx in range(n_equations):
            i = idx_i[shift_eq_idx]
            j = idx_j[shift_eq_idx]
            # get the common line indices based on the rotations from i and j images
            c_ij, c_ji = self._get_cl_indices(rotations, i, j, n_theta_half)

            # Extract the Fourier rays that correspond to the common line
            pf_i = pf[i, c_ij]

            # Check whether need to flip or not Fourier ray of j image
            # Is the common line in image j in the positive
            # direction of the ray (is_pf_j_flipped=False) or in the
            # negative direction (is_pf_j_flipped=True).
            is_pf_j_flipped = c_ji >= n_theta_half
            if not is_pf_j_flipped:
                pf_j = pf[j, c_ji]
            else:
                pf_j = pf[j, c_ji - n_theta_half]

            # Use ray from opposite side of origin.
            # Correpsonds to `freqs` convention in PFT,
            #   where the legacy code used a negated frequency grid.
            pf_i, pf_j = np.conj(pf_i), np.conj(pf_j)

            # perform bandpass filter, normalize each ray of each image,
            pf_i = self._apply_filter_and_norm("i, i -> i", pf_i, r_max, h)
            pf_j = self._apply_filter_and_norm("i, i -> i", pf_j, r_max, h)

            # apply the shifts to images
            pf_i_flipped = np.conj(pf_i)
            pf_i_stack = pf_i[:, None] * shift_phases.T
            pf_i_flipped_stack = pf_i_flipped[:, None] * shift_phases.T

            c1 = 2 * np.dot(pf_i_stack.T.conj(), pf_j).real
            c2 = 2 * np.dot(pf_i_flipped_stack.T.conj(), pf_j).real

            # find the indices for the maximum values
            # and apply corresponding shifts
            sidx1 = np.argmax(c1)
            sidx2 = np.argmax(c2)
            sidx = sidx1 if c1[sidx1] > c2[sidx2] else sidx2
            dx = -self.offsets_max_shift + sidx * self.offsets_shift_step

            # angle of common ray in image i
            shift_alpha = c_ij * d_theta
            # Angle of common ray in image j.
            shift_beta = c_ji * d_theta
            # Row index to construct the sparse equations
            shift_i[shift_eq_idx] = shift_eq_idx
            # Columns of the shift variables that correspond to the current pair [i, j]
            shift_j[shift_eq_idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            # Right hand side of the current equation
            shift_b[shift_eq_idx] = dx

            # Compute the coefficients of the current equation
            if not is_pf_j_flipped:
                shift_eq[shift_eq_idx] = np.array(
                    [
                        np.sin(shift_alpha),
                        np.cos(shift_alpha),
                        -np.sin(shift_beta),
                        -np.cos(shift_beta),
                    ]
                )
            else:
                shift_beta = shift_beta - np.pi
                shift_eq[shift_eq_idx] = np.array(
                    [
                        -np.sin(shift_alpha),
                        -np.cos(shift_alpha),
                        -np.sin(shift_beta),
                        -np.cos(shift_beta),
                    ]
                )

        # create sparse matrix object only containing non-zero elements
        shift_equations = sparse.csr_matrix(
            (shift_eq.flatten(), (shift_i.flatten(), shift_j.flatten())),
            shape=(n_equations, 2 * n_img),
            dtype=self.dtype,
        )

        return shift_equations, shift_b

    def _estimate_num_shift_equations(self, n_img):
        """
        Estimate total number of shift equations in images

        The function computes total number of shift equations based on
        number of images and preselected memory factor.

        :param n_img:  The total number of input images
        :return: Estimated number of shift equations
        """
        # Number of equations that will be used to estimation the shifts
        n_equations_total = int(np.ceil(n_img * (self.n_check - 1) / 2))

        # Estimated memory requirements for the full system of equation.
        # This ignores the sparsity of the system, since backslash seems to
        # ignore it.
        memory_total = self.offsets_equations_factor * (
            n_equations_total * 2 * n_img * self.dtype.itemsize
        )

        if memory_total < (self.offsets_max_memory * 10**6):
            n_equations = int(
                np.ceil(self.offsets_equations_factor * n_equations_total)
            )
        else:
            subsampling_factor = (self.offsets_max_memory * 10**6) / memory_total
            subsampling_factor = min(1.0, subsampling_factor)
            n_equations = int(np.ceil(n_equations_total * subsampling_factor))

        if n_equations < n_img:
            logger.warning(
                "Too few equations. Increase memory_factor. Setting n_equations to n_img."
            )
            n_equations = n_img

        if n_equations < 2 * n_img:
            logger.warning(
                "Number of equations is small. Consider increase memory_factor."
            )

        return n_equations

    def _generate_index_pairs(self, n_equations):
        """
        Generate two index lists for [i, j] pairs of images
        """

        # Generate the i,j tuples of indices representing the upper triangle above the diagonal.
        idx_i, idx_j = np.triu_indices(self.n_img, k=1)

        # Select random pairs based on the size of n_equations
        rp = choice(np.arange(len(idx_j)), size=n_equations, replace=False)

        return idx_i[rp], idx_j[rp]

    def _get_cl_indices(self, rotations, i, j, n_theta):
        """
        Get common line indices based on the rotations from i and j images

        :param rotations: Rotation object
        :param i: Index for i image
        :param j: Index for j image
        :param n_theta: Total number of common lines
        :return: Common line indices for i and j images
        """
        # get the common line indices based on the rotations from i and j images
        c_ij, c_ji = rotations.invert().common_lines(i, j, 2 * n_theta)

        # To match clmatrix, c_ij is always less than PI
        # and c_ji may be be larger than PI.
        if c_ij >= n_theta:
            c_ij -= n_theta
            c_ji -= n_theta
        if c_ji < 0:
            c_ji += 2 * n_theta

        return c_ij, c_ji

    def _apply_filter_and_norm(self, subscripts, pf, r_max, h):
        """
        Apply common line filter and normalize each ray

        :subscripts: Specifies the subscripts for summation of Numpy
            `einsum` function

        :param pf: Fourier transform of images
        :param r_max: Maximum index for common line detection
        :param h: common lines filter
        :return: filtered and normalized i images
        """

        # Note if we'd rather not have the dtype and casting args,
        #   we can control h.dtype instead.
        pf = np.einsum(subscripts, pf, h, dtype=pf.dtype)

        # This is a high pass filter, cutting out the lowest frequency
        # (DC has already been removed).
        pf[..., 0] = 0
        pf /= np.linalg.norm(pf, axis=-1)[..., np.newaxis]

        return pf
