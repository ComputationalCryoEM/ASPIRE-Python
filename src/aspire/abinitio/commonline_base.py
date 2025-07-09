import logging
import math
import os

import numpy as np
import scipy.sparse as sparse

from aspire.image import Image
from aspire.operators import PolarFT
from aspire.utils import Rotation, complex_type, fuzzy_mask, tqdm
from aspire.utils.random import choice

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
        self.mask = mask
        self._pf = None

        # Sanity limit to match potential clmatrix dtype of int16.
        if self.n_img > (2**15 - 1):
            raise NotImplementedError(
                "Commonlines implementation limited to <2**15 images."
            )

        # Auto configure GPU
        self.__gpu_module = None
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() >= 1:
                gpu_id = cp.cuda.runtime.getDevice()
                logger.info(
                    f"cupy and GPU {gpu_id} found by cuda runtime; enabling cupy."
                )
                self.__gpu_module = self.__init_cupy_module()
            else:
                logger.info("GPU not found, defaulting to numpy.")

        except ModuleNotFoundError:
            logger.info("cupy not found, defaulting to numpy.")

        # Outputs
        self.clmatrix = None
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
    def clmatrix(self):
        """
        Returns Common Lines Matrix.

        Computes if `clmatrix` is None.

        :return: Common Lines Matrix
        """
        if self._clmatrix is None:
            self.build_clmatrix()
        else:
            logger.info("Using existing estimated `clmatrix`.")
        return self._clmatrix

    @clmatrix.setter
    def clmatrix(self, value):
        self._clmatrix = value

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

    def build_clmatrix(self):
        """
        Build common-lines matrix from Fourier stack of 2D images

        Wrapper for cpu/gpu dispatch.
        """

        logger.info("Begin building Common Lines Matrix")

        # host/gpu dispatch
        if self.__gpu_module:
            res = self.build_clmatrix_cu()
        else:
            res = self.build_clmatrix_host()

        # Unpack result
        self._shifts_1d, self.clmatrix = res

        return self.clmatrix

    def build_clmatrix_host(self):
        """
        Build common-lines matrix from Fourier stack of 2D images
        """

        n_img = self.n_img
        n_check = self.n_check

        if self.n_theta % 2 == 1:
            msg = "n_theta must be even"
            logger.error(msg)
            raise NotImplementedError(msg)

        n_theta_half = self.n_theta // 2

        # need to do a copy to prevent modifying self.pf for other functions
        pf = self.pf.copy()

        # Allocate local variables for return
        # clmatrix represents the common lines matrix.
        # Namely, clmatrix[i,j] contains the index in image i of
        # the common line with image j. Note the common line index
        # starts from 0 instead of 1 as Matlab version. -1 means
        # there is no common line such as clmatrix[i,i].
        clmatrix = -np.ones((n_img, n_img), dtype=self.dtype)
        # When cl_dist[i, j] is not -1, it stores the maximum value
        # of correlation between image i and j for all possible 1D shifts.
        # We will use cl_dist[i, j] = -1 (including j<=i) to
        # represent that there is no need to check common line
        # between i and j. Since it is symmetric,
        # only above the diagonal entries are necessary.
        cl_dist = -np.ones((n_img, n_img), dtype=self.dtype)

        # Allocate variables used for shift estimation

        # set maximum value of 1D shift (in pixels) to search
        # between common-lines.
        max_shift = self.max_shift
        # Set resolution of shift estimation in pixels. Note that
        # shift_step can be any positive real number.
        shift_step = self.shift_step
        # 1D shift between common-lines
        shifts_1d = np.zeros((n_img, n_img))

        # Prepare the shift phases to try and generate filter for common-line detection
        r_max = pf.shape[2]
        shifts, shift_phases, h = self._generate_shift_phase_and_filter(
            r_max, max_shift, shift_step
        )

        # Apply bandpass filter, normalize each ray of each image
        # Note that only use half of each ray
        pf = self._apply_filter_and_norm("ijk, k -> ijk", pf, r_max, h)

        # Setup a progress bar
        _total_pairs_to_test = self.n_img * (self.n_check - 1) // 2
        pbar = tqdm(desc="Searching over common line pairs", total=_total_pairs_to_test)

        # Search for common lines between [i, j] pairs of images.
        # Creating pf and building common lines are different to the Matlab version.
        # The random selection is implemented.
        for i in range(n_img - 1):
            p1 = pf[i]
            p1_real = np.real(p1)
            p1_imag = np.imag(p1)

            # build the subset of j images if n_check < n_img
            n_remaining = n_img - i - 1
            n_j = min(n_remaining, n_check)
            subset_j = np.sort(choice(n_remaining, n_j, replace=False) + i + 1)

            for j in subset_j:
                p2_flipped = np.conj(pf[j])

                for shift in range(len(shifts)):
                    shift_phase = shift_phases[shift]
                    p2_shifted_flipped = (shift_phase * p2_flipped).T
                    # Compute correlations in the positive r direction
                    part1 = p1_real.dot(np.real(p2_shifted_flipped))
                    # Compute correlations in the negative r direction
                    part2 = p1_imag.dot(np.imag(p2_shifted_flipped))

                    c1 = part1 - part2
                    sidx = c1.argmax()
                    cl1, cl2 = np.unravel_index(sidx, c1.shape)
                    sval = c1[cl1, cl2]

                    c2 = part1 + part2
                    sidx = c2.argmax()
                    cl1_2, cl2_2 = np.unravel_index(sidx, c2.shape)
                    sval2 = c2[cl1_2, cl2_2]

                    if sval2 > sval:
                        cl1 = cl1_2
                        cl2 = cl2_2 + n_theta_half
                        sval = sval2
                    sval = 2 * sval
                    if sval > cl_dist[i, j]:
                        clmatrix[i, j] = cl1
                        clmatrix[j, i] = cl2
                        cl_dist[i, j] = sval
                        shifts_1d[i, j] = shifts[shift]
                pbar.update()
        pbar.close()

        return shifts_1d, clmatrix

    def build_clmatrix_cu(self):
        """
        Build common-lines matrix from Fourier stack of 2D images
        """

        import cupy as cp

        n_img = self.n_img
        r = self.pf.shape[2]

        if self.n_theta % 2 == 1:
            msg = "n_theta must be even"
            logger.error(msg)
            raise NotImplementedError(msg)

        # Copy to prevent modifying self.pf for other functions
        # Simultaneously place on GPU
        pf = cp.array(self.pf)

        # Allocate local variables for return
        # clmatrix represents the common lines matrix.
        # Namely, clmatrix[i,j] contains the index in image i of
        # the common line with image j. Note the common line index
        # starts from 0 instead of 1 as Matlab version. -1 means
        # there is no common line such as clmatrix[i,i].
        clmatrix = -cp.ones((n_img, n_img), dtype=np.int16)

        # Allocate variables used for shift estimation
        #
        # Set maximum value of 1D shift (in pixels) to search
        # between common-lines.
        # Set resolution of shift estimation in pixels. Note that
        # shift_step can be any positive real number.
        #
        # Prepare the shift phases to try and generate filter for common-line detection
        #
        # Note the CUDA implementation has been optimized to not
        # compute or return diagnostic 1d shifts.
        _, shift_phases, h = self._generate_shift_phase_and_filter(
            r, self.max_shift, self.shift_step
        )
        # Transfer to device, dtypes must match kernel header.
        shift_phases = cp.asarray(shift_phases, dtype=complex_type(self.dtype))

        # Apply bandpass filter, normalize each ray of each image
        # Note that this only uses half of each ray
        pf = self._apply_filter_and_norm("ijk, k -> ijk", pf, r, h)

        # Tranpose `pf` for better (CUDA) memory access pattern, and cast as needed.
        pf = cp.ascontiguousarray(pf.T, dtype=complex_type(self.dtype))

        # Get kernel
        if self.dtype == np.float64:
            build_clmatrix_kernel = self.__gpu_module.get_function(
                "build_clmatrix_kernel"
            )
        elif self.dtype == np.float32:
            build_clmatrix_kernel = self.__gpu_module.get_function(
                "fbuild_clmatrix_kernel"
            )
        else:
            raise NotImplementedError(
                "build_clmatrix_kernel only implemented for float32 and float64."
            )

        # Configure grid of blocks
        blkszx = 32
        # Enough blocks to cover n_img-1
        nblkx = (self.n_img + blkszx - 2) // blkszx
        blkszy = 32
        # Enough blocks to cover n_img
        nblky = (self.n_img + blkszy - 1) // blkszy

        # Launch
        logger.info("Launching `build_clmatrix_kernel`.")
        build_clmatrix_kernel(
            (nblkx, nblky),
            (blkszx, blkszy),
            (
                n_img,
                pf.shape[1],
                r,
                pf,
                clmatrix,
                len(shift_phases),
                shift_phases,
            ),
        )

        # Copy result device arrays to host
        clmatrix = clmatrix.get().astype(self.dtype, copy=False)

        # Note diagnostic 1d shifts are not computed in the CUDA implementation.
        return None, clmatrix

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

        :param equations_factor: The factor to rescale the number of shift equations
            (=1 in default)
        :param max_memory: If there are N images and N_check selected to check
            for common lines, then the exact system of equations solved for the shifts
            is of size 2N x N(N_check-1)/2 (2N unknowns and N(N_check-1)/2 equations).
            This may be too big if N is large. The algorithm will use `equations_factor`
            times the total number of equations if the resulting total number of memory
            requirements is less than `max_memory` (in megabytes); otherwise it will
            reduce the number of equations by approximation to fit in `max_memory`.
        """

        # Generate approximated shift equations from estimated rotations
        shift_equations, shift_b = self._get_shift_equations_approx(
            equations_factor, max_memory
        )

        # Solve the linear equation, optionally printing numerical debug details.
        show = False
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            show = True

        # Estimate shifts.
        est_shifts = sparse.linalg.lsqr(shift_equations, shift_b, show=show)[0]
        self.shifts = est_shifts.reshape((self.n_img, 2))

        return self.shifts

    def estimate(self, **kwargs):
        """
        Estimate orientation and shifts for all 2D images.

        :return: (rotations, shifts)
        """

        self.estimate_rotations(**kwargs)
        self.estimate_shifts(**kwargs)

        return self.rotations, self.shifts

    def _get_shift_equations_approx(self, equations_factor=1, max_memory=4000):
        """
        Generate approximated shift equations from estimated rotations

        The function computes the common lines from the estimated rotations,
        and then, for each common line, estimates the 1D shift between its two
        Fourier rays (one in image i and the other in image j). Using the common
        lines and the 1D shifts, shift equations are generated randomly based
        on a memory factor and represented by sparse matrix.

        This function processes the (Fourier transformed) images exactly as the
        `build_clmatrix` function.

        :param equations_factor: The factor to rescale the number of shift equations
            (=1 in default)
        :param max_memory: If there are N images and N_check selected to check
            for common lines, then the exact system of equations solved for the shifts
            is of size 2N x N(N_check-1)/2 (2N unknowns and N(N_check-1)/2 equations).
            This may be too big if N is large. The algorithm will use `equations_factor`
            times the total number of equations if the resulting total number of
            memory requirements is less than `max_memory` (in megabytes); otherwise it
            will reduce the number of equations to fit in `max_memory`.

        :return; The left and right-hand side of shift equations
        """

        n_theta_half = self.n_theta // 2
        n_img = self.n_img

        # `estimate_shifts()` requires that rotations have already been estimated.
        rotations = Rotation(self.rotations)

        pf = self.pf.copy()

        # Estimate number of equations that will be used to calculate the shifts
        n_equations = self._estimate_num_shift_equations(
            n_img, equations_factor, max_memory
        )

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
        _, shift_phases, h = self._generate_shift_phase_and_filter(
            r_max, self.offsets_max_shift, self.offsets_shift_step
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

            # perform bandpass filter, normalize each ray of each image,
            pf_i = self._apply_filter_and_norm("i, i -> i", pf_i, r_max, h)
            pf_j = self._apply_filter_and_norm("i, i -> i", pf_j, r_max, h)

            # apply the shifts to images
            pf_i_flipped = np.conj(pf_i)
            pf_i_stack = np.einsum("i, ji -> ij", pf_i, shift_phases)
            pf_i_flipped_stack = np.einsum("i, ji -> ij", pf_i_flipped, shift_phases)

            c1 = 2 * np.real(np.dot(np.conj(pf_i_stack.T), pf_j))
            c2 = 2 * np.real(np.dot(np.conj(pf_i_flipped_stack.T), pf_j))

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
            coefs = np.array(
                [
                    np.cos(shift_alpha),
                    np.sin(shift_alpha),
                    -np.cos(shift_beta),
                    -np.sin(shift_beta),
                ]
            )
            shift_eq[shift_eq_idx] = (
                [-1, -1, 0, 0] * coefs if is_pf_j_flipped else coefs
            )

        # create sparse matrix object only containing non-zero elements
        shift_equations = sparse.csr_matrix(
            (shift_eq.flatten(), (shift_i.flatten(), shift_j.flatten())),
            shape=(n_equations, 2 * n_img),
            dtype=self.dtype,
        )

        return shift_equations, shift_b

    def _estimate_num_shift_equations(self, n_img, equations_factor=1, max_memory=4000):
        """
        Estimate total number of shift equations in images

        The function computes total number of shift equations based on
        number of images and preselected memory factor.

        :param n_img:  The total number of input images
        :param equations_factor: The factor to rescale the number of shift equations
            (=1 in default)
        :param max_memory: If there are N images and N_check selected to check
            for common lines, then the exact system of equations solved for the shifts
            is of size 2N x N(N_check-1)/2 (2N unknowns and N(N_check-1)/2 equations).
            This may be too big if N is large. The algorithm will use `equations_factor`
            times the total number of equations if the resulting total number of
            memory requirements is less than `max_memory` (in megabytes); otherwise it
            will reduce the number of equations to fit in `max_memory`.
        :return: Estimated number of shift equations
        """
        # Number of equations that will be used to estimation the shifts
        n_equations_total = int(np.ceil(n_img * (self.n_check - 1) / 2))
        # Estimated memory requirements for the full system of equation.
        # This ignores the sparsity of the system, since backslash seems to
        # ignore it.
        memory_total = equations_factor * (
            n_equations_total * 2 * n_img * self.dtype.itemsize
        )
        if memory_total < (max_memory * 10**6):
            n_equations = int(np.ceil(equations_factor * n_equations_total))
        else:
            subsampling_factor = (max_memory * 10**6) / memory_total
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

    def _generate_shift_phase_and_filter(self, r_max, max_shift, shift_step):
        """
        Prepare the shift phases and generate filter for common-line detection

        The shift phases are pre-defined in a range of max_shift that can be
        applied to maximize the common line calculation. The common-line filter
        is also applied to the radial direction for easier detection.

        :param r_max: Maximum index for common line detection
        :param max_shift: Maximum value of 1D shift (in pixels) to search
        :param shift_step: Resolution of shift estimation in pixels
        :return: shift phases matrix and common lines filter
        """

        # Number of shifts to try
        n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))

        # only half of ray, excluding the DC component.
        rk = np.arange(1, r_max + 1, dtype=self.dtype)

        # Generate all shift phases
        shifts = -max_shift + shift_step * np.arange(n_shifts, dtype=self.dtype)
        shift_phases = np.exp(np.outer(shifts, -2 * np.pi * 1j * rk / (2 * r_max + 1)))
        # Set filter for common-line detection
        h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * (r_max / 4) ** 2))

        return shifts, shift_phases, h

    def _generate_index_pairs(self, n_equations):
        """
        Generate two index lists for [i, j] pairs of images
        """
        idx_i = []
        idx_j = []
        for i in range(self.n_img - 1):
            tmp_j = range(i + 1, self.n_img)
            idx_i.extend([i] * len(tmp_j))
            idx_j.extend(tmp_j)
        idx_i = np.array(idx_i, dtype="int")
        idx_j = np.array(idx_j, dtype="int")

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

    @staticmethod
    def __init_cupy_module():
        """
        Private utility method to read in CUDA source and return as
        compiled CuPy module.
        """

        import cupy as cp

        # Read in contents of file
        fp = os.path.join(os.path.dirname(__file__), "commonline_base.cu")
        with open(fp, "r") as fh:
            module_code = fh.read()

        # CuPy compile the CUDA code
        # Note these optimizations are to steer aggresive optimization
        # for single precision code.  Fast math will potentionally
        # reduce accuracy in single precision.
        return cp.RawModule(
            code=module_code,
            backend="nvcc",
            options=("-O3", "--use_fast_math", "--extra-device-vectorization"),
        )
