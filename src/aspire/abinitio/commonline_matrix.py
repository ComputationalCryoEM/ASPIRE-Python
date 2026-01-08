import logging
import os

import numpy as np

from aspire.abinitio import CLOrient3D
from aspire.utils import complex_type, tqdm
from aspire.utils.random import choice

from .commonline_utils import _generate_shift_phase_and_filter

logger = logging.getLogger(__name__)


class CLMatrix(CLOrient3D):
    """
    An intermediate base class to serve commonline algorithms that use
    a commonline matrix.
    """

    def __init__(self, src, disable_gpu=False, **kwargs):
        """
        Initialize an object for estimating 3D orientations with a
        commonline algorithm that uses a constructed commonlines matrix.
        """
        super().__init__(src, **kwargs)

        # Sanity limit to match potential clmatrix dtype of int16.
        if self.n_img > (2**15 - 1):
            raise NotImplementedError(
                "Commonlines implementation limited to <2**15 images."
            )

        # Auto configure GPU
        self.__gpu_module = None
        if not disable_gpu:
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

        return res

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
        shifts, shift_phases, h = _generate_shift_phase_and_filter(
            r_max, max_shift, shift_step, self.dtype
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
        _, shift_phases, h = _generate_shift_phase_and_filter(
            r, self.max_shift, self.shift_step, self.dtype
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

    @staticmethod
    def __init_cupy_module():
        """
        Private utility method to read in CUDA source and return as
        compiled CuPy module.
        """

        import cupy as cp

        # Read in contents of file
        fp = os.path.join(os.path.dirname(__file__), "commonline_matrix.cu")
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
