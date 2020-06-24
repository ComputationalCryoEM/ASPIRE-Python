import logging
import math
import numpy as np
import scipy.special as sp
import scipy.sparse as sps

from aspire import config

from aspire.basis.polar_2d import PolarBasis2D
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


class Orient3D:
    """
    Define a base class for estimating 3D orientations
    """
    def __init__(self, src, n_rad=None, n_theta=None):
        """
        Initialize an object for estimating 3D orientations

        :param src: The source object of 2D denoised or class-averaged imag
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """
        self.src = src
        self.n_img = src.n
        self.n_res = self.src.L
        if n_rad is None:
            self.n_rad = math.ceil(config.orient.r_ratio * self.n_res)
        else:
            self.n_rad = n_rad
        if n_theta is None:
            self.n_theta = config.orient.n_theta
        else:
            self.n_theta = n_theta

        self.max_shift = math.ceil(config.orient.max_shift * self.n_res)
        self.shift_step = config.orient.shift_step

        self.clmatrix = None
        self.cormatrix = None
        self.shifts_1d = None
        self.shift_equations = None
        self.shift_equations_map = None

        self.rotations = None

        self._build()

    def _build(self):
        """
        Build the internal data structure for orientation estimation
        """
        imgs = self.src.images(start=0, num=np.inf).asnumpy()
        masked = True
        if masked:
            mask_radius = self.n_res * 0.45
            self.rise_time = config.orient.rise_time
            self.fuzzy_mask_dims = config.orient.fuzzy_mask_dims
            # mask_radius is of the form xxx.5
            if mask_radius * 2 == int(mask_radius * 2):
                mask_radius = math.ceil(mask_radius)
            # mask is not of the form xxx.5
            else:
                mask_radius = int(round(mask_radius))
            # mask projections
            m = self._fuzzy_mask(self.n_res, mask_radius)
            masked_projs = imgs.copy()
            masked_projs = masked_projs.transpose((2, 0, 1))
            masked_projs *= m
            masked_projs = masked_projs.transpose((1, 2, 0))
        else:
            masked_projs = imgs

        # Obtain coefficients in polar Fourier basis for input 2D images
        self.basis = PolarBasis2D((self.n_res, self.n_res), self.n_rad, self.n_theta)
        self.pf = self.basis.evaluate_t(masked_projs)
        self.pf = m_reshape(self.pf, (self.n_rad, self.n_theta, self.n_img))

        n_theta = self.n_theta
        if n_theta % 2 == 1:
            logger.error('n_theta must be even')
        n_theta = n_theta // 2
        # The first two dimension of pf is of size n_rad x n_theta. Convert pf into an
        # array of size (2xn_r-1) x n_theta/2, that is, take then entire ray
        # through the origin, but take the angles only up PI.
        # This seems redundant: The original images are real, and thus
        # each ray is conjugate symmetric. We therefore gain nothing by taking
        # longer correlations (of length 2*n_r-1 instead of n_r), as the two halves
        # are exactly the same. Taking shorter correlation would speed the
        # computation by a factor of two.
        self.pf = np.concatenate((np.flip(self.pf[1:, n_theta:], 0), self.pf[:, :n_theta]), 0)

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images

        Subclasses should implement this function.
        """
        raise NotImplementedError('subclasses should implement this')

    def output(self):
        """
        Output the 3D orientations in a star file
        """
        pass

    def build_clmatrix(self, n_check=None):
        """
        Build common-lines matrix from Fourier stack of 2D images

        :param n_check: For each image/projection find its common-lines with
            n_check images. If n_check is less than the total number of images,
            a random subset of n_check images is used.
        """

        n_img = self.n_img
        if n_check is None:
            n_check = n_img
        n_theta = self.n_theta
        if n_theta % 2 == 1:
            logger.error('n_theta must be even')
        n_theta = n_theta // 2

        # need to do a copy to prevent modifying self.pf for other functions
        pf = self.pf.copy()

        # Allocate local variables for return
        # clstack represents the common lines matrix:
        # clstack[i, j] and clstack[j ,i] contain the index
        # of the common line of projections i and j. Namely,
        # clstack[i,j] contains the index of the common line
        # in the image i. clstack[j,i] contains the index of
        # the common line in j image.
        clstack = -np.ones((n_img, n_img))
        # corrstack defines the correlation of the common line
        # between image i and j. Since corrstack is symmetric,
        # only above the diagonal entries are necessary.
        # corrstack[i, j] measures how ''common'' is the between
        # image i and j. Small value means high-similarity.
        corrstack = np.zeros((n_img, n_img))

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
        r_max = int((pf.shape[0] - 1) / 2)
        shifts, shift_phases, h = self._generate_shift_phase_and_filter(r_max, max_shift, shift_step)
        all_shift_phases = shift_phases.T

        # Apply bandpass filter and normalize each ray of each image.
        np.einsum('ijk, i -> ijk', pf, h, out=pf)
        pf[r_max - 1:r_max + 2] = 0
        pf /= np.linalg.norm(pf, axis=0)

        # Only half of each ray is needed
        pf = pf[0:r_max]
        pf = pf.transpose((2, 1, 0))

        # Search for common lines between [i, j] pairs of images.
        # Creating pf and building common lines are different to the Matlab version.
        # The random selection is implemented.
        for i in range(n_img-1):

            p1 = pf[i]
            p1_real = np.real(p1)
            p1_imag = np.imag(p1)

            # build the subset of j images if n_check < n_img
            n2 = min(n_img - i, n_check)
            subset_j = np.sort(np.random.choice(n_img - i - 1, n2 - 1,
                                                 replace=False) + i + 1)
            for j in subset_j:

                p2_flipped = np.conj(pf[j])

                for shift in range(len(shifts)):
                    shift_phases = all_shift_phases[shift]
                    p2_shifted_flipped = (shift_phases * p2_flipped).T
                    part1 = p1_real.dot(np.real(p2_shifted_flipped))
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
                        cl2 = cl2_2 + n_theta
                        sval = 2 * sval2
                    else:
                        sval *= 2

                    if sval > corrstack[i, j]:
                        clstack[i, j] = cl1
                        clstack[j, i] = cl2
                        corrstack[i, j] = sval
                        shifts_1d[i, j] = shifts[shift]

        mask = np.where(corrstack != 0)
        corrstack[mask] = 1 - corrstack[mask]

        self.clmatrix = clstack
        self.cormatrix = corrstack
        self.shifts_1d = shifts_1d

    def get_shifts_equations(self, n_check=None):
        """
        Obtain shifts equations from common-lines matrix

        :param n_check: For each image/projection find its common-lines with
            n_check images. If n_check is less than the total number of images,
            a random subset of n_check images is used.
        """

        clstack = self.clmatrix
        corrstack = self.cormatrix
        shifts_1d = self.shifts_1d

        n_img = self.n_img
        if n_check is None:
            n_check = n_img

        n_theta = self.n_theta
        if n_theta % 2 == 1:
            logger.error('n_theta must be even')
        n_theta = n_theta // 2

        # Allocate variables used for shift estimation

        # Based on the estimated common-lines, construct the equations
        # for determining the 2D shift of each image. The shift equations
        # are represented using a sparse matrix, since each row in the
        # system contains four non-zeros (as it involves exactly four unknowns).
        # The variables below are used to construct this sparse system.
        # The k'th non-zero element of the equations matrix is stored as
        # index (shift_i[k],shift_j[k]).

        # Row index for sparse equations system
        shift_i = np.zeros(4 * n_img * n_check)
        #  Column index for sparse equations system
        shift_j = np.zeros(4 * n_img * n_check)
        # The coefficients of the center estimation system ordered
        # as a single vector.
        shift_eq = np.zeros(4 * n_img * n_check)
        # shift_equations_map[k1,k2] is the index of the equation for
        # the common line between images i and j.
        shift_equations_map = np.zeros((n_img, n_img))
        # The equation number we are currently processing
        shift_equation_idx = 0
        # Right hand side of the system
        shift_b = np.zeros(n_img * (n_img - 1) // 2)
        # Not 2*pi/n_theta, since we divided n_theta by 2 to take rays of length 2*n_r-1.
        dtheta = np.pi / n_theta

        # Go through common lines between [i, j  pairs of projections based on the
        # corrstack values created when build the common lines matrix.
        for i in range(n_img-1):

            for j in range(i+1, n_img):
                # skip j image without correlation
                if corrstack[i, j] == 0:
                    continue
                # Create a shift equation for the projections pair (i, j).
                idx = np.arange(4 * shift_equation_idx, 4 * shift_equation_idx + 4)
                shift_alpha = clstack[i, j] * dtheta
                shift_beta = clstack[j, i] * dtheta
                shift_i[idx] = shift_equation_idx
                shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
                shift_b[shift_equation_idx] = shifts_1d[i, j]

                # Compute the coefficients of the current equation.
                if shift_beta < np.pi:
                    shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha),
                                     -np.sin(shift_beta), -np.cos(shift_beta)]
                else:
                    shift_beta -= np.pi
                    shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha),
                                     -np.sin(shift_beta), -np.cos(shift_beta)]

                shift_equations_map[i, j] = shift_equation_idx
                shift_equation_idx += 1

        ell = 4 * shift_equation_idx
        shift_eq[ell: ell + shift_equation_idx] = shift_b
        shift_i[ell: ell + shift_equation_idx] = np.arange(shift_equation_idx)
        shift_j[ell: ell + shift_equation_idx] = 2 * n_img
        mask = np.where(shift_eq != 0)[0]
        shift_eq = shift_eq[mask]
        shift_i = shift_i[mask]
        shift_j = shift_j[mask]
        ell += shift_equation_idx
        shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)),
                                         shape=(shift_equation_idx, 2 * n_img + 1))

        self.shift_equations = shift_equations
        self.shift_equations_map = shift_equations_map

    def estimate_shifts(self, memory_factor=10000):
        """
        Estimate 2D shifts in images using estimated rotations

        The function computes the common lines from the estimated rotations,
        and then, for each common line, estimates the 1D shift between its two
        Fourier rays (one in image i and one in image j). Using the common
        lines and the 1D shifts, the function solves the least-squares
        equations for the 2D shifts.
        This function processes the (Fourier transformed) images exactly as the
        `build_clmatrix` function.

        :param memory_factor: If there are N projections, then the system of
            equations solved for the shifts is of size 2N x N(N-1)/2 (2N
            unknowns and N(N-1)/2 equations). This may be too big if N is
            large. If `memory_factor` between 0 and 1, then it is the
            fraction of equation to retain. That is, the system of
            equations solved will be of size 2N x N*(N-1)/2*`memory_factor`.
            If `memory_factor` is larger than 100, then the number of
            equations is estimated in such a way that the memory used by the equations
            is roughly `memory_factor megabytes`. Default is 10000 (use all equations).
            The code will generate an error if `memory_factor` is between 1 and 100.
        :return: Estimated shifts for all images
        """

        if memory_factor < 0 or (memory_factor > 1 and memory_factor < 100):
            logger.error('Subsampling factor must be between 0 and 1 or larger than 100.')

        n_theta = self.n_theta // 2
        n_img = self.n_img
        rotations = self.rotations

        pf = self.pf.copy()

        # Estimate number of equations that will be used to calculate the shifts
        n_equations = self._estimate_num_shift_equations(n_img, memory_factor)
        # Allocate local variables for estimating 2D shifts based on the estimated number
        # of equations. The shift equations are represented using a sparse matrix,
        # since each row in the system contains four non-zeros (as it involves
        # exactly four unknowns). The variables below are used to construct
        # this sparse system. The k'th non-zero element of the equations matrix
        # is stored at index (shift_i(k),shift_i(k)).
        shift_i = np.zeros(5 * n_equations)
        shift_j = np.zeros(5 * n_equations)
        shift_eq = np.zeros(5 * n_equations)
        shift_b = np.zeros(n_equations)

        # Prepare the shift phases to try and generate filter for common-line detection
        max_shift = self.max_shift
        shift_step = self.shift_step
        r_max = (pf.shape[0] - 1) // 2
        _, shift_phases, h = self._generate_shift_phase_and_filter(r_max, max_shift, shift_step)

        d_theta = np.pi / n_theta

        # Generate two index lists for [i, j] pairs of images
        idx_i = []
        idx_j = []
        for i in range(n_img-1):
            tmp_j = range(i + 1, n_img)
            idx_i.extend([i] * len(tmp_j))
            idx_j.extend(tmp_j)
        idx_i = np.array(idx_i, dtype='int')
        idx_j = np.array(idx_j, dtype='int')
        # Select random pairs based on the size of n_equations
        rp = np.random.choice(np.arange(len(idx_j)), size=n_equations, replace=False)

        # Go through all shift equations in the size of n_equations
        for shift_eq_idx in range(n_equations):
            i = idx_i[rp[shift_eq_idx]]
            j = idx_j[rp[shift_eq_idx]]
            # get the common line indices based on the rotations from i and j images
            c_ij, c_ji = self._get_cl_indices(rotations, i, j, n_theta)

            # check whether need to flip or not
            is_pf_j_flipped = 0
            if c_ji < n_theta:
                pf_j = pf[:, c_ji, j]
            else:
                pf_j = pf[:, c_ji - n_theta, j]
                is_pf_j_flipped = 1
            pf_i = pf[:, c_ij, i]

            # perform bandpass filter, normalize each ray of each image,
            # and only keep half of each ray
            pf_i, pf_j = self._apply_filter_and_norm(pf_i, pf_j, r_max, h)

            # apply the shifts to images
            pf_i_flipped = np.conj(pf_i)
            pf_i_stack = np.einsum('i, ij -> ij', pf_i, shift_phases)
            pf_i_flipped_stack = np.einsum('i, ij -> ij', pf_i_flipped, shift_phases)

            c1 = 2 * np.real(np.dot(np.conj(pf_i_stack.T), pf_j))
            c2 = 2 * np.real(np.dot(np.conj(pf_i_flipped_stack.T), pf_j))

            # find the indices for the maximum values
            # and apply corresponding shifts
            sidx1 = np.argmax(c1)
            sidx2 = np.argmax(c2)

            if c1[sidx1] > c2[sidx2]:
                dx = -max_shift + sidx1 * shift_step
            else:
                dx = -max_shift + sidx2 * shift_step

            idx = np.arange(4 * shift_eq_idx, 4 * shift_eq_idx + 4)
            shift_alpha = c_ij * d_theta
            shift_beta = c_ji * d_theta
            shift_i[idx] = shift_eq_idx
            shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            shift_b[shift_eq_idx] = dx

            # check with compare cl
            if not is_pf_j_flipped:
                shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha),
                                 -np.sin(shift_beta), -np.cos(shift_beta)]
            else:
                shift_beta -= np.pi
                shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha),
                                 -np.sin(shift_beta), -np.cos(shift_beta)]

        t = 4 * n_equations
        shift_eq[t: t + n_equations] = shift_b
        shift_i[t: t + n_equations] = np.arange(n_equations)
        shift_j[t: t + n_equations] = 2 * n_img
        mask = np.where(shift_eq != 0)[0]
        shift_eq = shift_eq[mask]
        shift_i = shift_i[mask]
        shift_j = shift_j[mask]
        shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)),
                                         shape=(n_equations, 2 * n_img + 1))

        est_shifts = sps.linalg.lsqr(shift_equations[:, :-1], shift_b)[0]
        est_shifts = est_shifts.reshape((2, n_img), order='F')

        return est_shifts, shift_equations

    def _estimate_num_shift_equations(self, n_img, memory_factor):
        """
        Estimate total number of shift equations in images

        The function computes total number of shift equations based on
        number of images and preselected memory factor.
        :param n_img:  The total number of input images
        :param memory_factor: If there are N projections, then the system of
            equations solved for the shifts is of size 2N x N(N-1)/2 (2N
            unknowns and N(N-1)/2 equations). This may be too big if N is
            large. If `memory_factor` between 0 and 1, then it is the
            fraction of equation to retain. That is, the system of
            equations solved will be of size 2N x N*(N-1)/2*`memory_factor`.
            If `memory_factor` is larger than 100, then the number of
            equations is estimated in such a way that the memory used by the equations
            is roughly `memory_factor megabytes`. Default is 10000 (use all equations).
            The code will generate an error if `memory_factor` is between 1 and 100.
        :return: Estimated number of shift equations
        """
        # Number of equations that will be used to estimation the shifts
        n_equations_total = int(np.ceil(n_img * (n_img - 1) / 2))
        # Estimated memory requirements for the full system of equation.
        # This ignores the sparsity of the system, since backslash seems to
        # ignore it.
        memory_total = n_equations_total * 2 * n_img * 8

        if memory_factor <= 1:
            # Number of equations that will be used to estimation the shifts
            n_equations = int(np.ceil(n_img * (n_img - 1) * memory_factor / 2))
        else:
            # By how much we need to subsample the system of equations in order to
            # use roughly memoryfactor MB.
            subsampling_factor = (memory_factor * 10 ** 6) / memory_total
            if subsampling_factor < 1:
                n_equations = int(np.ceil(n_img * (n_img - 1) * subsampling_factor / 2))
            else:
                n_equations = n_equations_total

        if n_equations < n_img:
            logger.warning('Too few equations. Increase memory_factor. Setting n_equations to n_img.')
            n_equations = n_img

        if n_equations < 2 * n_img:
            logger.warning('Number of equations is small. Consider increase memory_factor.')

        return n_equations

    def _generate_shift_phase_and_filter(self, r_max, max_shift, shift_step):
        """
        Prepare the shift phases and generate filter for common-line detection

        :param r_max: Maximum index for common line detection
        :param max_shift: Maximum value of 1D shift (in pixels) to search
        :param shift_step: Resolution of shift estimation in pixels
        :return: shift phases matrix and common lines filter
        """

        # Number of shifts to try
        n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))

        rk = np.arange(-r_max, r_max + 1)
        # Only half of ray is needed
        rk2 = rk[0:r_max]
        # Generate all shift phases
        shifts = np.array([-max_shift + i * shift_step for i in range(n_shifts)], dtype='int')
        shift_phases = np.exp(
            np.outer(-2 * np.pi * 1j * rk2 / (2 * r_max + 1),
                     np.arange(-max_shift, -max_shift + n_shifts * shift_step)))

        # Set filter for common-line detection
        h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * (r_max / 4) ** 2))
        return shifts, shift_phases, h

    def _get_cl_indices(self, rotations, i, j, n_theta):
        """
        Get common line indices based on the rotations from i and j images

        :param rotations: Array of rotation matrices
        :param i: Index for i image
        :param j: Index for j image
        :param n_theta: Total number of common lines
        :return: Common line indices for i and j images
        """
        # get the common line indices based on the rotations from i and j images
        r_i = rotations[:, :, i]
        r_j = rotations[:, :, j]
        c_ij, c_ji = self._common_line_r(r_i.T, r_j.T, 2 * n_theta)

        if c_ij >= n_theta:
            c_ij -= n_theta
            c_ji -= n_theta
        if c_ji < 0:
            c_ji += 2 * n_theta

        c_ij = int(c_ij)
        c_ji = int(c_ji)
        return c_ij, c_ji

    def _apply_filter_and_norm(self, pf_i, pf_j, r_max, h):
        """
        Apply common line filter and normalize each ray

        :param pf_i: Fourier transform of i image
        :param pf_j: Fourier transform of j image
        :param r_max: Maximum index for common line detection
        :param h: common lines filter
        :return: filtered and normalized i and j images
        """
        pf_i *= h
        pf_i[r_max - 1:r_max + 2] = 0
        pf_i /= np.linalg.norm(pf_i)
        pf_i = pf_i[0:r_max]

        pf_j *= h
        pf_j[r_max - 1:r_max + 2] = 0
        pf_j /= np.linalg.norm(pf_j)
        pf_j = pf_j[0:r_max]

        return pf_i, pf_j

    def _common_line_r(self, r1, r2, ell):
        """
        Compute the common line induced by rotation matrices r1 and r2.
        """
        ut = np.dot(r2, r1.T)
        alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
        alpha_ji = np.arctan2(ut[0, 2], -ut[1, 2]) + np.pi

        ell_ij = alpha_ij * ell / (2 * np.pi)
        ell_ji = alpha_ji * ell / (2 * np.pi)

        ell_ij = np.mod(np.round(ell_ij), ell)
        ell_ji = np.mod(np.round(ell_ji), ell)

        return ell_ij, ell_ji

    def _fuzzy_mask(self, n, r0, origin=None):
        """
        Create a centered 2d disc of radius r0

        Made with an error function with effective rise time.
        Todo: probably move it to utils folder

        """
        if isinstance(n, int):
            n = np.array([n])

        if isinstance(r0, int):
            r0 = np.array([r0])

        k = 1.782 / self.rise_time

        if self.fuzzy_mask_dims == 2:
            if origin is None:
                origin = np.floor(n / 2) + 1
                origin = origin.astype('int')
            if len(n) == 1:
                x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
            else:
                x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1]

            if len(r0) < 2:
                r = np.sqrt(np.square(x) + np.square(y))
            else:
                r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]))
        else:
            logger.error('only 2D is allowed!')

        m = 0.5 * (1 - sp.erf(k * (r - r0[0])))

        return m
