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
    def __init__(self, src, n_rad=None, n_theta=None, masked=True):
        """
        Initialize an object for estimating 3D orientations

        :param src: The source object of 2D denoised or class-averaged imag
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param masked: Whether applying a mask to the images or not
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

        self.basis = PolarBasis2D((self.n_res, self.n_res), self.n_rad, self.n_theta)

        self.max_shift = math.ceil(config.orient.max_shift * self.n_res)
        self.shift_step = config.orient.shift_step

        imgs = self.src.images(start=0, num=np.inf).asnumpy()

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
            raise ValueError('n_theta must be even')
        n_theta = n_theta // 2
        self.pf = np.concatenate((np.flip(self.pf[1:, n_theta:], 0), self.pf[:, :n_theta]), 0)

        self.clmatrix = None
        self.cormatrix = None
        self.shift_equations = None
        self.shift_equations_map = None
        self.clmatrix_mask = None
        self.rotations = None

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

        :param n_check: For each projection find its common-lines with n_check
            projections. If n_check is less than the total number a projection,
            a random subset of nk projections is used.
        """
        max_shift = self.max_shift
        shift_step = self.shift_step
        n_img = self.n_img
        if n_check is None:
            n_check = n_img
        n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
        n_theta = self.n_theta
        if n_theta % 2 == 1:
            raise ValueError('n_theta must be even')
        n_theta = n_theta // 2

        pf = self.pf

        # Allocate variables
        clstack = -np.ones((n_img, n_img))
        corrstack = np.zeros((n_img, n_img))
        clstack_mask = np.zeros((n_img, n_img))

        # Allocate variables used for shift estimation
        shifts_1d = np.zeros((n_img, n_img))
        shift_i = np.zeros(4 * n_img * n_check)
        shift_j = np.zeros(4 * n_img * n_check)
        shift_eq = np.zeros(4 * n_img * n_check)
        shift_equations_map = np.zeros((n_img, n_img))
        shift_equation_idx = 0
        shift_b = np.zeros(n_img * (n_img - 1) // 2)
        dtheta = np.pi / n_theta

        # Search for common lines between pairs of projections.
        # Creating pf3 and building common lines are different to Matlab version.
        # The random selection is implemented.
        r_max = int((pf.shape[0] - 1) / 2)
        rk = np.arange(-r_max, r_max + 1)
        h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * np.square(r_max / 4)))

        pf3 = np.empty(pf.shape, dtype=pf.dtype)
        np.einsum('ijk, i -> ijk', pf, h, out=pf3)
        pf3[r_max - 1:r_max + 2] = 0
        pf3 /= np.linalg.norm(pf3, axis=0)
        pf3 = pf3[:r_max]
        pf3 = pf3.transpose((2, 1, 0))

        rk2 = rk[:r_max]

        all_shift_phases = np.zeros((n_shifts, r_max), 'complex128')
        shifts = np.array([-max_shift + i * shift_step for i in range(n_shifts)], dtype='int')
        for i in range(n_shifts):
            shift = shifts[i]
            all_shift_phases[i] = np.exp(-2 * np.pi * 1j * rk2 * shift / (2 * r_max + 1))

        for i in range(n_img):
            n2 = min(n_img - i, n_check)

            if n_img - i - 1 == 0:
                subset_k2 = []
            else:
                subset_k2 = np.sort(np.random.choice(n_img - i - 1, n2 - 1,
                                                     replace=False) + i + 1)

            proj1 = pf3[i]
            p1 = proj1
            p1_real = np.real(p1)
            p1_imag = np.imag(p1)

            for j in subset_k2:
                proj2 = pf3[j]
                p2_flipped = np.conj(proj2)

                for shift in range(n_shifts):
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

        tmp = np.where(corrstack != 0)
        corrstack[tmp] = 1 - corrstack[tmp]
        l = 4 * shift_equation_idx
        shift_eq[l: l + shift_equation_idx] = shift_b
        shift_i[l: l + shift_equation_idx] = np.arange(shift_equation_idx)
        shift_j[l: l + shift_equation_idx] = 2 * n_img
        tmp = np.where(shift_eq != 0)[0]
        shift_eq = shift_eq[tmp]
        shift_i = shift_i[tmp]
        shift_j = shift_j[tmp]
        l += shift_equation_idx
        shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)),
                                         shape=(shift_equation_idx, 2 * n_img + 1))
        self.clmatrix = clstack
        self.cormatrix = corrstack
        self.shift_equations = shift_equations
        self.shift_equations_map = shift_equations_map
        self.clmatrix_mask = clstack_mask

    def estimate_shifts(self, memory_factor=10000):
        """
        Estimate 2D shifts in images

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

        max_shift = self.max_shift
        shift_step = self.shift_step

        if memory_factor < 0 or (memory_factor > 1 and memory_factor < 100):
            raise ValueError('Subsampling factor must be between 0 and 1 or larger than 100.')

        n_theta = self.n_theta // 2
        n_img = self.n_img
        rotations = self.rotations

        pf = self.pf
        n_equations_total = int(np.ceil(n_img * (n_img - 1) / 2))
        memory_total = n_equations_total * 2 * n_img * 8

        if memory_factor <= 1:
            n_equations = int(np.ceil(n_img * (n_img - 1) * memory_factor / 2))
        else:
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

        shift_i = np.zeros(4 * n_equations + n_equations)
        shift_j = np.zeros(4 * n_equations + n_equations)
        shift_eq = np.zeros(4 * n_equations + n_equations)
        shift_b = np.zeros(n_equations)

        n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
        r_max = (pf.shape[0] - 1) // 2
        rk = np.arange(-r_max, r_max + 1)
        rk2 = rk[:r_max]
        shift_phases = np.exp(
            np.outer(-2 * np.pi * 1j * rk2 / (2 * r_max + 1),
                     np.arange(-max_shift, -max_shift + n_shifts * shift_step)))

        h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * (r_max / 4) ** 2))

        d_theta = np.pi / n_theta

        idx_i = []
        idx_j = []
        for i in range(n_img):
            tmp_j = range(i + 1, n_img)
            idx_i.extend([i] * len(tmp_j))
            idx_j.extend(tmp_j)
        idx_i = np.array(idx_i, dtype='int')
        idx_j = np.array(idx_j, dtype='int')
        rp = np.random.choice(np.arange(len(idx_j)), size=n_equations, replace=False)

        # might be able to vectorize this
        for shift_eq_idx in range(n_equations):
            i = idx_i[rp[shift_eq_idx]]
            j = idx_j[rp[shift_eq_idx]]

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
            is_pf_j_flipped = 0
            if c_ji < n_theta:
                pf_j = pf[:, c_ji, j]
            else:
                pf_j = pf[:, c_ji - n_theta, j]
                is_pf_j_flipped = 1
            pf_i = pf[:, c_ij, i]

            pf_i *= h
            pf_i[r_max - 1:r_max + 2] = 0
            pf_i /= np.linalg.norm(pf_i)
            pf_i = pf_i[:r_max]

            pf_j *= h
            pf_j[r_max - 1:r_max + 2] = 0
            pf_j /= np.linalg.norm(pf_j)
            pf_j = pf_j[:r_max]

            pf_i_flipped = np.conj(pf_i)
            pf_i_stack = np.einsum('i, ij -> ij', pf_i, shift_phases)
            pf_i_flipped_stack = np.einsum('i, ij -> ij', pf_i_flipped, shift_phases)

            c1 = 2 * np.real(np.dot(np.conj(pf_i_stack.T), pf_j))
            c2 = 2 * np.real(np.dot(np.conj(pf_i_flipped_stack.T), pf_j))

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

        est_shifts = np.linalg.lstsq(shift_equations[:, :-1].todense(), shift_b, rcond=-1)[0]
        est_shifts = est_shifts.reshape((2, n_img), order='F')

        return est_shifts, shift_equations

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
