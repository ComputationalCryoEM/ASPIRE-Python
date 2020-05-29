from aspire.orientation import Orient3D
import logging
import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as spsl

logger = logging.getLogger(__name__)


class CommLineSync(Orient3D):
    """
    Define a class to estimate 3D orientations using synchronization matrix and voting method.

    The related publications are listed as below:
    Y. Shkolnisky, and A. Singer,
    Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

    A. Singer, R. R. Coifman, F. J. Sigworth, D. W. Chester, Y. Shkolnisky,
    Detecting Consistent Common Lines in Cryo-EM by Voting,
    Journal of Structural Biology, 169, 312-322 (2010).
    """
    def __init__(self, src, nrad=None, ntheta=None):
        """
        Initialize an object for estimating 3D orientations using synchronization matrix

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """
        super().__init__(src, nrad=nrad, ntheta=ntheta)
        self.syncmatrix = None

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images using synchronization matrix
        """
        if self.syncmatrix is None:
            self.syncmatrix_vote()

        s = self.syncmatrix
        sz = s.shape
        if len(sz) != 2:
            raise ValueError('clmatrix must be a square matrix')
        if sz[0] != sz[1]:
            raise ValueError('clmatrix must be a square matrix')
        if sz[0] % 2 == 1:
            raise ValueError('clmatrix must be a square matrix of size 2Kx2K')

        k = sz[0] // 2

        d, v = sps.linalg.eigs(s, 10)
        d = np.real(d)
        sort_idx = np.argsort(-d)

        v = np.real(v[:, sort_idx[:3]])
        v1 = v[:2*k:2].T.copy()
        v2 = v[1:2*k:2].T.copy()

        equations = np.zeros((3*k, 9))
        for i in range(3):
            for j in range(3):
                equations[0::3, 3*i+j] = v1[i] * v1[j]
                equations[1::3, 3*i+j] = v2[i] * v2[j]
                equations[2::3, 3*i+j] = v1[i] * v2[j]
        truncated_equations = equations[:, [0, 1, 2, 4, 5, 8]]

        b = np.ones(3 * k)
        b[2::3] = 0

        ata_vec = np.linalg.lstsq(truncated_equations, b, rcond=None)[0]
        ata = np.zeros((3, 3))
        ata[0, 0] = ata_vec[0]
        ata[0, 1] = ata_vec[1]
        ata[0, 2] = ata_vec[2]
        ata[1, 0] = ata_vec[1]
        ata[1, 1] = ata_vec[3]
        ata[1, 2] = ata_vec[4]
        ata[2, 0] = ata_vec[2]
        ata[2, 1] = ata_vec[4]
        ata[2, 2] = ata_vec[5]

        # numpy returns lower, matlab upper
        a = np.linalg.cholesky(ata)

        r1 = np.dot(a, v1)
        r2 = np.dot(a, v2)
        r3 = np.cross(r1, r2, axis=0)

        rotations = np.empty((k, 3, 3))
        rotations[:, :, 0] = r1.T
        rotations[:, :, 1] = r2.T
        rotations[:, :, 2] = r3.T
        u, _, v = np.linalg.svd(rotations)
        np.einsum('ijk, ikl -> ijl', u, v, out=rotations)
        self.rotations = rotations.transpose((1, 2, 0)).copy()

    def syncmatrix_vote(self):
        """
        Construct the synchronization matrix using voting method
        """
        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        sz = clmatrix.shape
        l = self.ntheta
        if len(sz) != 2:
            raise ValueError('clmatrix must be a square matrix')
        if sz[0] != sz[1]:
            raise ValueError('clmatrix must be a square matrix')

        k = sz[0]
        s = np.eye(2 * k)

        for i in range(k - 1):
            stmp = np.zeros((2, 2, k))
            for j in range(i + 1, k):
                stmp[:, :, j] = self._syncmatrix_ij_vote(
                    clmatrix, i, j, np.arange(k), l)

            for j in range(i + 1, k):
                r22 = stmp[:, :, j]
                s[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = r22
                s[2 * j:2 * (j + 1), 2 * i:2 * (i + 1)] = r22.T

        self.syncmatrix = s

    def _syncmatrix_ij_vote(self, clmatrix, i, j, k, l):
        """
        Compute the (i,j) rotation block of the synchronization matrix using voting method
        """
        tol = 1e-12

        good_k, _, _ = self._vote_ij(clmatrix, l, i, j, k)

        rs, good_rotations = self._rotratio_eulerangle_vec(
            clmatrix, i, j, good_k, l)

        if len(good_rotations) > 0:
            rk = np.mean(rs, 2)
            tmp_r = rs[:2, :2]
            diff = tmp_r - rk[:2, :2, np.newaxis]
            err = np.linalg.norm(diff) / np.linalg.norm(tmp_r)
            if err > tol:
                pass
        else:
            rk = np.zeros((3, 3))

        r22 = rk[:2, :2]
        return r22

    def _rotratio_eulerangle_vec(self, cl, i, j, good_k, n_theta):
        """
        Compute the rotation that takes image i to image j

        Given a 3x3 common lines matrix, where the index of each common line
        is between 1 and n_theta.
        """
        
        r = np.zeros((3, 3, len(good_k)))
        if i == j:
            return 0, 0

        tol = 1e-12

        idx1 = cl[good_k, j] - cl[good_k, i]
        idx2 = cl[j, good_k] - cl[j, i]
        idx3 = cl[i, good_k] - cl[i, j]

        a = np.cos(2 * np.pi * idx1 / n_theta)
        b = np.cos(2 * np.pi * idx2 / n_theta)
        c = np.cos(2 * np.pi * idx3 / n_theta)

        cond = 1 + 2 * a * b * c - (np.square(a)
                                    + np.square(b) + np.square(c))
        too_small_idx = np.where(cond <= 1.0e-5)[0]
        good_idx = np.where(cond > 1.0e-5)[0]

        a = a[good_idx]
        b = b[good_idx]
        c = c[good_idx]
        idx2 = idx2[good_idx]
        idx3 = idx3[good_idx]
        c_alpha = (a - b * c) / np.sqrt(1 - np.square(b)) / np.sqrt(1 - np.square(c))

        ind1 = np.logical_or(idx3 > n_theta / 2 + tol,
                             np.logical_and(idx3 < -tol, idx3 > -n_theta / 2))
        ind2 = np.logical_or(idx2 > n_theta / 2 + tol,
                             np.logical_and(idx2 < -tol, idx2 > -n_theta / 2))
        c_alpha[np.logical_xor(ind1, ind2)] = -c_alpha[np.logical_xor(ind1, ind2)]

        aa = cl[i, j] * 2 * np.pi / n_theta
        bb = cl[j, i] * 2 * np.pi / n_theta
        alpha = np.arccos(c_alpha)

        ang1 = np.pi - bb
        ang2 = alpha
        ang3 = aa - np.pi
        sa = np.sin(ang1)
        ca = np.cos(ang1)
        sb = np.sin(ang2)
        cb = np.cos(ang2)
        sc = np.sin(ang3)
        cc = np.cos(ang3)

        r[0, 0, good_idx] = cc * ca - sc * cb * sa
        r[0, 1, good_idx] = -cc * sa - sc * cb * ca
        r[0, 2, good_idx] = sc * sb
        r[1, 0, good_idx] = sc * ca + cc * cb * sa
        r[1, 1, good_idx] = -sa * sc + cc * cb * ca
        r[1, 2, good_idx] = -cc * sb
        r[2, 0, good_idx] = sb * sa
        r[2, 1, good_idx] = sb * ca
        r[2, 2, good_idx] = cb

        if len(too_small_idx) > 0:
            r[:, :, too_small_idx] = 0

        return r, good_idx

    def _vote_ij(self, clmatrix, l, i, j, k):
        """
        Apply the voting algorithm for images i and j.

        clmatrix is the common lines matrix, constructed using angular resolution l.
        K are the images used for voting of the pair (i ,j).
        """
        ntics = 60
        x = np.linspace(0, 180, ntics, True)
        phis = np.zeros((len(k), 2))
        rejected = np.zeros(len(k))
        idx = 0
        rej_idx = 0
        if i != j and clmatrix[i, j] != -1:
            l_idx12 = clmatrix[i, j]
            l_idx21 = clmatrix[j, i]
            k = k[np.logical_and(
                np.logical_and(k != i, clmatrix[i, k] != -1), clmatrix[j, k] != -1)]

            l_idx13 = clmatrix[i, k]
            l_idx31 = clmatrix[k, i]
            l_idx23 = clmatrix[j, k]
            l_idx32 = clmatrix[k, j]

            theta1 = (l_idx13 - l_idx12) * 2 * np.pi / l
            theta2 = (l_idx21 - l_idx23) * 2 * np.pi / l
            theta3 = (l_idx32 - l_idx31) * 2 * np.pi / l

            c1 = np.cos(theta1)
            c2 = np.cos(theta2)
            c3 = np.cos(theta3)

            cond = 1 + 2 * c1 * c2 * c3 - (
                    np.square(c1) + np.square(c2) + np.square(c3))

            good_idx = np.where(cond > 1e-5)[0]
            bad_idx = np.where(cond <= 1e-5)[0]

            cos_phi2 = (c3[good_idx] - c1[good_idx] *
                        c2[good_idx]) / (np.sin(theta1[good_idx])
                                         * np.sin(theta2[good_idx]))
            check_idx = np.where(np.abs(cos_phi2) > 1)[0]
            if np.any(np.abs(cos_phi2) - 1 > 1e-12):
                logger.warning('GCAR:numericalProblem')
            elif len(check_idx) == 0:
                cos_phi2[check_idx] = np.sign(cos_phi2[check_idx])

            phis[:idx + len(good_idx), 0] = cos_phi2
            phis[:idx + len(good_idx), 1] = k[good_idx]
            idx += len(good_idx)

            rejected[: rej_idx + len(bad_idx)] = k[bad_idx]
            rej_idx += len(bad_idx)

        phis = phis[:idx]
        rejected = rejected[:rej_idx]

        good_k = []
        peakh = -1
        alpha = []

        if idx > 0:
            angles = np.arccos(phis[:, 0]) * 180 / np.pi
            sigma = 3.0

            tmp = np.add.outer(np.square(angles), np.square(x))
            h = np.sum(np.exp((2 * np.multiply.outer(angles, x)
                               - tmp) / (2 * sigma ** 2)), 0)
            peak_idx = h.argmax()
            peakh = h[peak_idx]
            idx = np.where(np.abs(angles - x[peak_idx]) < 360 / ntics)[0]
            good_k = phis[idx, 1]
            alpha = phis[idx, 0]

        return good_k.astype('int'), peakh, alpha
