import logging
import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from aspire.orientation import CLOrient3D
from aspire.utils import ensure

logger = logging.getLogger(__name__)


class CommLineSync(CLOrient3D):
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
    def __init__(self, src, n_rad=None, n_theta=None):
        """
        Initialize an object for estimating 3D orientations using synchronization matrix

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        """
        super().__init__(src, n_rad=n_rad, n_theta=n_theta)
        self.syncmatrix = None

    def estimate_rotations(self):
        """
        Estimate orientation matrices for all 2D images using synchronization matrix
        """
        if self.syncmatrix is None:
            self.syncmatrix_vote()

        S = self.syncmatrix
        sz = S.shape
        ensure(sz[0] == sz[1], 'clmatrix must be a square matrix.')
        ensure(sz[0] % 2 == 0, 'clmatrix must be a square matrix of size 2Kx2K.')

        n_check = sz[0] // 2

        # S is a 2Kx2K matrix (K=n_check), containing KxK blocks of size 2x2.
        # The [i,j] block is given by [r11 r12; r12 r22], where
        # r_{kl}=<R_{i}^{k},R_{j}^{l}>, k,l=1,2, namely, the dot product of
        # column k of R_{i} and columns l of R_{j}. Thus, given the true
        # rotations R_{1},...,R_{K}, S is decomposed as S=W^{T}W where
        # W=(R_{1}^{1},R_{1}^{2},...,R_{K}^{1},R_{K}^{2}), where R_{j}^{k}
        # the k column of R_{j}. Therefore, S is a rank-3 matrix, and thus, it
        # three eigenvectors that correspond to non-zero eigenvealues, are linear
        # combinations of the column space of S, namely, W^{T}.

        # Extract three eigenvectors corresponding to non-zero eigenvalues.
        d, v = sps.linalg.eigs(S, 10)
        d = np.real(d)
        sort_idx = np.argsort(-d)
        v = np.real(v[:, sort_idx[:3]])

        # According to the structure of W^{T} above, the odd rows of V, denoted V1,
        # are a linear combination of the vectors R_{i}^{1}, i=1,...,K, that is of
        # column 1 of all rotation matrices. Similarly, the even rows of V,
        # denoted, V2, are linear combinations of R_{i}^{1}, i=1,...,K.
        v1 = v[:2*n_check:2].T.copy()
        v2 = v[1:2*n_check:2].T.copy()

        # We look for a linear transformation (3 x 3 matrix) A such that
        # A*V1'=R1 and A*V2=R2 are the columns of the rotations matrices.
        # Therefore:
        # V1 * A'*A V1' = 1
        # V2 * A'*A V2' = 1
        # V1 * A'*A V2' = 0
        # These are 3*K linear equations for 9 matrix entries of A'*A
        # Actually, there are only 6 unknown variables, because A'*A is symmetric.

        # Obtain 3*K equations in 9 variables (3 x 3 matrix entries).
        equations = np.zeros((3*n_check, 9))
        for i in range(3):
            for j in range(3):
                equations[0::3, 3*i+j] = v1[i] * v1[j]
                equations[1::3, 3*i+j] = v2[i] * v2[j]
                equations[2::3, 3*i+j] = v1[i] * v2[j]

        # Truncate from 9 variables to 6 variables corresponding
        # to the upper half of the matrix A'*A
        truncated_equations = equations[:, [0, 1, 2, 4, 5, 8]]

        # b = [1 1 0 1 1 0 ...]' is the right hand side vector
        b = np.ones(3 * n_check)
        b[2::3] = 0

        # Find the least squares approximation of A'*A in vector form
        ata_vec = np.linalg.lstsq(truncated_equations, b, rcond=None)[0]

        # Construct the matrix A'*A from the vectorized matrix.
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

        # The Cholesky decomposition of A'*A gives A
        # numpy returns lower, matlab upper
        a = np.linalg.cholesky(ata)

        # Recover the rotations. The first two columns of all rotation
        # matrices are given by unmixing V1 and V2 using A. The third
        # column is the cross product of the first two.
        r1 = np.dot(a, v1)
        r2 = np.dot(a, v2)
        r3 = np.cross(r1, r2, axis=0)

        rotations = np.empty((n_check, 3, 3))
        rotations[:, :, 0] = r1.T
        rotations[:, :, 1] = r2.T
        rotations[:, :, 2] = r3.T
        # Make sure that we got rotations by enforcing R to be
        # a rotation (in case the error is large)
        u, _, v = np.linalg.svd(rotations)
        np.einsum('ijk, ikl -> ijl', u, v, out=rotations)

        self.rotations = rotations.transpose((1, 2, 0)).copy()

    def syncmatrix_vote(self):
        """
        Construct the synchronization matrix using voting method

        A pre-computed common line matrix is required as input.
        """
        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        sz = clmatrix.shape
        ell = self.n_theta

        if sz[0] != sz[1]:
            raise ValueError('clmatrix must be a square matrix.')

        k = sz[0]
        s = np.eye(2 * k)

        for i in range(k - 1):
            stmp = np.zeros((2, 2, k))
            for j in range(i + 1, k):
                stmp[:, :, j] = self._syncmatrix_ij_vote(
                    clmatrix, i, j, np.arange(k), ell)

            for j in range(i + 1, k):
                r22 = stmp[:, :, j]
                s[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = r22
                s[2 * j:2 * (j + 1), 2 * i:2 * (i + 1)] = r22.T

        self.syncmatrix = s

    def _syncmatrix_ij_vote(self, clmatrix, i, j, k_list, n_theta):
        """
        Compute the (i,j) rotation block of the synchronization matrix using voting method

        Given the common lines matrix `clmatrix` and a list of images specified in klist
        and the number of common lines n_theta.
        :param clmatrix: The common lines matrix
        :param i: The i image
        :param j: The j image
        :param k_list: The list of images for the third image for voting algorithm
        :param n_theta: The number of points in the theta direction (common lines)
        :return: The (i,j) rotation block of the synchronization matrix
        """
        tol = 1e-12

        good_k, _, _ = self._vote_ij(clmatrix, n_theta, i, j, k_list)

        rs, good_rotations = self._rotratio_eulerangle_vec(
            clmatrix, i, j, good_k, n_theta)

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

        Given a common lines matrix, where the index of each common line
        is in the range of n_theta and a list of good image k from voting results.
        :param clmatrix: The common lines matrix
        :param i: The i image
        :param j: The j image
        :param good_k: The list of good images k from voting algorithm
        :param n_theta: The number of points in the theta direction (common lines)
        :return: The rotation matrix that takes image i to image j for good index of k.
        """
        
        r = np.zeros((3, 3, len(good_k)))
        if i == j:
            return 0, 0

        tol = 1e-12

        idx1 = cl[good_k, j] - cl[good_k, i]    #  theta3
        idx2 = cl[j, good_k] - cl[j, i]         # -theta2
        idx3 = cl[i, good_k] - cl[i, j]         #  theta1

        a = np.cos(2 * np.pi * idx1 / n_theta)  # c3
        b = np.cos(2 * np.pi * idx2 / n_theta)  # c2
        c = np.cos(2 * np.pi * idx3 / n_theta)  # c1

        # Make sure that the triangle is not too small. This will happen if the
        # common line between (say) cl(1,2) is close to cl(1,3).
        # To eliminate that, we require that det(G)=1+2abc-(a^2+b^2+c^2) is large.
        # enough.

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

        # Fix the angles between c_ij(c_ji) and c_ik(c_jk) to be smaller than pi/2
        # otherwise there will be an ambiguity between alpha and pi-alpha.
        ind1 = np.logical_or(idx3 > n_theta / 2 + tol,
                             np.logical_and(idx3 < -tol, idx3 > -n_theta / 2))
        ind2 = np.logical_or(idx2 > n_theta / 2 + tol,
                             np.logical_and(idx2 < -tol, idx2 > -n_theta / 2))
        c_alpha[np.logical_xor(ind1, ind2)] = -c_alpha[np.logical_xor(ind1, ind2)]

        aa = cl[i, j] * 2 * np.pi / n_theta
        bb = cl[j, i] * 2 * np.pi / n_theta
        alpha = np.arccos(c_alpha)

        # Convert the Euler angles with ZXZ conversion to rotation matrices
        # Euler angle (a,b,c) to rotation
        # ra = [  ca,  -sa,   0;
        #         sa,   ca,   0;
        #          0,    0,   1]
        # rb = [   1,    0,   0;
        #          0,   cb, -sb;
        #          0,   sb,  cb]
        # rc = [  cc,  -sc,   0;
        #         sc,   cc,   0;
        #          0,    0,   1]
        # orthm = rc*rb*ra
        # ca is short for cos(a) and sa is for sin(a).
        #
        # This function does the conversion simultaneously for N Euler angles.

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

    def _vote_ij(self, clmatrix, n_theta, i, j, k_list):
        """
        Apply the voting algorithm for images i and j.

        clmatrix is the common lines matrix, constructed using angular resolution,
        n_theta. k_list are the images to be used for voting of the pair of images
        (i ,j).
        :param clmatrix: The common lines matrix
        :param n_theta: The number of points in the theta direction (common lines)
        :param i: The i image
        :param j: The j image
        :param k_list: The list of images for the third image for voting algorithm
        :return:  good_k, the list of all third images in the peak of the histogram
            corresponding to the pair of images (i,j); alpha, the estimated angle
            between them; and the rotation matrix that takes image i to image j
            for good index of k.
        """
        # Parameters used to compute the smoothed angle histogram.
        ntics = 60
        x = np.linspace(0, 180, ntics, True)

        # Angle between i and i induced by each third image k.
        # The second column is the cosine of that angle,
        # The first column is the index l in k_list of the image that
        # creates that angle.
        phis = np.zeros((len(k_list), 2))
        rejected = np.zeros(len(k_list))
        idx = 0
        rej_idx = 0
        if i != j and clmatrix[i, j] != -1:
            # Some of the entries in clmatrix may be zero if we cleared
            # them due to small correlation, or if for each image
            # we compute intersections with only some of the other images.
            # l_idx=clmatrix[[i j k],[i j k]]
            #
            # Note that as long as the diagonal of the common lines matrix is
            # zero, the conditions (i != j) && (j != k) are not needed, since
            # if i == j then clmatrix[i, k] == 0 and similarly for i == k or
            # j == k. Thus, the previous voting code (from the JSB paper) is
            # correct even though it seems that we should test also that
            # (i != j) && (i != k) && (j != k), and only (i != j) && (i != k)
            #  as tested there.

            l_idx12 = clmatrix[i, j]
            l_idx21 = clmatrix[j, i]
            k_list = k_list[np.logical_and(
                np.logical_and(k_list != i, clmatrix[i, k_list] != -1), clmatrix[j, k_list] != -1)]

            l_idx13 = clmatrix[i, k_list]
            l_idx31 = clmatrix[k_list, i]
            l_idx23 = clmatrix[j, k_list]
            l_idx32 = clmatrix[k_list, j]

            # theta1 is the angle on C1 created by its intersection with C3 and C2.
            # theta2 is the angle on C2 created by its intersection with C1 and C3.
            # theta3 is the angle on C3 created by its intersection with C2 and C1.
            theta1 = (l_idx13 - l_idx12) * 2 * np.pi / n_theta
            theta2 = (l_idx21 - l_idx23) * 2 * np.pi / n_theta
            theta3 = (l_idx32 - l_idx31) * 2 * np.pi / n_theta

            c1 = np.cos(theta1)
            c2 = np.cos(theta2)
            c3 = np.cos(theta3)

            # Each common-line corresponds to a point on the unit sphere. Denote the
            # coordinates of these points by (Pix, Piy Piz), and put them in the matrix
            #   M=[ P1x  P2x  P3x ;
            #       P1y  P2y  P3y ;
            #       P1z  P2z  P3z ].
            #
            # Then the matrix
            #   C=[  1  c1  c2 ;
            #       c1   1  c3 ;
            #       c2  c3   1 ],
            # where c1,c2,c3 are given above, is given by C = M.T @ M.
            # For the points P1,P2, and P3 to form a triangle on the unit sphere, a
            # necessary and sufficient condition is for C to be positive definite. This
            # is equivalent to
            #       1+2*c1*c2*c3-(c1^2+c2^2+c3^2) > 0.
            # However, this may result in a triangle that is too flat, that is, the
            # angle between the projections is very close to zero. We therefore use the
            # condition below
            #       1+2*c1*c2*c3-(c1^2+c2^2+c3^2) > 1.0e-5.
            # This ensures that the smallest singular value (which is actually
            # controlled by the determinant of C) is big enough, so the matrix is far
            # from singular. This condition is equivalent to computing the singular
            # values of C, followed by checking that the smallest one is big enough.
            cond = 1 + 2 * c1 * c2 * c3 - (
                    np.square(c1) + np.square(c2) + np.square(c3))

            good_idx = np.where(cond > 1e-5)[0]
            bad_idx = np.where(cond <= 1e-5)[0]

            cos_phi2 = (c3[good_idx] - c1[good_idx] *
                        c2[good_idx]) / (np.sin(theta1[good_idx])
                                         * np.sin(theta2[good_idx]))
            check_idx = np.where(np.abs(cos_phi2) > 1)[0]
            if np.any(np.abs(cos_phi2) - 1 > 1e-12):
                logger.warning(f'Globally Consistent Angular Reconstruction(GCAR) exists'
                               ' numerical problem: abs(cos_phi2) >1, with the '
                               ' difference of {np.abs(cos_phi2)-1}.')
            elif len(check_idx) == 0:
                cos_phi2[check_idx] = np.sign(cos_phi2[check_idx])

            phis[:idx + len(good_idx), 0] = cos_phi2
            phis[:idx + len(good_idx), 1] = k_list[good_idx]
            idx += len(good_idx)

            rejected[: rej_idx + len(bad_idx)] = k_list[bad_idx]
            rej_idx += len(bad_idx)

        phis = phis[:idx]
        rejected = rejected[:rej_idx]

        good_k = []
        peakh = -1
        alpha = []

        if idx > 0:
            # Compute the histogram of the angles between images i and j.
            angles = np.arccos(phis[:, 0]) * 180 / np.pi
            # Angles that are up to 10 degrees apart are considered
            # similar. This sigma ensures that the width of the density
            # estimation kernel is roughly 10 degrees. For 15 degrees, the
            # value of the kernel is negligible.
            sigma = 3.0

            tmp = np.add.outer(np.square(angles), np.square(x))
            h = np.sum(np.exp((2 * np.multiply.outer(angles, x)
                               - tmp) / (2 * sigma ** 2)), 0)

            # We assume that at the location of the peak we get the true angle
            # between images i and j. Find all third images k, that induce an
            # angle between i and j that is at most 10 off the true angle.
            # Even for debugging, don't put a value that is smaller than two
            # tics, since the peak might move a little bit due to wrong k images
            # that accidentally fall near the peak.
            peak_idx = h.argmax()
            peakh = h[peak_idx]
            idx = np.where(np.abs(angles - x[peak_idx]) < 360 / ntics)[0]
            good_k = phis[idx, 1]
            alpha = phis[idx, 0]

        return good_k.astype('int'), peakh, alpha
