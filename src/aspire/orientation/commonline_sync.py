import logging
import numpy as np

import scipy.sparse as sparse

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
        ensure(sz[0] == sz[1], 'syncmatrix must be a square matrix.')
        ensure(sz[0] % 2 == 0, 'syncmatrix must be a square matrix of size 2Kx2K.')

        n_img = sz[0] // 2

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
        d, v = sparse.linalg.eigs(S, 10)
        d = np.real(d)
        sort_idx = np.argsort(-d)
        v = np.real(v[:, sort_idx[:3]])

        # According to the structure of W^{T} above, the odd rows of V, denoted V1,
        # are a linear combination of the vectors R_{i}^{1}, i=1,...,K, that is of
        # column 1 of all rotation matrices. Similarly, the even rows of V,
        # denoted, V2, are linear combinations of R_{i}^{1}, i=1,...,K.
        v1 = v[:2*n_img:2].T.copy()
        v2 = v[1:2*n_img:2].T.copy()

        # We look for a linear transformation (3 x 3 matrix) A such that
        # A*V1'=R1 and A*V2=R2 are the columns of the rotations matrices.
        # Therefore:
        # V1 * A'*A V1' = 1
        # V2 * A'*A V2' = 1
        # V1 * A'*A V2' = 0
        # These are 3*K linear equations for 9 matrix entries of A'*A
        # Actually, there are only 6 unknown variables, because A'*A is symmetric.

        # Obtain 3*K equations in 9 variables (3 x 3 matrix entries).
        equations = np.zeros((3*n_img, 9))
        for i in range(3):
            for j in range(i, 3):
                equations[0::3, 3*i+j] = v1[i] * v1[j]
                equations[1::3, 3*i+j] = v2[i] * v2[j]
                equations[2::3, 3*i+j] = v1[i] * v2[j]

        # Truncate from 9 variables to 6 variables corresponding
        # to the upper half of the matrix A'*A
        truncated_equations = equations[:, [0, 1, 2, 4, 5, 8]]

        # b = [1 1 0 1 1 0 ...]' is the right hand side vector
        b = np.ones(3 * n_img)
        b[2::3] = 0

        # Find the least squares approximation of A'*A in vector form
        ATA_vec = np.linalg.lstsq(truncated_equations, b, rcond=None)[0]

        # Construct the matrix A'*A from the vectorized matrix.
        ATA = np.zeros((3, 3))
        upper_mask = np.triu(np.full((3, 3), True))
        ATA[upper_mask] = ATA_vec
        lower_mask = np.tril(np.full((3, 3), True))
        ATA[lower_mask] = ATA.T[lower_mask]

        # The Cholesky decomposition of A'*A gives A
        # numpy returns lower, matlab upper
        a = np.linalg.cholesky(ATA)

        # Recover the rotations. The first two columns of all rotation
        # matrices are given by unmixing V1 and V2 using A. The third
        # column is the cross product of the first two.
        r1 = np.dot(a, v1)
        r2 = np.dot(a, v2)
        r3 = np.cross(r1, r2, axis=0)

        rotations = np.empty((n_img, 3, 3))
        rotations[:, :, 0] = r1.T
        rotations[:, :, 1] = r2.T
        rotations[:, :, 2] = r3.T
        # Make sure that we got rotations by enforcing R to be
        # a rotation (in case the error is large)
        u, _, v = np.linalg.svd(rotations)
        np.einsum('ijk, ikl -> ijl', u, v, out=rotations)

        self.rotations = rotations

    def syncmatrix_vote(self):
        """
        Construct the synchronization matrix using voting method

        A pre-computed common line matrix is required as input.
        """
        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        sz = clmatrix.shape
        n_theta = self.n_theta

        ensure(sz[0] == sz[1], 'clmatrix must be a square matrix.')

        n_img = sz[0]
        S = np.eye(2 * n_img)

        # Build Synchronization matrix from the rotation blocks in X and Y
        for i in range(n_img - 1):
            for j in range(i + 1, n_img):
                rot_block = self._syncmatrix_ij_vote(
                    clmatrix, i, j, np.arange(n_img), n_theta)
                S[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = rot_block
                S[2 * j:2 * (j + 1), 2 * i:2 * (i + 1)] = rot_block.T

        self.syncmatrix = S

    def _syncmatrix_ij_vote(self, clmatrix, i, j, k_list, n_theta):
        """
        Compute the (i,j) rotation block of the synchronization matrix using voting method

        Given the common lines matrix `clmatrix`, a list of images specified in k_list
        and the number of common lines n_theta, find the (i, j) rotation block (in X and Y)
        of the synchronization matrix.
        :param clmatrix: The common lines matrix
        :param i: The i image
        :param j: The j image
        :param k_list: The list of images for the third image for voting algorithm
        :param n_theta: The number of points in the theta direction (common lines)
        :return: The (i,j) rotation block of the synchronization matrix
        """
        tol = 1e-12

        good_k = self._vote_ij(clmatrix, n_theta, i, j, k_list)

        rots = self._rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta)

        if rots.shape[2] > 0:
            rot_mean = np.mean(rots, 2)
            rot_block = rots[:2, :2]
            diff = rot_block - rot_mean[:2, :2, np.newaxis]
            err = np.linalg.norm(diff) / np.linalg.norm(rot_block)
            if err > tol:
                # This means that images i and j have inconsistent rotations.
                # simply pass it as Matlab code.
                pass
        else:
            # This for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # Simply put to zero as Matlab code.
            rot_mean = np.zeros((3, 3))

        # return the rotation matrix in X and Y
        r22 = rot_mean[:2, :2]
        return r22

    def _rotratio_eulerangle_vec(self, clmatrix, i, j, good_k, n_theta):
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

        idx1 = clmatrix[good_k, j] - clmatrix[good_k, i]    #  theta3
        idx2 = clmatrix[j, good_k] - clmatrix[j, i]         # -theta2
        idx3 = clmatrix[i, good_k] - clmatrix[i, j]         #  theta1

        a = np.cos(2 * np.pi * idx1 / n_theta)  # c3
        b = np.cos(2 * np.pi * idx2 / n_theta)  # c2
        c = np.cos(2 * np.pi * idx3 / n_theta)  # c1

        # Make sure that the triangle is not too small. This will happen if the
        # common line between (say) cl(1,2) is close to cl(1,3).
        # To eliminate that, we require that det(G)=1+2abc-(a^2+b^2+c^2) is large.
        # enough.

        cond = 1 + 2 * a * b * c - (np.square(a)
                                    + np.square(b) + np.square(c))

        good_idx = np.where(cond > 1.0e-5)[0]

        a = a[good_idx]
        b = b[good_idx]
        c = c[good_idx]
        idx2 = idx2[good_idx]
        idx3 = idx3[good_idx]
        c_alpha = (a - b * c) / np.sqrt(1 - np.square(b)) / np.sqrt(1 - np.square(c))

        # Fix the angles between c_ij(c_ji) and c_ik(c_jk) to be smaller than pi/2
        # otherwise there will be an ambiguity between alpha and pi-alpha.
        ind1 = (idx3 > n_theta / 2 + tol) | ((idx3 < -tol) & (idx3 > -n_theta / 2))
        ind2 = (idx2 > n_theta / 2 + tol) | ((idx2 < -tol) & (idx2 > -n_theta / 2))
        c_alpha[ind1 ^ ind2] = -c_alpha[ind1 ^ ind2]
        aa = clmatrix[i, j] * 2 * np.pi / n_theta
        bb = clmatrix[j, i] * 2 * np.pi / n_theta
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

        return r[:, :, good_idx]

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
            corresponding to the pair of images (i,j)
        """

        # Angle between i and j induced by each third image k.
        phis = np.zeros(len(k_list))
        # the index l in k_list of the image that creates that angle.
        inds = np.zeros(len(k_list))

        idx = 0

        if i == j or clmatrix[i, j] == -1:
            return []

        if i != j and clmatrix[i, j] != -1:
            # Some of the entries in clmatrix may be zero if we cleared
            # them due to small correlation, or if for each image
            # we compute intersections with only some of the other images.
            #
            # Note that as long as the diagonal of the common lines matrix is
            # -1, the conditions (i != j) && (j != k) are not needed, since
            # if i == j then clmatrix[i, k] == -1 and similarly for i == k or
            # j == k. Thus, the previous voting code (from the JSB paper) is
            # correct even though it seems that we should test also that
            # (i != j) && (i != k) && (j != k), and only (i != j) && (i != k)
            #  as tested there.
            cl_idx12 = clmatrix[i, j]
            cl_idx21 = clmatrix[j, i]
            k_list = k_list[(k_list != i) & (clmatrix[i, k_list] != -1) & (clmatrix[j, k_list] != -1)]
            cl_idx13 = clmatrix[i, k_list]
            cl_idx31 = clmatrix[k_list, i]
            cl_idx23 = clmatrix[j, k_list]
            cl_idx32 = clmatrix[k_list, j]

            # C1, C2, and C3 are unit circles of image i, j, and k
            # theta1 is the angle on C1 created by its intersection with C3 and C2.
            # theta2 is the angle on C2 created by its intersection with C1 and C3.
            # theta3 is the angle on C3 created by its intersection with C2 and C1.
            theta1 = (cl_idx13 - cl_idx12) * 2 * np.pi / n_theta
            theta2 = (cl_idx21 - cl_idx23) * 2 * np.pi / n_theta
            theta3 = (cl_idx32 - cl_idx31) * 2 * np.pi / n_theta

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
            # where c1, c2, c3 are given above, is given by C = M.T @ M.
            # For the points P1, P2, and P3 to form a triangle on the unit sphere, a
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

            # Calculated cos values of angle between i and j images
            cos_phi2 = (c3[good_idx] - c1[good_idx] *
                        c2[good_idx]) / (np.sin(theta1[good_idx])
                                         * np.sin(theta2[good_idx]))
            if np.any(np.abs(cos_phi2) - 1 > 1e-12):
                logger.warning(f'Globally Consistent Angular Reconstruction (GCAR) exists'
                               f' numerical problem: abs(cos_phi2) > 1, with the'
                               f' difference of {np.abs(cos_phi2)-1}.')
            cos_phi2 = np.clip(cos_phi2, -1, 1)
            phis[:len(good_idx)] = cos_phi2
            inds[:len(good_idx)] = k_list[good_idx]
            idx += len(good_idx)

        phis = phis[:idx]
        if phis.shape[0] == 0:
                return []
       
        good_k = []
        if idx > 0:
            # Parameters used to compute the smoothed angle histogram.
            ntics = 60
            angles_grid = np.linspace(0, 180, ntics, True)
            # Get angles between images i and j for computing the histogram
            angles = np.arccos(phis[:]) * 180 / np.pi
            # Angles that are up to 10 degrees apart are considered
            # similar. This sigma ensures that the width of the density
            # estimation kernel is roughly 10 degrees. For 15 degrees, the
            # value of the kernel is negligible.
            sigma = 3.0

            # Compute the histogram of the angles between images i and j
            squared_values = np.add.outer(np.square(angles), np.square(angles_grid))
            angles_hist = np.sum(np.exp((2 * np.multiply.outer(angles, angles_grid)
                               - squared_values) / (2 * sigma ** 2)), 0)

            # We assume that at the location of the peak we get the true angle
            # between images i and j. Find all third images k, that induce an
            # angle between i and j that is at most 10 off the true angle.
            # Even for debugging, don't put a value that is smaller than two
            # tics, since the peak might move a little bit due to wrong k images
            # that accidentally fall near the peak.
            peak_idx = angles_hist.argmax()
            idx = np.where(np.abs(angles - angles_grid[peak_idx]) < 360 / ntics)[0]
            good_k = inds[idx]

        return good_k.astype('int')
