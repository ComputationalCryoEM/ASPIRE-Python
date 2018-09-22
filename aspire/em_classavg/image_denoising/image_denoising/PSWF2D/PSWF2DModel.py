from aspire.em_classavg.image_denoising.image_denoising.PSWF2D.PSWF2D_utils import init_pswf_2d, t_radial_part_mat
import numpy as np


class PSWF2D:
    """
    Class for numerical evaluation te 2D Prolate Spheroidal Wave Functions
    (2D PSWFs, see TODO: put Slepians paper ref) at arbitrary points in the unit disk.
    The evaluation is based on the paper TODO: put Yoel's paper title.
    Given c (bandlimit) and eps (prescribed accuracy), the class computes and stores the quantities involving all PSWFs
    with bandlimit c whose associated (normalized) eigenvalues are above the threshold eps.
    :param c: bandlimit
    :param eps: epsilon
    Attributes:
        alpha_all (list of arrays):
        alpha = alpha_all[i] contains all the eigenvalues for N=i such that lambda> eps, where lambda is the normalized
        alpha values (i.e. lambda is between 0 and 1) , given by lambda=sqrt(c*np.absolute(alpha)/(2*pi)).
        d_vec_all (list of 2D lists): the corresponding eigenvectors for alpha_all
        lengths (list of ints): lengths[i] = len(alpha_all[i])
    """

    def __init__(self, c, eps=1e-16):
        self.c = c
        self.eps = eps
        self.d_vec_all, self.alpha_all, self.lengths = init_pswf_2d(c, eps)

    def evaluate_all(self, x, y, max_ns):
        """
        Evaluate for all N's, up to certain given n for each N
        :param x: Radial part to evaluate
        :param y: Phase part to evaluate
        :param max_ns: List of ints max_ns[i] is max n to to use for N=i, not included. If max_ns[i]<1  N=i wont be used
        :return: (len(x), sum(max_ns)) ndarray
            Indices are corresponding to the list (N, n)
            (0, 0),..., (0, max_ns[0]), (1, 0),..., (1, max_ns[1]),... , (len(max_ns)-1, 0), (len(max_ns)-1, max_ns[-1])
        """
        max_ns_ints = [int(max_n) for max_n in max_ns]
        out_mat = []
        for i, max_n in enumerate(max_ns_ints):
            if max_n < 1:
                continue

            d_vec = self.d_vec_all[i]

            phase_part = np.exp(1j * i * y) / np.sqrt(2 * np.pi)
            range_array = np.arange(len(d_vec))
            r_radial_part_mat = t_radial_part_mat(x, i, range_array, len(d_vec)).dot(d_vec[:, :max_n])

            # pswf_n_n_mat = r_radial_part_mat * phase_part.reshape((len(phase_part), 1)).dot(np.ones((1, max_n)))
            pswf_n_n_mat = (phase_part * r_radial_part_mat.T)

            out_mat.extend(pswf_n_n_mat)
        out_mat = np.array(out_mat).T
        return out_mat
