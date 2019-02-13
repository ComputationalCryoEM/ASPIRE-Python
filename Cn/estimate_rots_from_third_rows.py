import Cn.utils as utils
from Cn.config_symm import AbinitioSymmConfig
import numpy as np
import scipy
from tqdm import tqdm


def estimate_rots_from_third_rows_c2(vis, Rijs):
    """
    estimate the rotation matrices given the third row of each rotation matrix and the relative orientations
    :param vis: an mx3 array holding the third row of each of the unknown m rotation matrices
    :param Rijs: an mchoose2x2x3x3 array where the ij-th entry holds the two 3x3 estimates of RiRj and RigRj
    :return: an mx3x3 array of estimated rotation matrices
    """
    print('estimating in-plane rotation angles')
    n_images = len(vis)
    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    assert len(Rijs) == m_choose_2
    #  Step 1: construct all rotation matrices Ri_tildes whose third row is equal to the corresponding third rows vis
    Ri_tildes = np.array([utils.complete_third_row_to_rot(vi) for vi in vis])
    H = np.zeros((n_images, n_images), dtype=complex)
    c = 0
    for i in range(n_images):
        for j in range(i+1, n_images):
            Ri_tilde = Ri_tildes[i]
            Rj_tilde = Ri_tildes[j]
            Rij = Rijs[c, 0]
            Rij_g = Rijs[c, 1]

            Uij = np.linalg.multi_dot([Ri_tilde, Rij, Rj_tilde.T])  # U_ij is x-y-plane rotation
            u, _, vh = np.linalg.svd(Uij[:-1, :-1])
            Uij = np.dot(u, vh)

            Uij_g = np.linalg.multi_dot([Ri_tilde, Rij_g, Rj_tilde.T])  # U_ij is x-y-plane rotation
            u_g, _, vh_g = np.linalg.svd(Uij_g[:-1, :-1])
            Uij_g = np.dot(u_g, vh_g)

            U = (np.dot(Uij, Uij) + np.dot(Uij_g, Uij_g)) / 2
            u, _, vh = np.linalg.svd(U)
            U = np.dot(u, vh)
            H[i, j] = U[0, 0] - 1j * U[1, 0]
            c += 1

    H = H + np.conj(H).T
    H = H + np.eye(n_images)  # put 1 on diagonal since : exp^(i*0) = 1

    eig_vals, eig_vecs = scipy.linalg.eigh(H, eigvals=(n_images - 5, n_images - 1))

    print("H top 5 eigenvalues are " + str(eig_vals))
    evect1 = eig_vecs[:, -1]

    R_thetas = np.zeros((n_images, 3, 3))
    for i in np.arange(n_images):
        zi = evect1[i]
        zi = zi / np.abs(zi)  # rescale so it lies on unit circle
        c = np.real(zi ** (1 / 2))  # we have twice the required angle
        s = np.imag(zi ** (1 / 2))
        R_thetas[i] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    Ris = np.zeros((n_images, 3, 3))
    for i, (R_theta, Ri_tilde) in enumerate(zip(R_thetas, Ri_tildes)):
        Ris[i] = np.dot(R_theta, Ri_tilde)
    return Ris, R_thetas


def estimate_rots_from_third_rows(npf, vis):
    """
    estimate the rotation matrices given the tird row of each rotation matrix and the Fourier-transformed images
    :param npf: an array of size m-by-n_theta-by-n_r consisting of the m images in fourier space
    :param vis: an mx3 array where the i-th row is the estimate for the third row of the i-th (unknown) rotation matrix
    :return: an mx3x3 array, holding the m estimated 3x3 rotation matrices
    """
    print('estimating in-plane rotation angles')
    assert len(vis) == len(npf)
    n_symm = AbinitioSymmConfig.n_symm
    n_theta = AbinitioSymmConfig.n_theta
    inplane_rot_res_deg = AbinitioSymmConfig.inplane_rot_res_deg
    max_shift_1d = np.ceil(2 * np.sqrt(2) * AbinitioSymmConfig.max_shift)
    shift_step = AbinitioSymmConfig.shift_step
    n_r = AbinitioSymmConfig.n_r
    assert n_symm > 2
    n_images = len(vis)

    #  Step 1: construct all rotation matrices Ri_tildes whose third row is equal to the corresponding third rows vis
    Ri_tildes = np.array([utils.complete_third_row_to_rot(vi) for vi in vis])

    max_angle = (360//n_symm)*n_symm  # TODO: get rid of the 360 and work instead with radians and resolution number

    theta_ijs = np.arange(0, max_angle, inplane_rot_res_deg)*np.pi/180
    n_theta_ijs = len(theta_ijs)

    # Step 2: construct all in-plane rotation matrices
    cos_theta_ijs = np.array([np.cos(theta_ij) for theta_ij in theta_ijs])
    sin_theta_ijs = np.array([np.sin(theta_ij) for theta_ij in theta_ijs])
    R_theta_ijs = np.array([[[c, -s, 0], [s, c, 0], [0, 0, 1]] for c, s in zip(cos_theta_ijs, sin_theta_ijs)])

    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    max_corrs = np.zeros(m_choose_2)
    max_idx_corrs = np.zeros(m_choose_2)
    counter = 0
    H = np.zeros((n_images, n_images), dtype=complex)

    shift_phases = utils.calc_shift_phases(n_r, max_shift_1d, shift_step)
    n_shifts = len(shift_phases)
    with tqdm(total=n_images) as pbar:
        for i in range(n_images):
            npf_i = npf[i]
            npf_i_shifted = np.array([npf_i * shift_phase for shift_phase in shift_phases])
            # normalize each ray to have norm equal to 1
            npf_i_shifted = np.array([ray / np.linalg.norm(ray) for ray in npf_i_shifted])
            for j in range(i+1, n_images):
                npf_j = npf[j]
                # normalize each ray to have norm equal to 1
                npf_j = np.array([ray / np.linalg.norm(ray) for ray in npf_j])

                Ri_tilde = Ri_tildes[i]
                Rj_tilde = Ri_tildes[j]

                Us = np.array([np.linalg.multi_dot([Ri_tilde.T, R_theta_ij, Rj_tilde]) for R_theta_ij in R_theta_ijs])
                c1s = np.array([[-U[1, 2],  U[0, 2]] for U in Us])
                c2s = np.array([[ U[2, 1], -U[2, 0]] for U in Us])

                c1s = utils.clAngles2Ind__(c1s, n_theta)
                c2s = utils.clAngles2Ind__(c2s, n_theta)

                corrs = np.array([np.dot(npf_i[c1], np.conj(npf_j[c2]))
                                  for npf_i in npf_i_shifted for c1, c2 in zip(c1s, c2s)])

                assert np.mod(n_theta_ijs, n_symm) == 0
                corrs = corrs.reshape((n_shifts, n_symm, n_theta_ijs//n_symm))

                if n_shifts > 1:  # each line may have a different shift so max the shifts out
                    corrs = np.max(np.real(corrs), axis=0)
                else:
                    corrs = np.squeeze(corrs, axis=0)

                # take the mean score over all groups of n pairs of lines, and find the group that attains the maximum
                corrs = np.mean(np.real(corrs), axis=0)
                max_idx_corr = np.argmax(corrs)
                max_corr = corrs[max_idx_corr]

                max_corrs[counter] = max_corr  # this is only for stats
                max_idx_corrs[counter] = max_idx_corr  # this is only for stats

                theta_ij = inplane_rot_res_deg*max_idx_corr*np.pi/180

                H[i, j] = np.cos(n_symm * theta_ij) - 1j * np.sin(n_symm * theta_ij)
                counter += 1
            # update the bar
            if np.mod(i, 10) == 0:
                pbar.update(10)

    H = H + np.conj(H).T
    H = H + np.eye(n_images)  # put 1 on diagonal since : exp^(i*0) = 1

    eig_vals, eig_vecs = scipy.linalg.eigh(H, eigvals=(n_images - 5, n_images - 1))

    print("H top 5 eigenvalues are " + str(eig_vals))
    evect1 = eig_vecs[:, -1]

    R_thetas = np.zeros((n_images, 3, 3))
    for i in np.arange(n_images):
        zi = evect1[i]
        zi = zi / np.abs(zi)  # rescale so it lies on unit circle
        c = np.real(zi ** (1 / n_symm))
        s = np.imag(zi ** (1 / n_symm))
        # TODO: fix the bug in matlab (both c2 and cn)! the angles retrieved from the eigenvector
        #  are known up to an arbitrary angle.
        #  Thus, we need the eigenvector to be exp(i*(theta_i+theta)) and not exp(i*(-theta_i+theta))
        R_thetas[i] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    Ris = np.zeros((n_images, 3, 3))
    for i, (R_theta, Ri_tilde) in enumerate(zip(R_thetas, Ri_tildes)):
        Ris[i] = np.dot(R_theta, Ri_tilde)
    return Ris, R_thetas


def estimate_inplane_rots_angles_gt(n_symm, vis, rots_gt):

    assert len(rots_gt) == len(vis)
    Ri_tildes = [utils.complete_third_row_to_rot(vi) for vi in vis]
    R_thetas = [np.dot(rot_gt, Ri_tilde.T) for rot_gt, Ri_tilde in zip(rots_gt, Ri_tildes)]
    thetas = [np.arctan2(R_theta[1][0], R_theta[0][0]) for R_theta in R_thetas]
    # we can (and need) to merely estimate the in-plane rotation angles modulo 2pi/n
    thetas_in = np.array([np.mod(theta, 2 * np.pi / n_symm) for theta in thetas])
    # note entry (i,j) corresponds to exp^(i(-theta_i+theta_i)). Therefore to
    # construct the Hermitian matrix, entry (j,i) is the **conjugate** of entry (i,j)
    n_images = thetas_in.size
    H = np.zeros((n_images, n_images), dtype=complex)
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            theta_ij = np.mod(-thetas_in[i] + thetas_in[j], 2 * np.pi / n_symm)
            H[i, j] = np.cos(n_symm * theta_ij) - 1j * np.sin(n_symm * theta_ij)

    H = H + np.conj(H).T
    H = H + np.eye(n_images)  # put 1 on diagonal since : exp^(i*0) = 1

    eig_vals, eig_vecs = scipy.linalg.eigh(H, eigvals=(n_images - 5, n_images - 1))

    print("H top 5 eigenvalues are " + str(eig_vals))
    # H is rank-1 whose eigenvector is given by (exp^{-2pi theta_1},...,exp^{-2pi theta_n})
    evect1 = eig_vecs[:, -1]

    in_plane_rots = np.zeros((n_images, 3, 3))
    for i in np.arange(n_images):
        zi = evect1[i]
        zi = zi / np.abs(zi)  # rescale so it lies on unit circle
        c = np.real(zi ** (1 / n_symm))
        s = np.imag(zi ** (1 / n_symm))
        # TODO: fix the bug in matlab! the angles retrieved from the eigenvector are known up to an arbitrary angle.
        #  Thus, we need the eigenvector to be exp(i*(theta_i+theta)) and not exp(i*(-theta_i+theta))
        in_plane_rots[i] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    Ris = np.zeros((n_images, 3, 3))
    for i in np.arange(n_images):
        Ris[i] = np.dot(in_plane_rots[i], Ri_tildes[i])

    return Ris, R_thetas


if __name__ == "__main__":
    n_images = 100
    n_symm = 4
    # test_estimate_inplane_rot_angles(n_symm, n_images)
