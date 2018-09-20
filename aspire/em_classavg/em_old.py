import numpy as np
import matplotlib.pyplot as plt
import time
import skcuda.linalg as linalg

import aspire.em_classavg.data_utils as data_utils
from aspire.em_classavg.image_denoising.image_denoising.ConverterModel.Converter import Converter


class EM:
    def __init__(self, images, trunc_param=10, beta=0.5, ang_jump=1,
                 max_shift=5, shift_jump=1, n_scales=10, is_remove_outliers=True, outliers_precent_removal=5):

        self.trunc_param = trunc_param
        self.beta = beta
        self.ang_jump = ang_jump
        self.is_remove_outliers = is_remove_outliers
        self.outliers_precent_removal = outliers_precent_removal

        self.em_params = dict()
        self.em_params['n_scales'] = n_scales
        self.em_params['max_shift'] = max_shift
        self.em_params['shift_jump'] = shift_jump
        self.em_params['thetas'] = np.arange(1, 361, self.ang_jump)
        self.em_params['shifts'] = np.arange(-1 * self.em_params['max_shift'],
                                             self.em_params['max_shift'] + 1, self.em_params['shift_jump'])

        self.im_size = np.shape(images)[-1]
        if np.ndim(images) == 3:
            self.n_images = len(images)
        else:
            self.n_images = 1

        images, self.mean_bg_ims, self.sd_bg_ims = data_utils.normalize_background(images)

        snr_est = EM.est_snr(images)
        est_scale = np.sqrt(snr_est * np.mean(self.sd_bg_ims) ** 2)
        self.em_params['scales'] = np.linspace(0.8 * est_scale, 1.2 * est_scale, self.em_params['n_scales'])

        self.converter = Converter(self.im_size, self.trunc_param, self.beta)
        self.converter.init_direct('full')

        self.c_ims = self.converter.direct_forward(images)
        self.const_terms = self.pre_compute_const_terms()

        self.phases = np.exp(-1j * 2 * np.pi / 360 *
                             np.outer(self.em_params['thetas'], self.converter.get_angular_frequency()))
        #  the expansion coefficients of each image for each possible rotation
        self.c_ims_rot = self.c_ims[:, np.newaxis, :] * self.phases[np.newaxis, :]

    def e_step(self, c_avg):

        print('e-step')
        n_scales = len(self.em_params['scales'])
        n_rots = len(self.em_params['thetas'])
        n_shifts_2d = len(self.em_params['shifts'])**2

        n_shifts_1d = len(self.em_params['shifts'])

        posteriors = np.zeros((self.n_images, n_shifts_2d, n_scales, n_rots))
        # posteriors = np.zeros((self.n_images, n_scales, n_rots, n_shifts_2d))

        # compute the terms that do not depend on the shifts
        ann_const = (np.linalg.norm(c_avg) * np.outer(1 / self.sd_bg_ims, self.em_params['scales']))**2
        cross_cnn_ann = np.outer(self.mean_bg_ims / (self.sd_bg_ims**2), self.em_params['scales']) * \
                        2 * np.real(np.vdot(c_avg, self.const_terms['c_all_ones_im']))

        ann_const_cross_cnn_anns = ann_const + cross_cnn_ann
        const_elms = ann_const_cross_cnn_anns + (self.const_terms['anni'] + self.const_terms['cnn'])[:, np.newaxis]

        for shift_x in self.em_params['shifts']:
            for shift_y in self.em_params['shifts']:

                if shift_y < shift_x:
                    continue

                A_shift = self.calc_A_shift(shift_x, shift_y)
                tmp1_shift = np.conj(self.const_terms['c_all_ones_im']).dot(A_shift)
                tmp2_shift = np.conj(c_avg).dot(A_shift)

                A_inv_shift = np.conj(np.transpose(A_shift))
                tmp1_inv_shift = np.conj(self.const_terms['c_all_ones_im']).dot(A_inv_shift)
                tmp2_inv_shift = np.conj(c_avg).dot(A_inv_shift)

                shifts = (np.array([[shift_y, -shift_y], [shift_x, -shift_x]]) + self.em_params['max_shift']) / \
                         self.em_params['shift_jump']
                inds = np.ravel_multi_index(shifts.astype(shift_y), (n_shifts_1d, n_shifts_1d))

                for i in np.arange(self.n_images):

                    # calculate the two cross terms
                    cross_anni_cnn = self.mean_bg_ims[i] / self.sd_bg_ims[i] * \
                                     2 * np.real(tmp1_shift.dot(np.transpose(self.c_ims_rot[i])))

                    cross_anni_ann = self.em_params['scales'][:, np.newaxis] / self.sd_bg_ims[i] * \
                                     2 * np.real(tmp2_shift.dot(np.transpose(self.c_ims_rot[i])))

                    # write down the log likelihood
                    posteriors[i, inds[0]] = cross_anni_ann - (const_elms[i][:, np.newaxis] + cross_anni_cnn)

                    if shift_y != shift_x:
                        cross_anni_cnn_minus = self.mean_bg_ims[i] / self.sd_bg_ims[i] * \
                                         2 * np.real(tmp1_inv_shift.dot(np.transpose(self.c_ims_rot[i])))

                        cross_anni_ann_minus = self.em_params['scales'][:, np.newaxis] / self.sd_bg_ims[i] * \
                                         2 * np.real(tmp2_inv_shift.dot(np.transpose(self.c_ims_rot[i])))

                        # write down the log likelihood
                        #  TODO: avoid elipsis by shifting shift indx to the beginning
                        posteriors[i, inds[1]] = cross_anni_ann_minus - \
                                                      (const_elms[i][:, np.newaxis] + cross_anni_cnn_minus)

        log_lik_per_image = np.zeros(self.n_images)
        for i in np.arange(self.n_images):

            omega_i = posteriors[i]
            max_omega = np.max(omega_i)

            omega_i = np.exp(omega_i - max_omega)

            log_lik_per_image[i] = max_omega + np.log(np.sum(omega_i))

            posteriors[i] = omega_i / np.sum(omega_i)

        return posteriors, log_lik_per_image

    def m_step(self, posteriors):

        print('m-step')
        n_images = self.n_images
        n_shifts_1d = len(self.em_params['shifts'])

        n_prolates = self.converter.get_num_prolates()

        W_shifts_marg = np.zeros((n_images, n_prolates)).astype('complex')

        c_avg = np.zeros(n_prolates).astype('complex')

        for shift_x in self.em_params['shifts']:
            for shift_y in self.em_params['shifts']:

                if shift_y < shift_x:
                    continue

                shifts = (np.array([[shift_y, -shift_y], [shift_x, -shift_x]]) + self.em_params['max_shift']) / \
                         self.em_params['shift_jump']
                inds = np.ravel_multi_index(shifts.astype(shift_y), (n_shifts_1d, n_shifts_1d))

                A_shift = self.calc_A_shift(shift_x, shift_y)
                A_inv_shift = np.conj(np.transpose(A_shift))

                non_neg_freqs = self.converter.get_non_neg_freq_inds()
                A_shift = A_shift[non_neg_freqs]
                A_inv_shift = A_inv_shift[non_neg_freqs]

                W = np.zeros((n_images, self.converter.get_num_prolates())).astype('complex')

                for i in np.arange(n_images):
                    W[i] = np.sum(np.dot(posteriors[i, inds[0]], self.phases), axis=0)

                c_avg[non_neg_freqs] += np.sum(A_shift.dot(np.transpose(W * self.c_ims)), axis=1)

                W_shifts_marg += W

                if shift_y != shift_x:

                    W_minus = np.zeros((n_images, self.converter.get_num_prolates())).astype('complex')

                    for i in np.arange(n_images):
                        W_minus[i] = np.sum(np.dot(posteriors[i, inds[1]], self.phases), axis=0)

                    c_avg[non_neg_freqs] += np.sum(A_inv_shift.dot(np.transpose(W_minus * self.c_ims)), axis=1)

                    W_shifts_marg += W_minus

        #  update the coeffs using with respect to the additive term
        c_avg[non_neg_freqs] += np.sum(np.transpose(W_shifts_marg * self.const_terms['c_additive_term']), axis=1)[non_neg_freqs]

        c_avg[self.converter.get_neg_freq_inds()] = np.conj(c_avg[self.converter.get_pos_freq_inds()])

        c = posteriors * self.em_params['scales'][:, np.newaxis] / \
            self.sd_bg_ims[:, np.newaxis, np.newaxis, np.newaxis]
        c = np.sum(c)
        c_avg = c_avg/c

        return c_avg

    def calc_A_shift(self, shift_x, shift_y):

        psis = self.converter.get_prolates_as_images()
        n_psis = len(psis)

        if shift_x == 0 and shift_y == 0:
            return np.eye(n_psis)

        A_shift = np.zeros((n_psis, n_psis)).astype('complex')
        non_neg_freqs = self.converter.get_non_neg_freq_inds()
        psis_non_neg_shifted = np.roll(np.roll(psis[non_neg_freqs], shift_y, axis=1), shift_x, axis=2)

        # mask the shifted psis
        psis_non_neg_shifted = self.converter.mask_points_inside_the_circle(psis_non_neg_shifted)

        # we need the conjugation by design
        A_shift[:, non_neg_freqs] = np.tensordot(np.conj(psis), psis_non_neg_shifted, axes=([1, 2], [1, 2]))

        zero_freq_inds = self.converter.get_zero_freq_inds()
        pos_freq_inds = self.converter.get_pos_freq_inds()
        neg_freq_inds = self.converter.get_neg_freq_inds()

        A_shift[zero_freq_inds, neg_freq_inds] = np.conj(A_shift[zero_freq_inds, pos_freq_inds])
        A_shift[pos_freq_inds, neg_freq_inds]  = np.conj(A_shift[neg_freq_inds, pos_freq_inds])
        A_shift[neg_freq_inds, neg_freq_inds]  = np.conj(A_shift[pos_freq_inds, pos_freq_inds])

        return A_shift

    def pre_compute_const_terms(self):

        const_terms = dict()
        im_size = self.im_size
        # we need the all-ones-image in order for the additive term due to normalization
        const_terms['c_all_ones_im'] = self.converter.direct_forward(np.ones((im_size, im_size)))

        const_terms['anni'] = np.linalg.norm(self.c_ims, axis=1)**2
        const_terms['cnn'] = (self.mean_bg_ims / self.sd_bg_ims * np.linalg.norm(const_terms['c_all_ones_im']))**2
        const_terms['c_additive_term'] = np.outer(self.mean_bg_ims / self.sd_bg_ims, const_terms['c_all_ones_im'])

        return const_terms

    def compute_opt_latent_vals(self, posteriors):

        n_images = len(posteriors)

        n_shifts_1d = len(self.em_params['shifts'])

        opt_latent = dict()
        opt_latent['rots'] = np.zeros(n_images)
        opt_latent['shifts_x'] = np.zeros(n_images)
        opt_latent['shifts_y'] = np.zeros(n_images)
        opt_latent['scales'] = np.zeros(n_images)
        for i in np.arange(n_images):
            om_i = posteriors[i]

            opt_scale_ind = np.argmax(np.sum(np.sum(om_i, axis=2), axis=1))
            opt_rot_ind = np.argmax(np.sum(np.sum(om_i, axis=2), axis=0))
            opt_shift_ind = np.argmax(np.sum(np.sum(om_i, axis=1), axis=0))

            opt_latent['scales'][i] = self.em_params['scales'][opt_scale_ind]
            opt_latent['rots'][i] = self.em_params['thetas'][opt_rot_ind]

            yy, xx = np.unravel_index(opt_shift_ind, (n_shifts_1d, n_shifts_1d))
            opt_latent['shifts_x'][i] = self.em_params['shifts'][xx]
            opt_latent['shifts_y'][i] = self.em_params['shifts'][yy]

        return opt_latent

    @staticmethod
    def est_snr(images):

        snr = data_utils.estimate_snr(images)[0]
        if snr <= 0:
            snr = 10 ** -4

        return snr

    @staticmethod
    def plot_images(init_avg_image, im_avg_est_prev, im_avg_est):

        # plt.figure(1)

        plt.subplot(131)
        plt.imshow(init_avg_image, cmap='gray')
        plt.subplot(132)
        plt.imshow(np.real(im_avg_est_prev), cmap='gray')
        plt.subplot(133)
        plt.imshow(np.real(im_avg_est), cmap='gray')

        plt.show()


def main():
    linalg.init()  # TODO: where to init this?
    images = data_utils.mat_to_npy('images')
    images = np.transpose(images, axes=(2, 0, 1))  # move to python convention

    is_use_matlab_params = True

    if is_use_matlab_params:
        trunc_param = data_utils.mat_to_npy_vec('T')[0]
        beta = data_utils.mat_to_npy_vec('beta')[0]
        ang_jump = data_utils.mat_to_npy_vec('ang_jump')[0]
        max_shift = data_utils.mat_to_npy_vec('max_shift')[0]  # max_shift

        shift_jump = data_utils.mat_to_npy_vec('shift_jump')[0]  # shift_jump
        n_scales = data_utils.mat_to_npy_vec('n_scales')[0]

        is_remove_outliers = data_utils.mat_to_npy_vec('is_remove_outliers')[0]
        outliers_precent_removal = data_utils.mat_to_npy_vec('outliers_precent_removal')[0]

        em = EM(images, trunc_param, beta, ang_jump, max_shift, shift_jump,
                n_scales, is_remove_outliers, outliers_precent_removal)
    else:
        em = EM(images,max_shift=0)

    init_avg_image = data_utils.mat_to_npy('init_avg_image')

    init_avg_image = data_utils.mask_decorator(init_avg_image, is_stack=True)

    c_avg = em.converter.direct_forward(init_avg_image)

    n_iters = 3  # data_utils.mat_to_npy_vec('nIters')[0]

    print("#images=%d\t#iterations=%d\tangualr-jump=%d,\tmax shift=%d,\tshift-jump=%d,\t#scales=%d" %
          (len(images), n_iters, em.ang_jump, em.em_params['max_shift'],em.em_params['shift_jump'], em.em_params['n_scales']))

    im_avg_est_prev = init_avg_image

    log_lik = dict()
    for round in range(2):
        round_str = str(round)
        log_lik[round_str] = np.zeros((n_iters, em.n_images))
        for it in range(n_iters):
            t = time.time()
            posteriors, log_lik[round_str][it] = em.e_step(c_avg)
            print('it %d: log likelihood=%.2f' % (it + 1, np.sum(log_lik[round_str][it])))
            print('took %.2f secs' % (time.time() - t))

            t = time.time()
            c_avg = em.m_step(posteriors)
            print('took %.2f secs' % (time.time() - t))

            im_avg_est = em.converter.direct_backward(c_avg)[0]
            EM.plot_images(init_avg_image, im_avg_est_prev, im_avg_est)

            im_avg_est_prev = im_avg_est

        if round == 0 and em.is_remove_outliers:  # maximum two rounds
            inds_sorted = np.argsort(log_lik[round_str][-1])
            outlier_ims_inds = inds_sorted[:int(em.outliers_precent_removal / 100 * em.n_images)]

            posteriors = np.delete(posteriors, outlier_ims_inds, axis=0)
            em.c_ims_rot = np.delete(em.c_ims_rot, outlier_ims_inds, axis=0)
            em.c_ims = np.delete(em.c_ims, outlier_ims_inds, axis=0)
            em.mean_bg_ims = np.delete(em.mean_bg_ims, outlier_ims_inds, axis=0)
            em.sd_bg_ims = np.delete(em.sd_bg_ims, outlier_ims_inds, axis=0)
            em.n_images = em.n_images - len(outlier_ims_inds)
            em.const_terms = em.pre_compute_const_terms()
        else:
            break

    # find the mode of each latent variable for each image
    opt_latent = em.compute_opt_latent_vals(posteriors)

    return im_avg_est, log_lik, opt_latent, outlier_ims_inds


if __name__ == "__main__":
    main()
