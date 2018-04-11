import numpy as np
import matplotlib.pyplot as plt
import time

import em_classavg.data_utils as data_utils
from em_classavg.image_denoising.image_denoising.ConverterModel.Converter import Converter


class EM:
    def __init__(self, images, mean_bg_ims, sd_bg_ims): # , est_scale=1, n_scale_ticks=10, ang_jump=1, max_shift=0, shift_jump=1):
        # self.est_scale = est_scale
        # self.n_scale_ticks = n_scale_ticks
        # self.ang_jump = ang_jump
        # self.max_shift = max_shift
        # self.shift_jump = shift_jump

        self.im_size = np.shape(images)[-1]
        if np.ndim(images) == 3:
            self.n_images = len(images)
        else:
            self.n_images = 1

        self.mean_bg_ims = mean_bg_ims
        self.sd_bg_ims = sd_bg_ims

        self.em_params = dict()
        self.init_params(images)

        self.trunc_param = data_utils.mat_to_npy_vec('T')[0]
        self.beta = data_utils.mat_to_npy_vec('beta')[0]

        self.converter = Converter(self.im_size, self.trunc_param, self.beta)
        self.converter.init_direct('full')

        self.c_ims = self.converter.direct_forward(images)
        self.const_terms = self.pre_compute_const_terms()

        self.phases = np.exp(-1j * 2 * np.pi / 360 *
                        np.outer(self.em_params['thetas'], self.converter.direct_get_angular_frequency()))
        #  the expansion coefficients of each image for each possible rotation
        self.c_ims_rot = self.c_ims[:, np.newaxis, :] * self.phases[np.newaxis, :]

    def init_params(self, images):

        ang_jump = data_utils.mat_to_npy_vec('ang_jump')[0]
        self.em_params['thetas'] = np.arange(1, 361, ang_jump)
        # data_utils.mat_to_npy_vec('em_params_thetas')  #
        self.em_params['max_shift'] = data_utils.mat_to_npy_vec('max_shift')[0]  # max_shift  #

        self.em_params['shift_jump'] = data_utils.mat_to_npy_vec('shift_jump')[0]  # shift_jump  #
        self.em_params['shifts'] = np.arange(-1 * self.em_params['max_shift'],
                                             self.em_params['max_shift'] + 1, self.em_params['shift_jump'])

        self.em_params['n_scales'] = data_utils.mat_to_npy_vec('n_scales')[0]
        snr_est = EM.est_snr(images)
        est_scale = np.sqrt(snr_est * np.mean(self.sd_bg_ims) ** 2)
        self.em_params['scales'] = np.linspace(0.8 * est_scale, 1.2 * est_scale, self.em_params['n_scales'])
        # em_params_scales = data_utils.mat_to_npy_vec('em_params_scales')

    def e_step(self, c_avg):

        print('e-step')
        n_scales = len(self.em_params['scales'])
        n_rots = len(self.em_params['thetas'])
        n_shifts_2d = len(self.em_params['shifts'])**2

        n_shifts_1d = len(self.em_params['shifts'])

        posteriors = np.zeros((self.n_images, n_scales, n_rots, n_shifts_2d))

        # compute the terms that do not depend on the shifts
        ann_const = (np.linalg.norm(c_avg) * np.outer(1 / self.sd_bg_ims, self.em_params['scales']))**2
        cross_cnn_ann = np.outer(self.mean_bg_ims / (self.sd_bg_ims**2), self.em_params['scales']) * \
                        2 * np.real(np.vdot(c_avg, self.const_terms['c_all_ones_im']))

        ann_const_cross_cnn_anns = ann_const + cross_cnn_ann
        const_elms = ann_const_cross_cnn_anns + (self.const_terms['anni'] + self.const_terms['cnn'])[:, np.newaxis]

        n_prolates = self.converter.direct_get_num_samples()
        for shift_x in self.em_params['shifts']:
            for shift_y in self.em_params['shifts']:

                if shift_y < shift_x:
                    continue

                if shift_x == 0 and shift_y == 0:
                    A_shift = np.eye(n_prolates)
                else:
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
                    posteriors[i, ..., inds[0]] = cross_anni_ann - (const_elms[i][:, np.newaxis] + cross_anni_cnn)

                    if shift_y != shift_x:
                        cross_anni_cnn_minus = self.mean_bg_ims[i] / self.sd_bg_ims[i] * \
                                         2 * np.real(tmp1_inv_shift.dot(np.transpose(self.c_ims_rot[i])))

                        cross_anni_ann_minus = self.em_params['scales'][:, np.newaxis] / self.sd_bg_ims[i] * \
                                         2 * np.real(tmp2_inv_shift.dot(np.transpose(self.c_ims_rot[i])))

                        # write down the log likelihood
                        #  TODO: avoid elipsis by shifting shift indx to the beginning
                        posteriors[i, ..., inds[1]] = cross_anni_ann_minus - \
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
        c = posteriors * self.em_params['scales'][:, np.newaxis, np.newaxis] / \
            self.sd_bg_ims[:, np.newaxis, np.newaxis, np.newaxis]

        c = np.sum(c)

        n_prolates = self.converter.direct_get_num_samples()

        W_shifts_marg = np.zeros((n_images, n_prolates)).astype('complex')

        c_avg = np.zeros(n_prolates).astype('complex')

        for shift_x in self.em_params['shifts']:
            for shift_y in self.em_params['shifts']:

                if shift_y < shift_x:
                    continue

                shifts = (np.array([[shift_y, -shift_y], [shift_x, -shift_x]]) + self.em_params['max_shift']) / \
                         self.em_params['shift_jump']
                inds = np.ravel_multi_index(shifts.astype(shift_y), (n_shifts_1d, n_shifts_1d))

                if shift_x == 0 and shift_y == 0:
                    A_shift = np.eye(n_prolates)
                else:
                    A_shift = self.calc_A_shift(shift_x, shift_y)

                A_inv_shift = np.conj(np.transpose(A_shift))

                W = np.zeros((n_images, self.converter.direct_get_num_samples())).astype('complex')

                for i in np.arange(n_images):
                    W[i] = np.sum(np.dot(posteriors[i, ..., inds[0]], self.phases), axis=0)

                c_avg += np.sum(A_shift.dot(np.transpose(W * self.c_ims)), axis=1)

                W_shifts_marg += W

                if shift_y != shift_x:

                    W_minus = np.zeros((n_images, self.converter.direct_get_num_samples())).astype('complex')

                    for i in np.arange(n_images):
                        W_minus[i] = np.sum(np.dot(posteriors[i, ..., inds[1]], self.phases), axis=0)

                    c_avg += np.sum(A_inv_shift.dot(np.transpose(W_minus * self.c_ims)), axis=1)

                    W_shifts_marg += W_minus

        #  update the coeffs using with respect to the additive term
        c_avg += np.sum(np.transpose(W_shifts_marg * self.const_terms['c_additive_term']), axis=1)

        c_avg = c_avg/c
        return c_avg

    def calc_A_shift(self, shift_x, shift_y):

        psis = self.converter.direct_get_samples_as_images()

        n_psis = len(psis)
        A_shift = np.zeros((n_psis, n_psis)).astype('complex')
        non_neg_freqs = self.converter.direct_get_non_neg_freq_inds()
        psis_non_neg_shifted = np.roll(np.roll(psis[non_neg_freqs], shift_y, axis=1), shift_x, axis=2)

        # mask the shifted psis
        psis_non_neg_shifted = self.converter.direct_mask_points_inside_the_circle(psis_non_neg_shifted)

        # we need the conjugation by design
        A_shift[:, non_neg_freqs] = np.tensordot(np.conj(psis), psis_non_neg_shifted, axes=([1, 2], [1, 2]))

        zero_freq_inds = self.converter.direct_get_zero_freq_inds()
        pos_freq_inds = self.converter.direct_get_pos_freq_inds()
        neg_freq_inds = self.converter.direct_get_neg_freq_inds()

        A_shift[zero_freq_inds[:, np.newaxis], neg_freq_inds] = np.conj(A_shift[zero_freq_inds[:, np.newaxis], pos_freq_inds])
        A_shift[pos_freq_inds[:, np.newaxis], neg_freq_inds]  = np.conj(A_shift[neg_freq_inds[:, np.newaxis], pos_freq_inds])
        A_shift[neg_freq_inds[:, np.newaxis], neg_freq_inds]  = np.conj(A_shift[pos_freq_inds[:, np.newaxis], pos_freq_inds])

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
    images = data_utils.mat_to_npy('images')
    images = np.transpose(images, axes=(2, 0, 1))  # move to python convention
    images, mean_bg_ims, sd_bg_ims = data_utils.normalize_background(images)

    em = EM(images, mean_bg_ims, sd_bg_ims)

    init_avg_image = data_utils.mat_to_npy('init_avg_image')
    c_avg = em.converter.direct_forward(init_avg_image)

    n_iters = 5  # data_utils.mat_to_npy_vec('nIters')[0]
    log_lik = np.zeros((n_iters, em.n_images))

    im_avg_est_prev = init_avg_image
    for it in range(n_iters):
        t = time.time()
        posteriors, log_lik[it] = em.e_step(c_avg)
        print('it %d: log likelihood=%.2f' % (it + 1, np.sum(log_lik[it])))
        print('took %.2f secs' % (time.time() - t))

        t = time.time()
        c_avg = em.m_step(posteriors)
        print('took %.2f secs' % (time.time() - t))

        im_avg_est = em.converter.direct_backward(c_avg)[0]
        EM.plot_images(init_avg_image, im_avg_est_prev, im_avg_est)

        im_avg_est_prev = im_avg_est


if __name__ == "__main__":
    main()
