import numpy as np
import matplotlib.pyplot as plt
import time
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import skcuda.misc as misc
import em_classavg.circ_shift_kernel as circ_shift_kernel
import progressbar

import em_classavg.config as config
import em_classavg.data_utils as data_utils
from em_classavg.image_denoising.image_denoising.ConverterModel.Converter import Converter


class EM:
    def __init__(self, images, init_avg_image, trunc_param=10, beta=0.5, ang_jump=1,
                 max_shift=5, shift_jump=1, n_scales=10, is_remove_outliers=True, outliers_precent_removal=5):

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

        images, self.mean_bg_ims, self.sd_bg_ims = data_utils.normalize_background(images)
        snr_est = EM.est_snr(images)
        est_scale = np.sqrt(snr_est * np.mean(self.sd_bg_ims) ** 2)
        self.em_params['scales'] = np.linspace(0.8 * est_scale, 1.2 * est_scale, self.em_params['n_scales'])

        self.n_images = len(images)
        self.im_size = np.shape(images)[-1]
        self.converter = Converter(self.im_size, trunc_param, beta)
        self.converter.init_direct('full')

        self.c_ims = self.converter.direct_forward(images)

        init_avg_image = data_utils.mask_decorator(init_avg_image, is_stack=True)
        self.c_avg = self.converter.direct_forward(init_avg_image).reshape(-1)

    def ravel_shift_index(self, shift_x, shift_y):
        n_shifts_1d = len(self.em_params['shifts'])
        shift = ([shift_y, shift_x] + self.em_params['max_shift']) / self.em_params['shift_jump']
        return np.ravel_multi_index(shift.astype(int), (n_shifts_1d, n_shifts_1d))

    def calc_e_step_const_elms(self, const_terms):
        ann_const = (np.linalg.norm(self.c_avg) * np.outer(1 / self.sd_bg_ims, self.em_params['scales'])) ** 2
        cross_cnn_ann = np.outer(self.mean_bg_ims / (self.sd_bg_ims ** 2), self.em_params['scales']) * \
                        2 * np.real(np.vdot(self.c_avg, const_terms['c_all_ones_im']))

        ann_const_cross_cnn_anns = ann_const + cross_cnn_ann
        return ann_const_cross_cnn_anns + (const_terms['anni'] + const_terms['cnn'])[:, np.newaxis]

    def e_step(self, phases, const_terms):

        print('e-step')
        n_scales = len(self.em_params['scales'])
        n_rots = len(self.em_params['thetas'])
        n_shifts_2d = len(self.em_params['shifts'])**2

        #  the expansion coefficients of each image for each possible rotation
        # TODO: in principle this could be caclucated once, but it is more readable to have the e-step calc it instead of passing as parameter
        if config.is_use_gpu:
            c_ims_rot = self.c_ims[:, np.newaxis, :] * phases.get()[np.newaxis, :]
        else:
            c_ims_rot = self.c_ims[:, np.newaxis, :] * phases[np.newaxis, :]

        posteriors = np.zeros((self.n_images, n_shifts_2d, n_scales, n_rots))
        # compute the terms that do not depend on the shifts
        const_elms = self.calc_e_step_const_elms(const_terms)

        for shift_x in progressbar.progressbar(self.em_params['shifts']):
            for shift_y in self.em_params['shifts']:

                if shift_y < shift_x:
                    continue

                A_shift = self.calc_A_shift(shift_x, shift_y)
                shift_ind = self.ravel_shift_index(shift_x, shift_y)
                posteriors[:, shift_ind] = self.calc_posteriors_wrt_shift(A_shift, c_ims_rot, const_elms, const_terms)

                if shift_y != shift_x:
                    A_inv_shift = np.conj(np.transpose(A_shift))
                    shift_minus_ind = self.ravel_shift_index(-shift_x, -shift_y)
                    posteriors[:, shift_minus_ind] = self.calc_posteriors_wrt_shift(A_inv_shift, c_ims_rot, const_elms, const_terms)

        log_lik_per_image = np.zeros(self.n_images)
        for i in np.arange(self.n_images):

            omega_i = posteriors[i]
            max_omega = np.max(omega_i)

            omega_i = np.exp(omega_i - max_omega)

            log_lik_per_image[i] = max_omega + np.log(np.sum(omega_i))

            posteriors[i] = omega_i / np.sum(omega_i)

        return posteriors, log_lik_per_image

    def calc_posteriors_wrt_shift(self, A_shift, c_ims_rot, const_elms, const_terms):
        n_scales = len(self.em_params['scales'])
        n_rots = len(self.em_params['thetas'])
        posteriors = np.zeros((self.n_images, n_scales, n_rots))
        tmp1_shift = np.conj(const_terms['c_all_ones_im']).dot(A_shift)
        tmp2_shift = np.conj(self.c_avg).dot(A_shift)
        for i in np.arange(self.n_images):
            # calculate the two cross terms
            cross_anni_cnn = self.mean_bg_ims[i] / self.sd_bg_ims[i] * \
                             2 * np.real(tmp1_shift.dot(np.transpose(c_ims_rot[i])))

            cross_anni_ann = self.em_params['scales'][:, np.newaxis] / self.sd_bg_ims[i] * \
                             2 * np.real(tmp2_shift.dot(np.transpose(c_ims_rot[i])))

            posteriors[i] = cross_anni_ann - (const_elms[i][:, np.newaxis] + cross_anni_cnn)
        return posteriors

    def m_step(self, posteriors, phases, const_terms):

        print('m-step')
        if config.is_use_gpu:
            posteriors = gpuarray.to_gpu(posteriors).astype('complex64')
        W_shifts_marg = np.zeros((self.n_images, self.converter.get_num_prolates()), np.complex64)
        self.c_avg = np.zeros_like(self.c_avg)
        non_neg_freqs = self.converter.get_non_neg_freq_inds()
        for shift_x in progressbar.progressbar(self.em_params['shifts']):
            for shift_y in self.em_params['shifts']:

                if shift_y < shift_x:
                    continue

                A_shift = self.calc_A_shift(shift_x, shift_y)
                A_shift_minus = np.conj(np.transpose(A_shift))

                W = self.marginilize_rots_scales(posteriors, phases, shift_x, shift_y)
                self.c_avg[non_neg_freqs] += np.sum(A_shift[non_neg_freqs].dot(np.transpose(W * self.c_ims)), axis=1)
                W_shifts_marg += W

                if shift_y != shift_x:
                    W_minus = self.marginilize_rots_scales(posteriors, phases, -shift_x, -shift_y)
                    self.c_avg[non_neg_freqs] += np.sum(A_shift_minus[non_neg_freqs].dot(np.transpose(W_minus * self.c_ims)), axis=1)
                    W_shifts_marg += W_minus

        #  update the coeffs using with respect to the additive term
        self.c_avg[non_neg_freqs] += np.sum(np.transpose(W_shifts_marg * const_terms['c_additive_term']), axis=1)[non_neg_freqs]
        # take care of the negative freqs
        self.c_avg[self.converter.get_neg_freq_inds()] = np.conj(self.c_avg[self.converter.get_pos_freq_inds()])
        if config.is_use_gpu:
            posteriors = posteriors.get()
        c = posteriors * self.em_params['scales'][:, np.newaxis] / self.sd_bg_ims[:, np.newaxis, np.newaxis, np.newaxis]
        c = np.sum(c)
        self.c_avg = self.c_avg/c

    def marginilize_rots_scales(self, posteriors, phases, shift_x, shift_y):
        shift_ind = self.ravel_shift_index(shift_x, shift_y)
        W = np.zeros((self.n_images, self.converter.get_num_prolates()), np.complex64)
        if config.is_use_gpu:
            for i in np.arange(self.n_images):
                # TODO: make the assignment work if W is a gpuarray
                W[i] = misc.sum(linalg.dot(posteriors[i, shift_ind], phases), axis=0).get()
        else:
            for i in np.arange(self.n_images):
                W[i] = np.sum(np.dot(posteriors[i, shift_ind], phases), axis=0)
        return W

    def calc_A_shift(self, shift_x, shift_y):

        if config.is_use_gpu:
            return self.__calc_A_shift_gpu(shift_x, shift_y)
        else:
            return self.__calc_A_shift_cpu(shift_x, shift_y)

    def __calc_A_shift_cpu(self, shift_x, shift_y):

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
        pos_freq_inds  = self.converter.get_pos_freq_inds()
        neg_freq_inds  = self.converter.get_neg_freq_inds()

        A_shift[zero_freq_inds, neg_freq_inds] = np.conj(A_shift[zero_freq_inds, pos_freq_inds])
        A_shift[pos_freq_inds, neg_freq_inds]  = np.conj(A_shift[neg_freq_inds, pos_freq_inds])
        A_shift[neg_freq_inds, neg_freq_inds]  = np.conj(A_shift[pos_freq_inds, pos_freq_inds])

        return A_shift

    def __calc_A_shift_gpu(self, shift_x, shift_y):

        psis_gpu = self.converter.get_prolates_as_images()  # TODO: need to assert that returns indeed a gpuarray
        n_psis = len(psis_gpu)

        if shift_x == 0 and shift_y == 0:
            return np.eye(n_psis)

        A_shift = gpuarray.zeros((n_psis, n_psis),'complex64')
        non_neg_freqs = self.converter.get_non_neg_freq_inds()

        psis_gpu_non_neg_freqs = psis_gpu[non_neg_freqs]
        psis_non_neg_shifted = circ_shift_kernel.circ_shift(psis_gpu_non_neg_freqs,shift_x, shift_y)

        psis_non_neg_shifted = self.converter.mask_points_inside_the_circle(psis_non_neg_shifted)

        psis_non_neg_shifted = psis_non_neg_shifted.reshape(len(psis_non_neg_shifted), -1)
        psis_gpu = psis_gpu.reshape(n_psis, -1)
        A_shift[non_neg_freqs] = linalg.dot(psis_non_neg_shifted, psis_gpu, transb='C')

        zero_freq_inds = self.converter.get_zero_freq_inds()
        pos_freq_inds  = self.converter.get_pos_freq_inds()
        neg_freq_inds  = self.converter.get_neg_freq_inds()

        A_shift[neg_freq_inds, zero_freq_inds] = A_shift[pos_freq_inds, zero_freq_inds]
        A_shift[neg_freq_inds, pos_freq_inds] = A_shift[pos_freq_inds, neg_freq_inds]
        A_shift[neg_freq_inds, neg_freq_inds] = A_shift[pos_freq_inds, pos_freq_inds]

        A_shift[neg_freq_inds] = linalg.conj(A_shift[neg_freq_inds])
        # TODO: get rid of the transpose
        # return np.transpose(A_shift).copy()
        return np.transpose(A_shift).get().copy()

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
    linalg.init()
    images = data_utils.mat_to_npy('images')
    images = np.transpose(images, axes=(2, 0, 1))  # move to python convention

    init_avg_image = data_utils.mat_to_npy('init_avg_image')

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

        em = EM(images, init_avg_image, trunc_param, beta, ang_jump, max_shift, shift_jump,
                n_scales, is_remove_outliers, outliers_precent_removal)
    else:
        em = EM(images, init_avg_image, max_shift=3)

    const_terms = em.pre_compute_const_terms()
    phases = np.exp(-1j * 2 * np.pi / 360 *
                         np.outer(em.em_params['thetas'], em.converter.get_angular_frequency()))

    if config.is_use_gpu:
        phases = gpuarray.to_gpu(phases).astype('complex64')

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
            posteriors, log_lik[round_str][it] = em.e_step(phases, const_terms)
            print('it %d: log likelihood=%.2f' % (it + 1, np.sum(log_lik[round_str][it])))
            print('took %.2f secs' % (time.time() - t))

            t = time.time()
            em.m_step(posteriors, phases, const_terms)
            print('took %.2f secs' % (time.time() - t))

            im_avg_est = em.converter.direct_backward(em.c_avg)[0]
            EM.plot_images(init_avg_image, im_avg_est_prev, im_avg_est)

            im_avg_est_prev = im_avg_est

        if round == 0 and em.is_remove_outliers:  # maximum two rounds
            inds_sorted = np.argsort(log_lik[round_str][-1])
            outlier_ims_inds = inds_sorted[:int(em.outliers_precent_removal / 100 * em.n_images)]

            posteriors = np.delete(posteriors, outlier_ims_inds, axis=0)
            em.c_ims = np.delete(em.c_ims, outlier_ims_inds, axis=0)
            em.mean_bg_ims = np.delete(em.mean_bg_ims, outlier_ims_inds, axis=0)
            em.sd_bg_ims = np.delete(em.sd_bg_ims, outlier_ims_inds, axis=0)
            em.n_images = em.n_images - len(outlier_ims_inds)
            const_terms = em.pre_compute_const_terms()
        else:
            break

    # find the mode of each latent variable for each image
    opt_latent = em.compute_opt_latent_vals(posteriors)

    return im_avg_est, log_lik, opt_latent, outlier_ims_inds


if __name__ == "__main__":
    main()
