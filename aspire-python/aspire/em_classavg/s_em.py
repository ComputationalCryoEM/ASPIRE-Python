import numpy as np
import aspire.em_classavg.data_utils as data_utils
import aspire.em_classavg.em as em

is_load_large_images = True

if is_load_large_images:
    images_large = data_utils.mat_to_npy('images_large')
<<<<<<< Updated upstream
    init_avg_image_large = data_utils.mat_to_npy('init_avg_image_large')
=======
    init_avg_image_large = data_utils.mat_to_npy('init_avg_image_large').copy()
>>>>>>> Stashed changes
    images = images_large
    init_avg_image = init_avg_image_large
else:
    images = data_utils.mat_to_npy('images')
    init_avg_image = data_utils.mat_to_npy('init_avg_image')

<<<<<<< Updated upstream
images = np.transpose(images, axes=(2, 0, 1))  # move to python convention
=======
images = np.transpose(images, axes=(2, 0, 1)).copy()  # move to python convention
>>>>>>> Stashed changes

im_avg_est, im_avg_est_orig, log_lik, opt_latent, outlier_ims_inds = em.run(images, init_avg_image)

# is_load_params_from_mat = False
#
# if is_load_params_from_mat:
#     trunc_param, beta, ang_jump, max_shift, shift_jump, n_scales, \
#     is_remove_outliers, outliers_precent_removal = em.load_matlab_params()
#
#     im_avg_est, im_avg_est_orig, log_lik, opt_latent, outlier_ims_inds = \
#         em.run(images, init_avg_image, trunc_param, beta, ang_jump, max_shift, shift_jump,
#                n_scales, is_remove_outliers, outliers_precent_removal)