# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:24:08 2019

@author: Gabi
"""
import math
import numpy as np
import scipy
import Cn.utils as utils


if __name__ == "__main__":
    n_symm = 4
    n_theta = 360
    angle_tol_err_deg = 5
    n_images = 50

    g = generate_g(n_symm)
    rots_gt = utils.generate_rots(n_images)

    clmatrix = find_cl_gt(n_symm, n_theta, rots_gt)
    cl_detection_rate_single_cl(n_symm, clmatrix[3], rots_gt, n_theta, angle_tol_err_deg)

    # test_find_cl_scl_gt(n_symm=4, n_theta=360, n_images=10)
