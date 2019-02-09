class AbinitioSymmConfig:

    is_use_gt = False
    is_load_shifted_projs = True  # only relevant when loading simulated projs
    n_theta = 360
    n_r = 45
    max_shift = 0
    shift_step = 1
    n_symm = None
    angle_tol_err_deg = 5
    inplane_rot_res_deg = 1
    # cache_file_name = 'cn_cache_points200_ntheta360_res1_tmp_gt.pckl'
    # cache_file_name = 'cn_cache_points500_ntheta360_res1_gt.pckl'  # assign None if wish to create a new one
    cache_file_name = 'cn_cache_points1000_ntheta360_res1.pckl'  # assign None if wish to create a new one