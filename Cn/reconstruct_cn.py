import numpy as np
import os
import aspire.abinitio as abinitio
from aspire.common.logger import logger
from Cn.config_symm import AbinitioSymmConfig
import Cn.utils as utils


def reconstruct_cn(n_symm, projs, rotations):
    """
    Reconstructs a volume given projections and corresponding rotation matrices assuming the underlying is Cn.
    Essentially, this is the same reconstruction code as for C1,
    only that each rotation and each projection is duplicated n times
    :param projs: an array of m projection images
    :param rotations: an array of mx3x3 of rotation matrices where the i-th 3x3
    matrix is in the orbit of the rotations that corresponds to the i-th input projection image
    :return:
    """
    output_folder = AbinitioSymmConfig.output_folder
    vol = do_reconstruct_cn(n_symm, projs, rotations)
    vol = make_vol_cn(vol)
    # Since WriteMRC is not implemented yet in python, we save the output vol as an npy file.
    np.save(os.path.join(output_folder, "vol.npy"), vol)
    # call a utility method to transform npy file to mat files
    utils.npy_to_matlab(os.path.join(output_folder, "python_output"), output_folder)
    # TODO: Use the following matlab script to write the file in an mrc format so that it can be rendered using chimera
    #  path_folder = '/a/home/cc/math/gabipragier/ASPIRE-Python/Cn/output/';
    #  python_output = load(fullfile(path_folder, 'python_output.mat'));
    #  vol = python_output.vol;
    #  recon_mrc_fname = fullfile(path_folder, 'python_reconst.mrc');
    #  WriteMRC(vol, 1, recon_mrc_fname);


def do_reconstruct_cn(n_symm, projs, rotations):
    n_images = len(projs)
    assert len(rotations) == n_images
    n_r = AbinitioSymmConfig.n_r
    n_theta = AbinitioSymmConfig.n_theta

    shift_step = AbinitioSymmConfig.shift_step
    max_shift_1d = np.ceil(2 * np.sqrt(2) * AbinitioSymmConfig.max_shift)
    pf, _ = abinitio.cryo_pft_pystyle(projs, n_r, n_theta)

    projs_cn = projs
    for i in range(n_symm - 1):
        projs_cn = np.concatenate((projs_cn, projs), axis=0)

    pf_cn = pf
    for i in range(n_symm-1):
        pf_cn = np.concatenate((pf_cn, pf), axis=0)

    rotations_cn = np.zeros((n_images*n_symm, 3, 3))
    g = utils.generate_g(n_symm)
    for i, rot in enumerate(rotations):
        for s in range(n_symm):
            rotations_cn[s*n_images + i] = np.dot(np.linalg.matrix_power(g, s), rot)

    projs_cn = np.transpose(projs_cn, axes=(1, 2, 0))
    pf_cn = np.transpose(pf_cn, axes=(2, 1, 0))
    rotations_cn = np.transpose(rotations_cn, axes=(1, 2, 0))
    est_shifts, _ = abinitio.cryo_estimate_shifts(pf_cn, rotations_cn, max_shift_1d, shift_step)

    # reconstruct down-sampled volume with no CTF correction
    n = projs.shape[1]

    params = abinitio.fill_struct()
    params.rot_matrices = rotations_cn
    params.ctf = np.ones((n, n))
    params.ctf_idx = np.array([True] * projs_cn.shape[2])
    params.shifts = est_shifts
    params.ampl = np.ones(projs_cn.shape[2])

    basis = abinitio.DiracBasis((projs_cn.shape[0], projs_cn.shape[0], projs_cn.shape[0]))
    v1, _ = abinitio.cryo_estimate_mean(projs_cn, params, basis)
    ii1 = np.linalg.norm(v1.imag) / np.linalg.norm(v1)
    logger.info(f'Relative norm of imaginary components = {ii1}')
    v1 = v1.real
    return v1


def make_vol_cn(vol):
    #  TODO: implement along the lines of matlab function "make_vol_cn".
    #   Specifically nee to implement fastrotate3z in python for this to work
    return vol

