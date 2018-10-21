import os
import mrcfile
import numpy as np

from scipy.io import loadmat


def mat_to_npy(file_name, dir_path=None):
    if dir_path is None:
        return loadmat(file_name + '.mat')[file_name]
    else:
        file_path = os.path.join(dir_path, file_name)
        return loadmat(file_path + '.mat')[file_name]


def mat_to_npy_vec(file_name, dir_path=None):
    a = mat_to_npy(file_name, dir_path)
    return a.reshape(a.shape[0] * a.shape[1])


def read_mrc_like_matlab(mrc_file):

    mrc_stack = mrcfile.open(mrc_file).data

    if not np.isfortran(mrc_stack):
        mrc_stack = mrc_stack.swapaxes(0, 2)

    return mrc_stack


def write_mrc_like_matlab(mrc_filename, stack):

    if not np.isfortran(stack):
        stack = stack.swapaxes(0, 2)

    with mrcfile.new(mrc_filename) as fh:
        fh.set_data(stack)
