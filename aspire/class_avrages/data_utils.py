from scipy.io import loadmat
import os


def mat_to_npy(file_name, dir_path=None):
    if dir_path is None:
        return loadmat(file_name + '.mat')[file_name]
    else:
        file_path = os.path.join(dir_path, file_name)
        return loadmat(file_path + '.mat')[file_name]


def mat_to_npy_vec(file_name, dir_path=None):
    a = mat_to_npy(file_name, dir_path)
    return a.reshape(a.shape[0] * a.shape[1])
