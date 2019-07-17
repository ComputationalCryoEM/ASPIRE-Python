import os
import mrcfile
import numpy as np

from scipy.io import loadmat, savemat

from aspire.aspire.common.exceptions import WrongInput, UnknownFormat, DimensionsIncompatible
from aspire.aspire.common.logger import logger
from aspire.aspire.utils.helpers import accepts


@accepts(np.ndarray)
def validate_3d_array(stack):
    """ Validate stack is of 3D. Otherwise throw an error. """
    if stack.ndim != 3:
        raise DimensionsIncompatible("stack isn't a 3D array!")


@accepts(np.ndarray)
def fortran_to_c(stack):
    """ Convert Fortran-contiguous array to C-contiguous array. """
    if stack.flags.f_contiguous:
        logger.debug(f"loading F-contiguous stack")
    return stack.T if stack.flags.f_contiguous else stack


@accepts(np.ndarray)
def c_to_fortran(stack):
    """ Convert Fortran-indexed 3D array to C indexed array. """
    return stack.T if not stack.flags.f_contiguous else stack


@accepts(str)
def read_mrc_like_matlab(mrc_file):
    """ Read MRC stack and make sure stack is 'Fortran indexed' before returning it. """
    mrc_stack = mrcfile.open(mrc_file).data
    fortran_indexed_stack = c_to_fortran(mrc_stack)
    return fortran_indexed_stack


@accepts(str, np.ndarray)
def write_mrc_like_matlab(mrc_filename, stack):
    """ Make sure input stack is 'Fortran indexed' and save it to file. """
    fortran_indexed_stack = c_to_fortran(stack)
    with mrcfile.new(mrc_filename) as fh:
        fh.set_data(fortran_indexed_stack)


def mat_to_npy(file_name, dir_path=None):
    if dir_path is None:
        return loadmat(file_name + '.mat')[file_name]
    else:
        file_path = os.path.join(dir_path, file_name)
        return loadmat(file_path + '.mat')[file_name]


def mat_to_npy_vec(file_name, dir_path=None):
    a = mat_to_npy(file_name, dir_path)
    return a.reshape(a.shape[0] * a.shape[1])


def npy_to_mat(file_name, var_name, var):
    savemat(file_name, {var_name: var})


def load_stack_from_file(filepath, c_contiguous=True, return_format=None):
    """ Load projection-stack from file. Try different formats.
        Supported formats are MRC/MRCS/MAT/NPY. """

    # try MRC/MRCS
    try:
        stack = mrcfile.open(filepath).data
        if c_contiguous:
            stack = fortran_to_c(stack)
        if return_format:
            return stack, 'mrc'
        return stack
    except ValueError:
        pass

    # try NPY format
    try:
        stack = np.load(filepath)
        if not isinstance(stack, np.ndarray):
            raise WrongInput(f"File {filepath} doesn't contain a stack!")

        if c_contiguous:
            stack = fortran_to_c(stack)

        if return_format:
            return stack, 'npy'
        return stack

    except OSError:
        pass

    # try MAT format
    try:
        content = loadmat(filepath)
        # filter actual data
        data = [content[key] for key in content.keys() if key == key.strip('_')]
        if len(data) == 1 and hasattr(data[0], 'shape'):
            stack = data[0]

            if c_contiguous:
                stack = fortran_to_c(stack)

            if return_format:
                return stack, 'mat'
            return stack
        raise WrongInput(f"MAT file {filepath} doesn't contain a stack!")
    except ValueError:
        pass

    raise UnknownFormat(f"Couldn't determine stack format! {filepath}!")


def validate_square_projections(stack):
    if stack.ndim not in [2, 3]:
        raise WrongInput(f"Input isn't a projection-stack! {stack.shape}")

    x, y = (0, 1) if stack.ndim == 2 or np.isfortran(stack) else (1, 2)
    if stack.shape[x] != stack.shape[y]:
        raise DimensionsIncompatible("projections must me square!"
                                     f" (x={stack.shape[x]}, y={stack.shape[y]})")


def fctr(n):
    """ Center of an FFT-shifted image. We use this center
        coordinate for all rotations and centering operations. """

    if isinstance(n, int):
        n = np.array([n, n])

    return np.ceil((n + 1) / 2)
