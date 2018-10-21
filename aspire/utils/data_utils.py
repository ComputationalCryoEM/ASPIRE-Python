import os
import mrcfile
import numpy as np

from scipy.io import loadmat

from aspire.common.exceptions import DimensionsIncompatible


def accepts(*types):
    """ These types should match their respective positional args in the decorated function.

        Example:
        @accepts(int, str)
        def test_int_func(x, y):
            return x == int(y)

    """
    def check_accepts(f):

        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), f"arg {a} does not match type {t}"

            return f(*args, **kwds)

        new_f.func_name = f.__name__

        return new_f
    return check_accepts


@accepts(np.ndarray)
def validate_3d_array(stack):
    """ Validate stack is of 3D. Otherwise throw an error. """
    if stack.ndim != 3:
        raise DimensionsIncompatible("stack isn't a 3D array!")


@accepts(np.ndarray)
def fortran_to_c(stack):
    """ Convert Fortran-indexed 3D array to C indexed array. """

    if stack.ndim == 3 and np.isfortran(stack):
        return stack.swapaxes(0, 2)

    return stack


@accepts(np.ndarray)
def c_to_fortran(stack):
    """ Convert Fortran-indexed 3D array to C indexed array. """

    if stack.ndim == 3 and not np.isfortran(stack):
        return stack.swapaxes(0, 2)

    return stack


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
