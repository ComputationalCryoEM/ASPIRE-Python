import pyfftw.interfaces.scipy_fftpack as pyfft


def replace_arguments(f):
    """
    pyfftw uses the argument 'threads' instead of scipy.fft's 'workers' argument
    Passing in a -ve value will allow both libraries to choose the max. number of threads as per cpu_count()
    This decorator replaces any argument passed in as 'workers' with the argument 'threads' to make it work with
    pyfftw.
    :param f: The function to be decorated
    :return: The decorated function with the 'workers' argument replaced with 'threads'
    """
    def inner(*args, **kwargs):
        kwargs['threads'] = kwargs.pop('workers', None)
        return f(*args, **kwargs)
    return inner


class PyfftwFFT:

    fft = staticmethod(replace_arguments(pyfft.fft))
    ifft = staticmethod(replace_arguments(pyfft.ifft))
    fft2 = staticmethod(replace_arguments(pyfft.fft2))
    ifft2 = staticmethod(replace_arguments(pyfft.ifft2))
    fftn = staticmethod(replace_arguments(pyfft.fftn))
    ifftn = staticmethod(replace_arguments(pyfft.ifftn))

    fftshift = staticmethod(pyfft.fftshift)
    ifftshift = staticmethod(pyfft.ifftshift)
