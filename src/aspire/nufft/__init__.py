import logging
from collections import OrderedDict

import numpy as np

from aspire import config
from aspire.utils import complex_type, real_type

logger = logging.getLogger(__name__)

# Cached Plan Class objects, indexed by backend string identifier, and ordered by preference (highest first)
# The values are either Plan subclasses (for working backends), or None (for non-working backends)
# Populated by 'check_backends()' when first needed.
backends = None
# Default preferred Plan subclass
default_plan_class = None


def check_backends(raise_errors=True):
    """
    Check all NFFT backends in package configuration
    :param raise_errors: Whether to raise a RuntimeError if no backends detected.

    :return: On return, the global names 'backends'/'default_plan_class' have been populated
    """

    global backends, default_plan_class

    def _try_backend(backend):
        """
        This function tries out a particular NFFT backend by name.
        :param backend: A string representing the NFFT backend we want to try. Currently one of:
            'cufinufft'
                The Python wrapper for the CUDA variant of FINUFFT library
                https://github.com/flatironinstitute/cufinufft
            'finufft'
                The Python wrapper for the FlatIron Institute's FINUFFT library
                https://github.com/flatironinstitute/finufft
            'pynfft'
                The Python wrapper for the Chemnitz NFFT library
                https://www-user.tu-chemnitz.de/~potts/nfft/
        :return: The proper Plan-subclass if the backend is expected to work or None otherwise.

        It's important to keep these checks lightweight since all usable backend classes are cached on module load.
        We only check for working imports here to keep things simple and lightweight.
        If imports are working but the actual backend is not then we have bigger problems anyway.
        """

        logger.info(f"Trying NFFT backend {backend}")
        plan_class = None
        msg = None
        if backend == "cufinufft":
            try:
                # Note we expect the cu library is in LD path.
                #   That is managed by pip if using a wheel,
                #   or the user if they have built from source.
                from aspire.nufft.cufinufft import CufinufftPlan

                plan_class = CufinufftPlan
            except Exception as e:
                msg = str(e)

        elif backend == "finufft":
            try:
                from finufft import Plan  # noqa: F401

                from aspire.nufft.finufft import FinufftPlan

                plan_class = FinufftPlan
            except Exception as e:
                msg = str(e)

        elif backend == "pynfft":
            try:
                from pynfft.nfft import NFFT  # noqa: F401

                from aspire.nufft.pynfft import PyNfftPlan

                plan_class = PyNfftPlan
            except Exception as e:
                msg = str(e)

        if plan_class is None:
            logger.info(f"NFFT backend {backend} not usable:\n\t{msg}")
        else:
            logger.info(f"NFFT backend {backend} usable.")
            return plan_class

    backends = OrderedDict((k, _try_backend(k)) for k in config.nfft.backends)
    try:
        default_backend = next(k for k, v in backends.items() if v is not None)
        logger.info(f"Selected NFFT backend = {default_backend}.")
        default_plan_class = backends[default_backend]
    except StopIteration:
        msg = "No usable NFFT backend detected."
        logger.error(msg)
        default_plan_class = None
        if raise_errors:
            raise RuntimeError(msg) from None


def all_backends():
    """
    Determine all available NFFT backends

    :return: A list of strings representing available NFFT backends
    """
    global backends

    if backends is None:
        check_backends(raise_errors=False)
    return [k for k, v in backends.items() if v is not None]


def backend_available(backend):
    """
    Whether a particular NFFT backend is available

    :param backend: String representing the NFFT backend, e.g. 'finufft' or 'pynfft'
    :return: Boolean on whether the backend is available
    """
    return backend in all_backends()


class Plan:
    # TODO: move common functionality up the hierarchy
    def __new__(cls, *args, **kwargs):
        if cls is Plan:
            if "backend" in kwargs:
                if backend_available(kwargs["backend"]):
                    return super(Plan, cls).__new__(backends[kwargs["backend"]])
                else:
                    raise RuntimeError("Requested backend unavailable")
            else:
                # If a Plan was constructed as a generic Plan(), use the default (best) Plan class
                if default_plan_class is None:
                    check_backends(raise_errors=True)
                return super(Plan, cls).__new__(default_plan_class)
        else:
            # If a Plan-subclass was constructed directly, invoke default behavior
            return super(Plan, cls).__new__(cls)


def anufft(sig_f, fourier_pts, sz, real=False):
    """
    Wrapper for 1, 2, and 3 dimensional Non Uniform FFT Adjoint.
    Dimension is based on the dimension of fourier_pts and checked against sig_f.

    Selects best available package from `nfft` `backends` configuration list.

    :param sig_f: Array representing the signal(s) in Fourier space to be transformed. \
    sig_f either matches length of fourier_pts or sig_f.shape is stack of (`ntransforms`, ...).
    :param fourier_pts: The points in Fourier space where the Fourier transform is to be calculated,
            arranged as a dimension-by-K array. These need to be in the range [-pi, pi] in each dimension.
    :param sz: A tuple indicating the geometry of the signal.
    :param real: Optional Bool indicating if you would like only the real components, Defaults False.
    :return: The Non Uniform FFT adjoint transform.

    """

    if fourier_pts.dtype != real_type(sig_f.dtype):
        logger.warning(
            "anufft passed inconsistent dtypes."
            f" fourier_pts: {fourier_pts.dtype}"
            f" forcing precision of signal data: {sig_f.dtype}."
        )
        fourier_pts = fourier_pts.astype(real_type(sig_f.dtype))

    if sig_f.dtype != complex_type(sig_f.dtype):
        logger.debug("anufft passed real_type for signal, converting")
        sig_f = sig_f.astype(complex_type(sig_f.dtype))

    ntransforms = 1
    if len(sig_f.shape) == 2:
        ntransforms = sig_f.shape[0]

    plan = Plan(sz=sz, fourier_pts=fourier_pts, ntransforms=ntransforms)
    adjoint = plan.adjoint(sig_f)
    return np.real(adjoint) if real else adjoint


def nufft(sig_f, fourier_pts, real=False):
    """
    Wrapper for 1, 2, and 3 dimensional Non Uniform FFT
    Dimension is based on the dimension of fourier_pts and checked against sig_f.

    Selects best available package from `nfft` `backends` configuration list.

    :param sig_f: Array representing the signal(s) in real space to be transformed. \
    sig_f either matches `sz` or sig_f.shape is stack of (..., `ntransforms`).
    :param fourier_pts: The points in Fourier space where the Fourier transform is to be calculated,
            arranged as a dimension-by-K array. These need to be in the range [-pi, pi] in each dimension.
    :param real: Optional Bool indicating if you would like only the real components, Defaults False.
    :return: The Non Uniform FFT transform.

    """

    if fourier_pts.dtype != real_type(sig_f.dtype):
        logger.warning(
            "nufft passed inconsistent dtypes."
            f" fourier_pts: {fourier_pts.dtype}"
            f" forcing precision of signal data: {sig_f.dtype}."
        )
        fourier_pts = fourier_pts.astype(real_type(sig_f.dtype))

    if sig_f.dtype != complex_type(sig_f.dtype):
        logger.debug("nufft passed real_type for signal, converting")
        sig_f = sig_f.astype(complex_type(sig_f.dtype))

    # Unpack the dimension of the signal
    #   Note, infer dimension from fourier_pts because signal might be a stack.
    dimension = fourier_pts.shape[0]

    # Unpack the resolution of the signal
    resolution = sig_f.shape[1]

    # Construct tuple describing geometry of signal
    sz = (resolution,) * dimension

    ntransforms = 1
    if len(sig_f.shape) == dimension + 1:
        ntransforms = sig_f.shape[0]

    plan = Plan(sz=sz, fourier_pts=fourier_pts, ntransforms=ntransforms)
    transform = plan.transform(sig_f)
    return np.real(transform) if real else transform
