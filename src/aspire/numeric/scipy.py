"""
Utility wrappers for scipy methods.
"""

import scipy
from packaging.version import Version


def cg(*args, **kwargs):
    """
    Supports scipy cg before and after 1.14.0.
    """

    # older scipy cg interface uses `tol` instead of `rtol`
    if Version(scipy.__version__) < Version("1.14.0"):
        kwargs["tol"] = kwargs.pop("rtol", None)
    return scipy.sparse.linalg.cg(*args, **kwargs)
