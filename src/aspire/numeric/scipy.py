"""
Utility wrappers for scipy methods.
"""

import scipy
from packaging.version import Version


def cg(*args, **kwargs):
    """
    Supports scipy cg before and after 1.12.0.
    """

    # older (<1.12.0) scipy cg interface uses `tol` instead of `rtol`.
    # `tol` will be removed in scipy 1.14.0.
    if Version(scipy.__version__) < Version("1.12.0"):
        kwargs["tol"] = kwargs.pop("rtol", None)
    return scipy.sparse.linalg.cg(*args, **kwargs)
