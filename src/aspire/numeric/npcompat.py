"""
Util code for Numpy1 and Numpy2 cross compatibility.
"""

import numpy as np
from packaging.version import parse as parse_version

# The Numpy2 default of `None` is equivalent to `False` in Numpy1.
# No current version of Numpy1 appears to support `None` despite that
# being a pretty obvious solution to migrating code.
# This should allow common array operations (astype etc) to
# operate with both major numpy versions.
COPY_ME_MAYBE = None
if parse_version(np.version.version) < parse_version("2.0.0"):
    COPY_ME_MAYBE = False
