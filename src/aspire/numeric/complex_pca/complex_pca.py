"""
ComplexPCA

We're just going to copy scikits PCA and extend to complex.
They refuse to support complex, not because of this code,
but because of how complicated other portions of their
package would become when extended to complex.
They don't want to mix support by admitting complex in only
a few places and not supporting it/crashing in other areas of code.

Unfortunately we need a complex valued PCA, so we wrap theirs for now.
"""

import numpy as np
import scipy.sparse as sp
import sklearn
from packaging.version import Version
from sklearn.decomposition import PCA
from sklearn.utils._array_api import get_namespace

from .validation import check_array


class ComplexPCA(PCA):
    # In a more ideal world we could patch with something smaller
    # but some versions of scikit do not admit this.
    # def _validate_data(self, *args, **kwargs):
    #     # stuff
    #     return
    # We will instead need to override the method and some dependent methods directly.

    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if sp.issparse(X):
            raise TypeError(
                "PCA does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )

        X = check_array(
            X,
            dtype=[np.complex128, np.complex64, np.float64, np.float32],
            ensure_2d=True,
            copy=self.copy,
            allow_complex=True,
        )

        xp, is_array_api_compliant = get_namespace(X)

        # Handle n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            elif n_components >= 1 and n_components < 0.8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        # sci-kit changed `_fit_*()` API in latest release v1.5.0
        # which supports Python 3.9 - 3.12. This can be removed after
        # our minimal support is Python 3.9.
        API_dep = Version(sklearn.__version__) < Version("1.5.0")

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == "full":
            if API_dep:
                return self._fit_full(X, n_components)
            else:
                return self._fit_full(X, n_components, xp, is_array_api_compliant)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            if API_dep:
                return self._fit_truncated(X, n_components, self._fit_svd_solver)
            else:
                return self._fit_truncated(X, n_components, xp)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self._fit_svd_solver)
            )
