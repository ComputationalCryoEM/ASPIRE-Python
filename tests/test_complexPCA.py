from unittest import TestCase

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from aspire.numeric import ComplexPCA
from aspire.utils import complex_type


class ComplexPCACase(TestCase):
    """
    ComplexPCA is a wrapper of sklearn's PCA.

    We just want to make sure we cover the code/args we've touched with our wrapper.
    """

    def setUp(self):
        self.dtype = np.float32

        # Setup small experiment
        self.components_small = 2
        self.X_small = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=self.dtype
        )

        # Setup larger experiment
        self.n_samples = n_samples = 2048
        self.m_features = m_features = 600
        self.components_large = 300
        self.X_large = np.random.random(
            (n_samples, m_features)
        ) + 1j * np.random.random((n_samples, m_features))

    def testSmallFitTransform(self):
        # Reals
        pca = PCA(n_components=self.components_small)
        Y1 = pca.fit_transform(self.X_small)

        # Complex
        cpca = ComplexPCA(n_components=self.components_small)
        Y2 = cpca.fit_transform(self.X_small.astype(complex_type(self.dtype)))

        # Real part should be the same.
        self.assertTrue(np.allclose(np.real(Y2), Y1))
        # Imag part should be zero.
        self.assertTrue(np.allclose(np.imag(Y2), 0))

    def testLargeFitTransform(self):
        """
        Smoke test a more realistic size.
        """

        pca = ComplexPCA(n_components=self.components_large, copy=False)

        # Input data matrix X should be (n_samples, m_features)
        X = self.X_large
        # Resulting reduced Y should be (n_samples, components)
        Y = pca.fit_transform(X)

        # later come back and check reconstruction is within tol.
        Xest = pca.inverse_transform(Y)

        # Later come back and check reconstruction more clever like.
        Xest = pca.inverse_transform(Y)
        rmse = np.sqrt(np.mean(np.square(X - Xest)))
        # For now I think this should be around 1/2 + 1/2j (pure noise)
        self.assertTrue(np.allclose(rmse, 0.5 + 0.5j, atol=0.01))

    def testSparseInputError(self):
        """
        Make a sparse input and check raises.
        """

        pca = ComplexPCA(n_components=self.components_small)
        sparse_X = csr_matrix(self.X_small)
        with pytest.raises(TypeError, match=r"PCA does not support sparse.*"):
            _ = pca.fit_transform(sparse_X)

    def testDefault_n_components(self):
        """
        Check we can take automatic n_component branches
        """

        pca = ComplexPCA(n_components=None)
        _ = pca.fit_transform(self.X_small)

        # Internally there is a different calculation when using arpack
        pca = PCA(n_components=None, svd_solver="arpack")
        _ = pca.fit_transform(self.X_small)

    def testSovlerArg(self):
        """
        Check we raise when given unsupported solver.
        """

        pca = ComplexPCA(n_components=self.components_small, svd_solver="notasolver")
        with pytest.raises(ValueError, match=r"Unrecognized svd_solver.*"):
            _ = pca.fit_transform(self.X_small)
