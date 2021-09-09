from unittest import TestCase

import numpy as np
import pytest
from numpy.linalg import norm, solve

from aspire.operators import BlkDiagMatrix


class BlkDiagMatrixTestCase(TestCase):
    def setUp(self):

        self.num_blks = 10

        self.blk_partition = [(i, i) for i in range(self.num_blks, 0, -1)]
        self.dense_shape = np.sum(self.blk_partition, axis=0)

        n = np.sum(np.prod(np.array(self.blk_partition), axis=1))
        self.flat = np.arange(n)
        self.revflat = self.flat[::-1].copy()

        diag_ind = np.array([0, 0])
        ind = 0
        zeros = []
        ones = []
        eyes = []
        A = []
        B = []
        self.dense = np.zeros(self.dense_shape)
        for blk_shp in self.blk_partition:
            zeros.append(np.zeros(blk_shp))
            ones.append(np.ones(blk_shp))
            eyes.append(np.eye(blk_shp[0]))

            offt = np.prod(blk_shp)
            blk = self.flat[ind : ind + offt].reshape(blk_shp)
            A.append(blk)
            B.append(self.revflat[ind : ind + offt].reshape(blk_shp))

            ind += offt

            # Also build a dense array.
            self.dense[
                diag_ind[0] : diag_ind[0] + blk_shp[0],
                diag_ind[1] : diag_ind[1] + blk_shp[1],
            ] = blk
            diag_ind += blk_shp

        self.blk_a = BlkDiagMatrix.from_list(A)
        self.blk_b = BlkDiagMatrix.from_list(B)
        self.blk_zeros = BlkDiagMatrix.from_list(zeros)
        self.blk_ones = BlkDiagMatrix.from_list(ones)
        self.blk_eyes = BlkDiagMatrix.from_list(eyes)

    def tearDown(self):
        pass

    def allallfunc(self, A, B, func=np.allclose):
        """Checks assertTrue(func()) as it iterates through A, B."""
        for (a, b) in zip(A, B):
            self.assertTrue(func(a, b))

    def allallid(self, A, B_ids, func=np.allclose):
        """Checks id(a) matches b_id for (a, b_id) in zip(A, B_ids)."""
        return self.allallfunc(A, B_ids, func=lambda x, y: id(x) == y)

    def testBlkDiagMatrixCompat(self):
        """Check incompatible matrix raises exception."""
        # Create a differently shaped matrix
        x = BlkDiagMatrix.from_list(self.blk_a[1:-1])
        # code should raise
        with pytest.raises(RuntimeError):
            _ = x + self.blk_a

    def testBlkDiagMatrixPartition(self):
        # Test class attribute
        self.allallfunc(self.blk_a.partition, self.blk_partition)

        # Test utility function
        blk_partition = self.blk_a.partition
        self.allallfunc(blk_partition, self.blk_partition)

    def testBlkDiagMatrixZeros(self):
        blk_zeros = BlkDiagMatrix.zeros(self.blk_partition)
        self.allallfunc(blk_zeros, self.blk_zeros)

        blk_zeros = BlkDiagMatrix.zeros_like(self.blk_a)
        self.allallfunc(blk_zeros, self.blk_zeros)

    def testBlkDiagMatrixOnes(self):
        blk_ones = BlkDiagMatrix.ones(self.blk_partition)
        self.allallfunc(blk_ones, self.blk_ones)

    def testBlkDiagMatrixEye(self):
        blk_eye = BlkDiagMatrix.eye(self.blk_partition)
        self.allallfunc(blk_eye, self.blk_eyes)

        blk_eye = BlkDiagMatrix.eye_like(self.blk_a)
        self.allallfunc(blk_eye, self.blk_eyes)

    def testBlkDiagMatrixAdd(self):
        result = [np.add(*tup) for tup in zip(self.blk_a, self.blk_b)]

        blk_c = self.blk_a + self.blk_b
        self.allallfunc(result, blk_c)

        blk_c = self.blk_a.add(self.blk_b)
        self.allallfunc(result, blk_c)

    def testBlkDiagMatrixSub(self):
        result = [np.subtract(*tup) for tup in zip(self.blk_a, self.blk_b)]

        blk_c = self.blk_a - self.blk_b
        self.allallfunc(result, blk_c)

        blk_c = self.blk_a.sub(self.blk_b)
        self.allallfunc(result, blk_c)

    def testBlkDiagMatrixApply(self):
        m = np.sum(self.blk_a.partition[:, 1])
        k = 3
        coeffm = np.arange(k * m).reshape(m, k).astype(self.blk_a.dtype)

        # Manually compute
        ind = 0
        res = np.empty_like(coeffm)
        for b, blk in enumerate(self.blk_a):
            col = self.blk_a.partition[b, 1]
            res[ind : ind + col, :] = blk @ coeffm[ind : ind + col, :]
            ind += col

        # Check ndim 1 case
        c = self.blk_a.apply(coeffm[:, 0])
        self.allallfunc(c, res[:, 0])

        # Check ndim 2 case
        d = self.blk_a.apply(coeffm)
        self.allallfunc(res, d)

        # Here we are checking that the ndim 2 case distributes as described.
        # Specifically d = A.apply([[r0], ... [ri]])
        # should be equivalent to e = [A.apply(r0), ... A.apply(ri)].
        e = np.empty((m, k))
        for i in range(k):
            e[:, i] = self.blk_a.apply(coeffm[:, i])
        self.allallfunc(e, d)

        # We can use syntactic sugar @ for apply as well
        f = self.blk_a @ coeffm
        self.allallfunc(f, d)

        # Test the rapply is also functional
        coeffm = coeffm.T  # matmul dimensions
        res = coeffm @ self.blk_a.dense()
        d = self.blk_a.rapply(coeffm)
        self.allallfunc(res, d)

        # And the syntactic sugar @
        d = coeffm @ self.blk_a
        self.allallfunc(res, d)

        # And test some incorrrect invocations:
        # inplace not supported for matmul of mixed classes.
        with pytest.raises(RuntimeError, match=r".*method not supported.*"):
            self.blk_a @= coeffm

        # Test left operand of an __rmatmul__ must be an ndarray
        with pytest.raises(
            RuntimeError, match=r".*only defined for np.ndarray @ BlkDiagMatrix.*"
        ):
            _ = list(coeffm) @ self.blk_a

    def testBlkDiagMatrixMatMult(self):
        result = [np.matmul(*tup) for tup in zip(self.blk_a, self.blk_b)]

        blk_c = self.blk_a @ self.blk_b
        self.allallfunc(blk_c, result)

        self.blk_a.matmul(self.blk_b)
        self.allallfunc(blk_c, result)

    def testBlkDiagMatrixScalarMult(self):
        result = [blk * 42.0 for blk in self.blk_a]

        blk_c = self.blk_a * 42.0
        self.allallfunc(blk_c, result)

        # also test right multiply
        blk_c = 42.0 * self.blk_a
        self.allallfunc(blk_c, result)

    def testBlkDiagMatrixScalarAdd(self):
        result = [blk + 42.0 for blk in self.blk_a]

        blk_c = self.blk_a + 42.0
        self.allallfunc(blk_c, result)

        # and test the right add
        blk_c = 42.0 + self.blk_a
        self.allallfunc(blk_c, result)

    def testBlkDiagMatrixScalarSub(self):
        result_1 = [blk - 42.0 for blk in self.blk_a]

        # a-b = -1*(b-a)
        result_2 = [-1 * x for x in result_1]

        blk_c = self.blk_a - 42.0
        self.allallfunc(blk_c, result_1)

        # and test the right sub
        blk_c = 42.0 - self.blk_a
        self.allallfunc(blk_c, result_2)

    def testBlkDiagMatrixDeepCopy(self):
        blk_a_copy_1 = self.blk_a.copy()
        blk_a_copy_2 = self.blk_a.copy()

        # test same values as blk_a
        self.allallfunc(blk_a_copy_1, self.blk_a)
        self.allallfunc(blk_a_copy_2, self.blk_a)

        # change a copy, test that copy is changed
        blk_a_copy_1 *= 2.0
        self.allallfunc(
            blk_a_copy_1, self.blk_a, func=lambda x, y: not np.allclose(x, y)
        )
        self.allallfunc(
            blk_a_copy_1, blk_a_copy_2, func=lambda x, y: not np.allclose(x, y)
        )

        # and blk_a is unchanged
        self.allallfunc(blk_a_copy_2, self.blk_a)

    def testBlkDiagMatrixInPlace(self):
        """Tests sequence of in place optimized arithmetic (add, sub, mul)"""
        _ = [x + x + 10.0 for x in self.blk_a]
        _ = [np.ones(x.shape) * 5.0 for x in self.blk_a]

        # make a block diagonal object to mutate
        blk_c = self.blk_a.copy()
        # store the python object ids of each array in blk_c
        #  we want to ensure the actual object refs are _not_ changing
        #  for in place operations.
        id0 = [id(x) for x in blk_c]

        blk_c += self.blk_a
        self.allallid(blk_c, id0)

        blk_c += 10.0
        self.allallid(blk_c, id0)

        blk_a5 = BlkDiagMatrix.ones(self.blk_partition)
        id1 = [id(x) for x in blk_a5]
        blk_a5 *= 5.0
        self.allallid(blk_a5, id1)

        blk_c -= blk_a5
        blk_c -= blk_a5
        self.allallid(blk_c, id0)

        blk_c -= self.blk_a
        self.allallid(blk_c, id0)

        self.allallfunc(blk_c, self.blk_a)

    def testBlkDiagMatrixNorm(self):
        result = np.max([norm(blk, ord=2) for blk in self.blk_a])
        self.assertTrue(result == self.blk_a.norm())

    def testBlkDiagMatrixSolve(self):
        # We'll need a non singular matrix
        B = self.blk_a + self.blk_eyes

        m = np.sum(self.blk_a.partition[:, 1])
        k = 3
        coeffm = np.arange(k * m).reshape(m, k).astype(self.blk_a.dtype)

        # Manually compute
        ind = 0
        res = np.empty_like(coeffm)
        for b, blk in enumerate(B):
            col = self.blk_a.partition[b, 1]
            res[ind : ind + col, :] = solve(blk, coeffm[ind : ind + col, :])
            ind += col

        coeff_est = B.solve(coeffm)
        self.allallfunc(res, coeff_est)

    def testBlkDiagMatrixTranspose(self):
        blk_c = [blk.T for blk in self.blk_a]
        self.allallfunc(blk_c, self.blk_a.transpose())
        self.allallfunc(blk_c, self.blk_a.T)

        blk_aaT = (self.blk_a @ self.blk_a).T
        blk_aTaT = self.blk_a.T @ self.blk_a.T
        self.allallfunc(blk_aaT, blk_aTaT)

    def testBlkDiagMatrixNeg(self):
        result = [-blk for blk in self.blk_a]

        blk_c = -self.blk_a
        self.allallfunc(blk_c, result)

    def testBlkDiagMatrixAbs(self):
        result = [np.abs(blk) for blk in self.blk_a]

        blk_c = abs(self.blk_a)

        self.allallfunc(blk_c, result)

    def testBlkDiagMatrixPow(self):
        result = [blk ** 2 for blk in self.blk_a]

        blk_c = self.blk_a ** 2.0
        self.allallfunc(blk_c, result)

        # In place power
        # store the python object ids of each array in blk_c
        #  we want to ensure the actual object refs are _not_ changing
        #  for in place operations.
        id0 = [id(x) for x in blk_c]

        blk_c **= 0.5
        self.allallid(blk_c, id0)
        self.allallfunc(blk_c, abs(self.blk_a))

    def testBlkDiagMatrixIsFinite(self):
        self.assertTrue(self.blk_a.isfinite)

        # construct a copy to mutate
        blk_inf = self.blk_a.copy()
        # assign inf value
        blk_inf[0][0] = np.inf
        self.assertFalse(blk_inf.isfinite)

        # construct a copy to mutate
        blk_nan = self.blk_a.copy()
        # assign inf value
        blk_nan[0][0] = np.nan
        self.assertFalse(blk_nan.isfinite)

    def testBlkDiagMatrixDense(self):
        """Test we correctly compute the right shape and array."""
        self.assertTrue(np.allclose(self.dense, self.blk_a.dense()))

    def testBlkDiagMatrixArith(self):
        """
        Compute a sequence of operations on `BlkDiagMatrix`
        and numpy arrays independently then compare.
        """

        inputs = [(self.blk_a.dense(), self.blk_b.dense()), (self.blk_a, self.blk_b)]

        # Note we intentionally skip scalar add/sub operations because,
        #   the equivalent numpy operations would require masking.
        results = []
        for A, B in inputs:
            res = abs(A @ B) * 0.5 - (B @ B.T) ** 2
            results.append(res)

        self.assertTrue(np.allclose(results[0], results[1].dense()))


class IrrBlkDiagMatrixTestCase(TestCase):
    """
    Tests Irregular (non square) Block Diagonal Matrices.
    """

    def setUp(self):

        partition = [[4, 5], [2, 3], [1, 1]]
        self.X = X = [(1 + np.arange(np.prod(p))).reshape(p) for p in partition]
        self.XT = XT = [x.T for x in X]

        self.blk_x = BlkDiagMatrix.from_list(X)
        self.blk_xt = BlkDiagMatrix.from_list(XT)

    def allallfunc(self, A, B, func=np.allclose):
        """Checks assertTrue(func()) as it iterates through A, B."""
        for (a, b) in zip(A, B):
            self.assertTrue(func(a, b))

    def testAdd(self):
        Y = [2 * x for x in self.X]
        BlkY = self.blk_x + self.blk_x

        self.allallfunc(Y, BlkY)

    def testAddIncompat(self):
        with pytest.raises(
            RuntimeError, match=r".*BlkDiagMatrix instances are not same shape.*"
        ):
            _ = self.blk_x + self.blk_xt

    def testSub(self):
        Y = BlkDiagMatrix.zeros_like(self.blk_x)
        BlkY = self.blk_x - self.blk_x

        self.allallfunc(Y, BlkY)

    def testSubIncompat(self):
        with pytest.raises(
            RuntimeError, match=r".*BlkDiagMatrix instances are not same shape.*"
        ):
            _ = self.blk_x - self.blk_xt

    def testScalars(self):
        self.allallfunc((42 - self.blk_x) * 13, [13 * (42 - x) for x in self.X])

    def testTranspose(self):
        self.allallfunc(self.blk_x.T, self.blk_xt)

    def testMatMul(self):
        result = [np.matmul(*tup) for tup in zip(self.X, self.XT)]

        blk_y = self.blk_x @ self.blk_xt
        self.allallfunc(blk_y, result)

        blk_y = self.blk_x.matmul(self.blk_xt)
        self.allallfunc(blk_y, result)

    def testMatMulIncompat(self):
        with pytest.raises(
            RuntimeError, match=r".*BlkDiagMatrix instances are not compatible.*"
        ):
            _ = self.blk_x @ self.blk_x

    def testApply(self):
        n = np.sum(self.blk_x.partition[:, 0])
        m = np.sum(self.blk_x.partition[:, 1])
        k = 3
        coeffm = np.arange(k * m).reshape(m, k).astype(self.blk_x.dtype)

        # Manually compute
        indc = 0
        indr = 0
        res = np.empty(shape=(n, k), dtype=coeffm.dtype)
        for b, blk in enumerate(self.blk_x):
            row, col = self.blk_x.partition[b]
            res[indr : indr + row, :] = blk @ coeffm[indc : indc + col, :]
            indc += col
            indr += row

        # Check ndim 1 case
        c = self.blk_x.apply(coeffm[:, 0])
        self.allallfunc(c, res[:, 0])

        # Check ndim 2 case
        d = self.blk_x.apply(coeffm)
        self.allallfunc(res, d)

        # Check against dense numpy matmul
        self.allallfunc(d, self.blk_x.dense() @ coeffm)

    def testSolve(self):
        """
        Test attempts to solve non square BlkDiagMatrix raise error.
        """

        # Setup a dummy coeff matrix
        n = np.sum(self.blk_x.partition[:, 0])
        k = 3
        coeffm = np.arange(n * k).reshape(n, k).astype(self.blk_x.dtype)

        with pytest.raises(
            NotImplementedError,
            match=r"BlkDiagMatrix.solve is only defined for square arrays.*",
        ):
            # Attemplt solve using the Block Diagonal implementation
            _ = self.blk_x.solve(coeffm)
