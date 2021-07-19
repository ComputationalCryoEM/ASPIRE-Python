import logging
from unittest import TestCase

import numpy as np

from aspire.basis import CoefContainer

logger = logging.getLogger(__name__)


class CoefContainerTestCase(TestCase):
    def setUp(self):

        # Construct an index mapping
        self.maps = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            ]
        )

        # Make a stack of data
        self.data = np.arange(1, 43).reshape(-1, 1) @ np.arange(
            self.maps.shape[-1]
        ).reshape(1, -1)

        # Setup a stack+2D container
        self.C2 = CoefContainer(self.data, self.maps[1:])

        # Setup a stack+3D container
        self.C3 = CoefContainer(self.data, self.maps)

    def assertAll(self, a, b):
        """
        Helper function, checks np compares all elements is True.
        """
        return self.assertTrue(np.all(a == b))

    def testContainerGetter(self):
        """
        Sanity test basic gets (accesses).
        """

        # 3D
        self.assertAll(self.data[0, 0], self.C3[:, 0:1, 0:1, 0:1])
        self.assertAll(self.data[0, 0], self.C3[:, 0, 0, 0])

        # 2D
        c = self.C2[:, 0, 0]
        self.assertAll(c[:, 0], np.zeros(c.shape[0]))
        # Should be multiples of 8
        self.assertAll(c[:, 1] // 8, np.arange(1, c.shape[0] + 1))
        # Int index should be consistent with slicing syntax
        self.assertAll(c, self.C2[:, 0:1, 0:1])

    def testContainerStackAxis(self):

        for i in range(self.data.shape[0]):
            self.assertAll(self.C3[i], self.data[i])

    def testContainerSetter(self):
        """
        Sanity check basic sets (assignments).
        """

        # Copy C3 so we can use as a reference later.
        C = self.C3.copy()

        # We'll excercise setting each axis in a different way.
        C[:, 1] = 0
        self.assertTrue(np.sum(C[:, 1]) == 0)

        C[:, :, 1:] = 1
        self.assertTrue(np.sum(C[:, :, 1:]) == 6 * 42)

        C[:, :, :, 0:2] = np.arange(6)
        self.assertTrue(np.sum(C[:, :, :, 0:2]) == 15 * 42)

        # Other entries should not have changed.
        self.assertAll(C[:, 0, 0, 2:], self.C3[:, 0, 0, 2:])

        # Sanity check stack axis assignment.
        x = np.arange(self.maps.shape[-1])
        C[13] = x
        self.assertAll(C[13], x)

    def testStringRepr(self):
        """
        String representations should not crash.
        """
        logger.info(repr(self.C2))
        logger.info(str(self.C2))
        logger.info(repr(self.C3))
        logger.info(str(self.C3))

    def testDims(self):
        self.assertTrue(self.C2.dim == 2)
        self.assertTrue(self.C3.dim == 3)

    def testShape(self):
        self.assertTrue(self.C2.shape == (2, 4))
        self.assertTrue(self.C3.shape == (2, 2, 4))

    def testBounds(self):
        self.assertTrue(self.C2.bounds == [(0, 1), (0, 3)])
        self.assertTrue(self.C3.bounds == [(0, 1), (0, 1), (0, 3)])

    def testNegative(self):
        """
        Container should handle negative indices.
        """

        # Setup dummy data imagining axes as:
        # ax1 [-1,0,1] ax2 [0,1,2,3,4]]
        nmap = [
            [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4],
        ]
        ndata = np.empty((123, 11))

        # Instantiate the container accessor
        C = CoefContainer(ndata, nmap)

        # Check shape and bounds
        self.assertTrue(C.shape == (3, 5))
        self.assertTrue(C.bounds == [(-1, 1), (0, 4)])

    def testIter(self):
        """
        Test we able to iterate over the container.

        Realistically only iterating the stack will be common.
        """

        # Iterate over stack
        for i, v in enumerate(self.C3):
            self.assertAll(v, (i + 1) * np.arange(self.maps.shape[-1]))

        # Iterate over first data axis of an element
        for i, ax1 in enumerate(self.C3[1]):
            self.assertTrue(2 * i == ax1)
