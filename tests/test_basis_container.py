from unittest import TestCase

import numpy as np

from aspire.basis import CoefContainer


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
        return self.assertTrue(np.all(a == b))

    def testContainerZero(self):

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

    def testBounds(self):
        pass

    def testDims(self):
        pass
