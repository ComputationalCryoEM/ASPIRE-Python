import numpy as np
from unittest import TestCase

from aspyre.basis.basis_func import besselj_zeros, num_besselj_zeros, sph_bessel, real_sph_harmonic
from aspyre.basis.basis_func import norm_assoc_legendre, unique_coords_nd, lgwt


class BesselTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testBesselJZeros(self):
        zeros = besselj_zeros(5.39, 10)
        self.assertTrue(np.allclose(
            zeros, [
                9.22762357, 12.82884915, 16.21119514, 19.50556547, 22.75392676,
                25.97476169, 29.17767578, 32.36821427, 35.54982352, 38.72476688
            ]
        ))

    def testNumBesselJZeros(self):
        n, zeros = num_besselj_zeros(10, 20)
        self.assertEqual(2, n)
        self.assertTrue(np.allclose(
            zeros, [
                14.47550069, 18.43346367
            ]
        ))

    def testSphBesselj(self):
        r = np.array([
            0, 0.785398163397448, 1.11072073453959, 1.36034952317566, 1.57079632679490, 1.75620368276018,
            1.92382474524280, 2.22144146907918, 2.35619449019235, 2.35619449019235, 2.48364706644903,
            2.60487101902358, 2.60487101902358, 2.72069904635133, 2.83179334978474, 2.93869083963475,
            2.93869083963475, 3.14159265358979
        ])
        sph = sph_bessel(0, r)
        self.assertTrue(np.allclose(
            sph, [
                1.000000000000000, 0.900316316157106, 0.806700467600633, 0.718887065276235, 0.636619772367581,
                0.559651051304123, 0.487741916756892, 0.358187786013244, 0.300105438719035, 0.300105438719035,
                0.246207521717852, 0.196294306927466, 0.196294306927466, 0.150173255502137, 0.107658809425615,
                0.068572188169309, 0.068572188169308, -0.000000000000000
            ]
        ))

    def testUniqGrid2d(self):
        res = unique_coords_nd(8, 2)
        self.assertTrue(np.allclose(
            res['r_unique'],
            [
                0., 0.25, 0.35355, 0.5, 0.55902, 0.70711, 0.75, 0.79057, 0.90139, 1.
            ]
        ))
        self.assertEqual(res['ang_unique'].shape, (32,))

    def testUniqGrid3d(self):
        res = unique_coords_nd(8, 3)
        self.assertTrue(np.allclose(
            res['r_unique'],
            [
                0., 0.25, 0.35355, 0.43301, 0.5, 0.55902, 0.61237, 0.70711,
                0.75, 0.79057, 0.82916, 0.86603, 0.90139, 0.93541, 1.
            ]
        ))
        self.assertEqual(res['ang_unique'].shape, (2, 218))

    def testNormAssocLegendre(self):
        res = norm_assoc_legendre(
            j=3,  # degree
            m=-2,  # order (abs(m) <= j)
            x=np.array([-1., -0.77777778, -0.55555556, -0.33333333, -0.11111111,
                0.11111111,  0.33333333,  0.55555556,  0.77777778,  1.])
        )
        self.assertTrue(np.allclose(
            res,
            [-0., -0.78714574, -0.98393217, -0.75903339, -0.28112348,
                0.28112348,  0.75903339,  0.98393217,  0.78714574,  0.]
        ))

    def testSphHarmonic(self):
        res = real_sph_harmonic(
            j=3,  # degree
            m=-2,  # order (abs(m) <= j)
            theta=np.array([2.1415, 1.492, 0.213]),
            phi=np.array([1.45, 0.213, 4.4234])
        )
        self.assertTrue(np.allclose(
            res,
            [-0.1322862, 0.04672082, 0.03448817]
        ))

    def testLGQuad(self):
        resx, resw = lgwt(
            ndeg=10,  # degree
            a=0.0,  # start x
            b=0.5  # end x
        )
        self.assertTrue(np.allclose(
            resx,
            [0.00652337, 0.03373416, 0.08014761, 0.14165115, 0.21278142,
             0.28721858, 0.35834885, 0.41985239, 0.46626584, 0.49347663
             ]
        ))

        self.assertTrue(np.allclose(
            resw,
            [0.01666784, 0.03736284, 0.05477159, 0.06731668, 0.07388106,
             0.07388106, 0.06731668, 0.05477159, 0.03736284, 0.01666784
             ]
        ))
