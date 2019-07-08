import numpy as np
from unittest import TestCase, skipUnless

from aspyre.nfft import Plan, backend_available

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class SimTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @skipUnless(backend_available('finufft'), 'unsupported backend')
    def testTransform1(self):
        vol = np.load(os.path.join(DATA_DIR, 'nfft_volume.npy'))
        fourier_pts = np.array([
            [ 0.88952655922411,  0.35922344760724, -0.17107966400962, -0.70138277562649],
            [ 1.87089316522016,  1.99362869011803,  2.11636421501590,  2.23909973991377],
            [-3.93035749861843, -3.36417300942290, -2.79798852022738, -2.23180403103185]
        ])

        plan = Plan(vol.shape, fourier_pts, backend='finufft')
        result = plan.transform(vol)

        self.assertTrue(np.allclose(
            result,
            [-0.05646675 + 1.503746j, 1.677600 + 0.6610926j, 0.9124417 - 0.7394574j, -0.9136836 - 0.5491410j]
        ))

    @skipUnless(backend_available('pynfft'), 'unsupported backend')
    def testTransform2(self):
        vol = np.load(os.path.join(DATA_DIR, 'nfft_volume.npy'))
        fourier_pts = np.array([
            [ 0.88952655922411,  0.35922344760724, -0.17107966400962, -0.70138277562649],
            [ 1.87089316522016,  1.99362869011803,  2.11636421501590,  2.23909973991377],
            [-3.93035749861843, -3.36417300942290, -2.79798852022738, -2.23180403103185]
        ])

        plan = Plan(vol.shape, fourier_pts, backend='pynfft')
        result = plan.transform(vol)

        self.assertTrue(np.allclose(
            result,
            [-0.05646675 + 1.503746j, 1.677600 + 0.6610926j, 0.9124417 - 0.7394574j, -0.9136836 - 0.5491410j]
        ))