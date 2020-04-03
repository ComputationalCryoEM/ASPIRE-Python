import numpy as np
from unittest import TestCase

from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter, IdentityFilter

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class SimTestCase(TestCase):
    def setUp(self):
        self.sim = Simulation(
            n=1024,
            L=8,
            filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            seed=0,
            noise_filter=IdentityFilter()
        )

    def tearDown(self):
        pass

    def testGaussianBlob(self):
        blobs = self.sim.vols
        self.assertTrue(np.allclose(blobs, np.load(os.path.join(DATA_DIR, 'sim_blobs.npy'))))

    def testSimulationRots(self):
        self.assertTrue(np.allclose(
            self.sim.rots[0, :, :],
            np.array([
                [0.91675498, 0.2587233, 0.30433956],
                [0.39941773, -0.58404652, -0.70665065],
                [-0.00507853, 0.76938412, -0.63876622]
            ])
        ))

    def testSimulationImages(self):
        images = self.sim.clean_images(0, 512).asnumpy()
        self.assertTrue(np.allclose(images, np.load(os.path.join(DATA_DIR, 'sim_clean_images.npy')), rtol=1e-2))

    def testSimulationImagesNoisy(self):
        images = self.sim.images(0, 512).asnumpy()
        self.assertTrue(np.allclose(images, np.load(os.path.join(DATA_DIR, 'sim_images_with_noise.npy')), rtol=1e-2))

    def testSimulationImagesDownsample(self):
        # The simulation already generates images of size 8 x 8; Downsampling to resolution 8 should thus have no effect
        self.sim.downsample(8)
        images = self.sim.clean_images(0, 512).asnumpy()
        self.assertTrue(np.allclose(images, np.load(os.path.join(DATA_DIR, 'sim_clean_images.npy')), rtol=1e-2))

    def testSimulationImagesShape(self):
        # The 'images' method should be tolerant of bounds - here we ask for 1000 images starting at index 1000,
        # so we'll get back 25 images in return instead
        images = self.sim.images(1000, 1000)
        self.assertTrue(images.shape, (8, 8, 25))

    def testSimulationEigen(self):
        eigs_true, lambdas_true = self.sim.eigs()
        self.assertTrue(np.allclose(
            eigs_true[:, :, 2, 0],
            np.array([
                [-1.67666201e-07, -7.95741380e-06, -1.49160041e-04, -1.10151654e-03,
                 -3.11287888e-03, -3.09157884e-03, -9.91418026e-04, -1.31673165e-04],
                [-1.15402077e-06, -2.49849709e-05, -3.51658906e-04, -2.21575261e-03,
                 -7.83315487e-03, -9.44795180e-03, -4.07636259e-03, -9.02186439e-04],
                [-1.88737249e-05, -1.91418396e-04, -1.09021540e-03, -1.02020288e-03,
                 1.39411855e-02,  8.58035963e-03, -5.54619730e-03, -3.86377703e-03],
                [-1.21280536e-04, -9.51461843e-04, -3.22565017e-03, -1.05731178e-03,
                 2.61375736e-02,  3.11595201e-02,  6.40814053e-03, -2.31698658e-02],
                [-2.44067283e-04, -1.40560151e-03, -6.73082832e-05,  1.44160679e-02,
                 2.99893934e-02,  5.92632964e-02,  7.75623545e-02,  3.06570008e-02],
                [-1.53507499e-04, -7.21709803e-04,  8.54929152e-04, -1.27235036e-02,
                 -5.34382043e-03,  2.18879692e-02,  6.22706190e-02,  4.51998860e-02],
                [-3.00595184e-05, -1.43038429e-04, -2.15870258e-03, -9.99002904e-02,
                 -7.79077187e-02, -1.53395887e-02,  1.88777559e-02,  1.68759506e-02],
                [ 3.22692649e-05,  4.07977635e-03,  1.63959339e-02, -8.68835449e-02,
                 -7.86240026e-02, -1.75694861e-02,  3.24984640e-03,  1.95389288e-03]
            ])
        ))

    def testSimulationMean(self):
        mean_vol = self.sim.mean_true()
        self.assertTrue(
            np.allclose(
                [
                    [9.3048275e-06, 3.3865887e-04, 4.9073379e-03, 1.9983694e-02,
                     3.8744867e-02, 4.6177626e-02, 2.9706439e-02, 9.6760402e-03],
                    [3.9041879e-05, 2.4739134e-03, 3.8184751e-02, 1.2325400e-01,
                     2.2278427e-01, 2.5246662e-01, 1.4093880e-01, 3.6834739e-02],
                    [1.4176674e-04, 1.1911459e-02, 1.4421061e-01, 3.8428229e-01,
                     7.8645325e-01, 8.6522675e-01, 4.4862464e-01, 1.6382274e-01],
                    [6.6035596e-04, 3.1378061e-02, 2.9226971e-01, 9.7105372e-01,
                     2.3941052e+00, 2.1709986e+00, 1.2359586e+00, 4.9233937e-01],
                    [2.7174791e-03, 5.4912899e-02, 4.9955705e-01, 2.0535610e+00,
                     3.7094145e+00, 3.0157866e+00, 1.5144194e+00, 5.2054578e-01],
                    [5.8484524e-03, 6.9626346e-02, 5.0568020e-01, 1.9964373e+00,
                     3.7741590e+00, 2.7603974e+00, 1.0460200e+00, 2.0633203e-01],
                    [5.3958315e-03, 6.0689718e-02, 4.7008950e-01, 1.1712804e+00,
                     1.8282105e+00, 1.1874394e+00, 3.0667788e-01, 4.8514768e-02],
                    [2.4636178e-03, 4.8677865e-02, 6.5284950e-01, 6.5238869e-01,
                     6.5745544e-01, 3.7955683e-01, 8.0530584e-02, 1.2100547e-02]
                ],
                mean_vol[:, :, 4]
            )
        )

    def testSimulationVolCoords(self):
        coords, norms, inners = self.sim.vol_coords()
        self.assertTrue(np.allclose([4.72837704, -4.72837709], coords, atol=1e-4))
        self.assertTrue(np.allclose([8.20515764e-07, 1.17550184e-06], norms, atol=1e-4))
        self.assertTrue(np.allclose([3.78030562e-06, -4.20475816e-06], inners, atol=1e-4))

    def testSimulationCovar(self):
        covar = self.sim.covar_true()
        self.assertTrue(
            np.allclose(
                [
                    [-2.88516425e-06, -5.83903949e-05, -1.89978754e-04, -1.24722391e-03,
                     -3.15552230e-05,  7.43356330e-03,  7.98143578e-03,  3.03416076e-03],
                    [-7.75685657e-06,  1.83711651e-04,  4.48674856e-03, -7.94970885e-03,
                     -2.98800201e-02, -1.85445526e-03,  1.78661267e-02,  6.85989776e-03],
                    [ 1.14370046e-05,  3.24029635e-03,  3.36405433e-02, -2.72520976e-03,
                     -8.97639252e-02, -5.40480774e-02,  2.68742911e-03, -3.08175925e-02],
                    [ 3.20402457e-05,  9.09853117e-03,  7.85994578e-02,  7.25429972e-02,
                     -1.93657403e-01, -9.00726550e-02, -1.57314658e-01, -1.56903155e-01],
                    [-4.05611099e-04,  6.85138898e-03,  1.10749890e-01,  3.52075781e-01,
                      1.72646707e-01, -1.66628913e-01, -1.50108727e-01, -1.42926613e-01],
                    [-1.07461651e-03, -4.97394612e-03,  4.63012620e-02,  3.80485863e-01,
                      4.79159054e-01,  5.37995975e-02, -1.18336775e-01, -3.37297508e-02],
                    [-2.96304261e-04, -4.85665009e-03, -6.40121867e-03,  2.20681883e-01,
                      1.54190501e-01,  8.28120514e-02,  3.37324333e-02,  1.03901452e-03],
                    [ 4.43231817e-04,  8.50533398e-03,  9.68386556e-02,  1.69595272e-01,
                      3.62910147e-02,  3.74060525e-02,  2.21235840e-02,  3.18127346e-03]
                ],
                covar[:, :, 4, 4, 4, 4]
            )
        )

    def testSimulationEvalMean(self):
        mean_est = np.load(os.path.join(DATA_DIR, 'mean_8_8_8.npy'))
        result = self.sim.eval_mean(mean_est)

        self.assertTrue(np.allclose(result['err'], 2.664116055950763, atol=1e-4))
        self.assertTrue(np.allclose(result['rel_err'], 0.1765943704851626, atol=1e-4))
        self.assertTrue(np.allclose(result['corr'], 0.9849211540734224, atol=1e-4))

    def testSimulationEvalCovar(self):
        covar_est = np.load(os.path.join(DATA_DIR, 'covar_8_8_8_8_8_8.npy'))
        result = self.sim.eval_covar(covar_est)

        self.assertTrue(np.allclose(result['err'], 13.322721549011165, atol=1e-4))
        self.assertTrue(np.allclose(result['rel_err'], 0.5958936073938558, atol=1e-4))
        self.assertTrue(np.allclose(result['corr'], 0.8405347287741631, atol=1e-4))

    def testSimulationEvalCoords(self):
        mean_est = np.load(os.path.join(DATA_DIR, 'mean_8_8_8.npy'))
        eigs_est = np.load(os.path.join(DATA_DIR, 'eigs_est_8_8_8_1.npy'))
        clustered_coords_est = np.load(os.path.join(DATA_DIR, 'clustered_coords_est.npy'))

        result = self.sim.eval_coords(mean_est, eigs_est, clustered_coords_est)

        self.assertTrue(np.allclose(
            result['err'][:10],
            [1.58382394, 1.58382394, 3.72076112, 1.58382394, 1.58382394,
             3.72076112, 3.72076112, 1.58382394, 1.58382394, 1.58382394]
        ))

        self.assertTrue(np.allclose(
            result['rel_err'][:10],
            [0.11048937, 0.11048937, 0.21684697, 0.11048937, 0.11048937,
             0.21684697, 0.21684697,0.11048937, 0.11048937, 0.11048937]
        ))

        self.assertTrue(np.allclose(
            result['corr'][:10],
            [0.99390133, 0.99390133, 0.97658719, 0.99390133, 0.99390133,
             0.97658719, 0.97658719, 0.99390133, 0.99390133, 0.99390133]
        ))
