import numpy as np
from unittest import TestCase

from aspyre.source import SourceFilter
from aspyre.source.simulation import Simulation
from aspyre.imaging.filters import RadialCTFFilter

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class SimTestCase(TestCase):
    def setUp(self):
        self.sim = Simulation(
            n=1024,
            L=8,
            filters=SourceFilter(
                filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
                n=1024
            )
        )

    def tearDown(self):
        pass

    def testGaussianBlob(self):
        blobs = self.sim.vols
        self.assertTrue(np.allclose(blobs, np.load(os.path.join(DATA_DIR, 'sim_blobs.npy'))))

    def testSimulationRots(self):
        self.assertTrue(np.allclose(
            self.sim.rots[:, :, 0],
            np.array([
                [0.91675498, 0.2587233, 0.30433956],
                [0.39941773, -0.58404652, -0.70665065],
                [-0.00507853, 0.76938412, -0.63876622]
            ])
        ))

    def testSimulationCleanImages(self):
        images = self.sim.images(0, 512)
        self.assertTrue(np.allclose(images, np.load(os.path.join(DATA_DIR, 'sim_clean_images.npy')), rtol=1e-2))

    def testSimulationImages(self):
        images = self.sim.images(0, 512, apply_noise=True)
        self.assertTrue(np.allclose(images, np.load(os.path.join(DATA_DIR, 'sim_images_with_noise.npy')), rtol=1e-2))

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
                [-1.67666201e-07, -7.95741380e-06, -1.49160041e-04, -1.10151654e-03, -3.11287888e-03, -3.09157884e-03, -9.91418026e-04, -1.31673165e-04],
                [-1.15402077e-06, -2.49849709e-05, -3.51658906e-04, -2.21575261e-03, -7.83315487e-03, -9.44795180e-03, -4.07636259e-03, -9.02186439e-04],
                [-1.88737249e-05, -1.91418396e-04, -1.09021540e-03, -1.02020288e-03,  1.39411855e-02,  8.58035963e-03, -5.54619730e-03, -3.86377703e-03],
                [-1.21280536e-04, -9.51461843e-04, -3.22565017e-03, -1.05731178e-03,  2.61375736e-02,  3.11595201e-02,  6.40814053e-03, -2.31698658e-02],
                [-2.44067283e-04, -1.40560151e-03, -6.73082832e-05,  1.44160679e-02,  2.99893934e-02,  5.92632964e-02,  7.75623545e-02,  3.06570008e-02],
                [-1.53507499e-04, -7.21709803e-04,  8.54929152e-04, -1.27235036e-02, -5.34382043e-03,  2.18879692e-02,  6.22706190e-02,  4.51998860e-02],
                [-3.00595184e-05, -1.43038429e-04, -2.15870258e-03, -9.99002904e-02, -7.79077187e-02, -1.53395887e-02,  1.88777559e-02,  1.68759506e-02],
                [ 3.22692649e-05,  4.07977635e-03,  1.63959339e-02, -8.68835449e-02, -7.86240026e-02, -1.75694861e-02,  3.24984640e-03,  1.95389288e-03]
            ])
        ))

    def testSimulationMean(self):
        mean_vol = self.sim.mean_true()
        self.assertTrue(
            np.allclose(
                [
                    [0.00000930, 0.00033866, 0.00490734, 0.01998369, 0.03874487, 0.04617764, 0.02970645, 0.00967604],
                    [0.00003904, 0.00247391, 0.03818476, 0.12325402, 0.22278425, 0.25246665, 0.14093882, 0.03683474],
                    [0.00014177, 0.01191146, 0.14421064, 0.38428235, 0.78645319, 0.86522675, 0.44862473, 0.16382280],
                    [0.00066036, 0.03137806, 0.29226971, 0.97105378, 2.39410496, 2.17099857, 1.23595858, 0.49233940],
                    [0.00271748, 0.05491289, 0.49955708, 2.05356097, 3.70941424, 3.01578689, 1.51441932, 0.52054572],
                    [0.00584845, 0.06962635, 0.50568032, 1.99643707, 3.77415895, 2.76039767, 1.04602003, 0.20633197],
                    [0.00539583, 0.06068972, 0.47008955, 1.17128026, 1.82821035, 1.18743944, 0.30667788, 0.04851476],
                    [0.00246362, 0.04867788, 0.65284950, 0.65238875, 0.65745538, 0.37955678, 0.08053055, 0.01210055],
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
                    [-0.00000289, -0.00005839, -0.00018998, -0.00124722, -0.00003155, +0.00743356, +0.00798143, +0.00303416],
                    [-0.00000776, +0.00018371, +0.00448675, -0.00794970, -0.02988000, -0.00185446, +0.01786612, +0.00685990],
                    [+0.00001144, +0.00324029, +0.03364052, -0.00272520, -0.08976389, -0.05404807, +0.00268740, -0.03081760],
                    [+0.00003204, +0.00909853, +0.07859941, +0.07254293, -0.19365733, -0.09007251, -0.15731451, -0.15690306],
                    [-0.00040561, +0.00685139, +0.11074986, +0.35207557, +0.17264650, -0.16662873, -0.15010859, -0.14292650],
                    [-0.00107461, -0.00497393, +0.04630126, +0.38048555, +0.47915877, +0.05379957, -0.11833663, -0.03372971],
                    [-0.00029630, -0.00485664, -0.00640120, +0.22068169, +0.15419035, +0.08281200, +0.03373241, +0.00103902],
                    [+0.00044323, +0.00850533, +0.09683860, +0.16959519, +0.03629097, +0.03740599, +0.02212356, +0.00318127],
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
            [1.58382394, 1.58382394, 3.72076112, 1.58382394, 1.58382394, 3.72076112, 3.72076112, 1.58382394, 1.58382394, 1.58382394]
        ))

        self.assertTrue(np.allclose(
            result['rel_err'][:10],
            [0.11048937, 0.11048937, 0.21684697, 0.11048937, 0.11048937, 0.21684697, 0.21684697,0.11048937, 0.11048937, 0.11048937]
        ))

        self.assertTrue(np.allclose(
            result['corr'][:10],
            [0.99390133, 0.99390133, 0.97658719, 0.99390133, 0.99390133, 0.97658719, 0.97658719, 0.99390133, 0.99390133, 0.99390133]
        ))
