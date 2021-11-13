import os
import pickle
import tempfile
from unittest import TestCase

import importlib_resources
import mrcfile
import numpy as np

import tests.saved_test_data
from aspire.noise import WhiteNoiseEstimator
from aspire.source.from_coordinates import ParticleCoordinateSource
from aspire.storage import StarFile


class ParticleCoordinateSourceTestCase(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        # load pickled list of picked particle centers
        with importlib_resources.path(
            tests.saved_test_data, "apple_centers.p"
        ) as centers_path:
            centers = pickle.load(open(str(centers_path), "rb"))

        # get path to test .mrc file
        with importlib_resources.path(tests.saved_test_data, "sample.mrc") as test_path:
            self.mrc_path = str(test_path)
        # get saved_test_data dir path as data_folder
        self.data_folder = os.path.dirname(self.mrc_path)

        # create a coord file (only centers listed)
        self.coord_fp = os.path.join(self.tmpdir.name, "sample.coord")
        # create a box file (lower left corner as well as X/Y dims of particle
        self.box_fp = os.path.join(self.tmpdir.name, "sample.box")
        # box file with nonsquare particles
        self.box_fp_nonsquare = os.path.join(self.tmpdir.name, "sample_nonsquare.box")
        # populate box and coord files
        with open(self.coord_fp, "w") as coord:
            for center in centers:
                # .coord file usually contains just the centers
                coord.write(f"{center[0]}\t{center[1]}\n")
        with open(self.box_fp, "w") as box:
            for center in centers:
                # to make a box file, we convert the centers to lower left
                # corners by subtracting half the particle size (here, 256).
                lower_left_corners = (center[0] - 128, center[1] - 128)
                box.write(
                    f"{lower_left_corners[0]}\t{lower_left_corners[1]}\t256\t256\n"
                )
        with open(self.box_fp_nonsquare, "w") as box_nonsquare:
            for center in centers:
                # make a bad box file with non square particles
                lower_left_corners = (center[0] - 128, center[1] - 128)
                box_nonsquare.write(
                    f"{lower_left_corners[0]}\t{lower_left_corners[1]}\t256\t100\n"
                )

        # default object from a .box file, for comparisons in multiple tests
        self.src_from_box = ParticleCoordinateSource([(self.mrc_path, self.box_fp)])

    def tearDown(self):
        self.tmpdir.cleanup()

    def testLoadFromCoord(self):
        src = ParticleCoordinateSource(
            [(self.mrc_path, self.coord_fp)],
            centers=True,
            particle_size=256,
        )
        self.assertEqual(src.n, 440)

    def testLoadFromRelion(self):
        ParticleCoordinateSource(
            data_folder=self.data_folder,
            relion_autopick_star="sample_relion_autopick.star",
            particle_size=256,
        )

    def testLoadFromCoordWithoutCentersTrue(self):
        # if loading only centers (coord file), centers must be set to true
        with self.assertRaises(ValueError):
            ParticleCoordinateSource(
                [(self.mrc_path, self.coord_fp)], particle_size=256
            )

    def testLoadFromCoordNoParticleSize(self):
        with self.assertRaises(ValueError):
            ParticleCoordinateSource([(self.mrc_path, self.coord_fp)], centers=True)

    def testNonSquareParticles(self):
        # nonsquare box sizes must fail
        with self.assertRaises(ValueError):
            ParticleCoordinateSource(
                [(self.mrc_path, self.box_fp_nonsquare)],
            )

    def testDataFolderMismatch(self):
        # our sample.mrc is located in saved_test_data
        # if we give an absolute path data_folder, and the dirnames do not match
        # there should be an error due to the ambiguity
        with self.assertRaises(ValueError):
            ParticleCoordinateSource(
                [(self.mrc_path, self.box_fp)], data_folder=self.tmpdir.name
            )

    def testOverrideParticleSize(self):
        # it is possible to override the particle size in the box file
        src_new_size = ParticleCoordinateSource(
            [(self.mrc_path, self.box_fp)], particle_size=100
        )
        src_from_centers = ParticleCoordinateSource(
            [(self.mrc_path, self.coord_fp)], centers=True, particle_size=100
        )
        imgs_new_size = src_new_size.images(0, 10)
        imgs_from_centers = src_from_centers.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_new_size[i], imgs_from_centers[i]))

    def testImages(self):
        # load from both the box format and the coord format
        # ensure the images obtained are the same
        src_from_coord = ParticleCoordinateSource(
            [(self.mrc_path, self.coord_fp)], particle_size=256, centers=True
        )
        src_from_relion = ParticleCoordinateSource(
            data_folder=self.data_folder,
            relion_autopick_star="sample_relion_autopick.star",
            particle_size=256,
        )
        imgs_box = self.src_from_box.images(0, 10)
        imgs_coord = src_from_coord.images(0, 10)
        imgs_star = src_from_relion.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_box[i], imgs_coord[i]))
            self.assertTrue(np.array_equal(imgs_coord[i], imgs_star[i]))

    def testMaxRows(self):
        # make sure max_rows loads the correct particles
        src_only100 = ParticleCoordinateSource(
            [(self.mrc_path, self.box_fp)], max_rows=100
        )
        imgs = self.src_from_box.images(0, 440)
        only100imgs = src_only100.images(0, src_only100.n)
        for i in range(100):
            self.assertTrue(np.array_equal(imgs[i], only100imgs[i]))

    def testSave(self):
        # we can save the source into an .mrcs stack with *no* metadata
        src = ParticleCoordinateSource([(self.mrc_path, self.box_fp)], max_rows=10)
        imgs = src.images(0, 10)
        star_path = os.path.join(self.tmpdir.name, "stack.star")
        mrcs_path = os.path.join(self.tmpdir.name, "stack_0_9.mrcs")
        src.save(star_path)
        saved_mrc = mrcfile.open(mrcs_path).data
        saved_star = StarFile(star_path)
        # assert that the particles saved are correct
        for i in range(10):
            self.assertTrue(np.array_equal(imgs[i], saved_mrc[i]))
        # assert that the star file has no metadata: the only col is _rlnImageName
        self.assertEqual(list(saved_star[""].columns), ["_rlnImageName"])

    def testPreprocessing(self):
        # ensure that the preprocessing methods that do not require CTF do not error
        src = ParticleCoordinateSource([(self.mrc_path, self.box_fp)], max_rows=10)
        src.downsample(60)
        src.normalize_background()
        noise_estimator = WhiteNoiseEstimator(src)
        src.whiten(noise_estimator.filter)
        src.invert_contrast()
