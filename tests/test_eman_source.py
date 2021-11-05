import os
import pickle
import tempfile
from unittest import TestCase

import importlib_resources
import numpy as np

import tests.saved_test_data
from aspire.source import EmanSource


class EmanSourceTestCase(TestCase):
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

    def tearDown(self):
        self.tmpdir.cleanup()

    def testLoadFromBox(self):
        src = EmanSource([(self.mrc_path, self.box_fp)])
        # loaded 440 particles from the box file (lower left corner and heights and widths given)
        self.assertEqual(src.n, 440)

    def testLoadFromCoord(self):
        src = EmanSource(
            [(self.mrc_path, self.coord_fp)],
            centers=True,
            particle_size=256,
        )
        self.assertEqual(src.n, 440)

    def testLoadFromRelion(self):
        EmanSource(
            data_folder=self.data_folder,
            relion_autopick_star="sample_relion_autopick.star",
            particle_size=256,
        )

    def testLoadFromCoordWithoutCentersTrue(self):
        # if loading only centers (coord file), centers must be set to true
        with self.assertRaises(ValueError):
            EmanSource([(self.mrc_path, self.coord_fp)], particle_size=256)

    def testLoadFromCoordNoParticleSize(self):
        with self.assertRaises(AssertionError):
            EmanSource([(self.mrc_path, self.coord_fp)], centers=True)

    def testNonSquareParticles(self):
        # nonsquare box sizes must fail
        with self.assertRaises(ValueError):
            EmanSource(
                [(self.mrc_path, self.box_fp_nonsquare)],
            )

    def testImages(self):
        # load from both the box format and the coord format
        # ensure the images obtained are the same
        src_from_box = EmanSource([(self.mrc_path, self.box_fp)])
        src_from_coord = EmanSource(
            [(self.mrc_path, self.coord_fp)], particle_size=256, centers=True
        )
        src_from_relion = EmanSource(
            data_folder=self.data_folder,
            relion_autopick_star="sample_relion_autopick.star",
            particle_size=256,
        )
        imgs_box = src_from_box.images(0, 10)
        imgs_coord = src_from_coord.images(0, 10)
        imgs_star = src_from_relion.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_box[i], imgs_coord[i]))
            self.assertTrue(np.array_equal(imgs_coord[i], imgs_star[i]))
