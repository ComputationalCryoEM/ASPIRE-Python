import os
import pickle
import random
import shutil
import tempfile
from collections import OrderedDict
from glob import glob
from unittest import TestCase

import importlib_resources
import mrcfile
import numpy as np
from click.testing import CliRunner
from pandas import DataFrame

import tests.saved_test_data
from aspire.commands.extract_particles import extract_particles
from aspire.noise import WhiteNoiseEstimator
from aspire.source import (
    CentersCoordinateSource,
    BoxesCoordinateSource,
    RelionCoordinateSource,
)
from aspire.storage import StarFile


class CoordinateSourceTestCase(TestCase):
    def setUp(self):
        # temporary directory to set up a toy dataset
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_folder = self.tmpdir.name
        # load pickled list of picked particle centers
        with importlib_resources.path(
            tests.saved_test_data, "apple_centers.p"
        ) as centers_path:
            # apple_centers.p was pickled with protocol 4
            centers = pickle.load(
                open(str(centers_path), "rb"),
            )
        # get path to test .mrc file
        with importlib_resources.path(tests.saved_test_data, "sample.mrc") as test_path:
            self.original_mrc_path = str(test_path)
        # save test data root dir
        self.test_dir_root = os.path.dirname(self.original_mrc_path)

        # First set up tests for BoxesCoordinateSource and CentersCoordinateSource
        # We will construct a source with two micrographs and two coordinate
        # files by using the same micrograph, but dividing the coordinates
        # among two files (this simulates a dataset with multiple micrographs)
        for i in range(2):
            # copy mrc to temp location
            _new_mrc_path = os.path.join(self.data_folder, f"sample{i+1}.mrc")
            shutil.copyfile(self.original_mrc_path, _new_mrc_path)

            # get the first half of the coordinates for i=0 [0:220] and the
            # second half for i=1 [220:440]
            _centers = centers[i * 220 : (i + 1) * 220]

            # create a coord file (only particle centers listed)
            self.coord_fp = os.path.join(self.data_folder, f"sample{i+1}.coord")
            # create a star file (only particle centers listed)
            self.star_fp = os.path.join(self.data_folder, f"sample{i+1}.star")
            # create a box file (lower left corner and X/Y sizes)
            self.box_fp = os.path.join(self.data_folder, f"sample{i+1}.box")
            # box file with nonsquare particles
            self.box_fp_nonsquare = os.path.join(
                self.data_folder, f"nonsquare_sample{i+1}.box"
            )

            # populate coord file with particle centers
            with open(self.coord_fp, "w") as coord:
                for center in _centers:
                    # .coord file usually contains just the centers
                    coord.write(f"{center[0]}\t{center[1]}\n")

            # populate star file with particle centers
            x_coords = [center[0] for center in _centers]
            y_coords = [center[1] for center in _centers]
            blocks = OrderedDict(
                {
                    "coordinates": DataFrame(
                        {"_rlnCoordinateX": x_coords, "_rlnCoordinateY": y_coords}
                    )
                }
            )
            starfile = StarFile(blocks=blocks)
            starfile.write(self.star_fp)

            # populate box file with coordinates in box format
            with open(self.box_fp, "w") as box:
                for center in _centers:
                    # to make a box file, we convert the centers to lower left
                    # corners by subtracting half the particle size (here: 256)
                    lower_left_corners = (center[0] - 128, center[1] - 128)
                    box.write(
                        f"{lower_left_corners[0]}\t{lower_left_corners[1]}\t256\t256\n"
                    )
            # populate the second box file with nonsquare coordinates
            with open(self.box_fp_nonsquare, "w") as box_nonsquare:
                for center in _centers:
                    # make a bad box file with non square particles
                    lower_left_corners = (center[0] - 128, center[1] - 128)
                    box_nonsquare.write(
                        f"{lower_left_corners[0]}\t{lower_left_corners[1]}\t256\t100\n"
                    )

        # Now construct a simulated Relion directory
        # to test RelionCoordinateSource
        # The autopick.star file points to sample1.mrc
        # (created earlier) and relion_coord.star as the coord file
        os.makedirs(os.path.join(self.data_folder, "AutoPick/job006/"))
        self.autopick_star_path = os.path.join(
            self.data_folder, "AutoPick/job006/sample_relion_autopick.star"
        )
        shutil.copyfile(
            os.path.join(self.test_dir_root, "sample_relion_autopick.star"),
            self.autopick_star_path,
        )
        shutil.copyfile(
            os.path.join(self.test_dir_root, "sample_relion_coord.star"),
            # renamed because later we glob for sample*star and we don't want this one
            os.path.join(self.data_folder, "relion_coord.star"),
        )

        # create default object from a .box file, for comparisons in tests
        # also provides an example of how one might use this in a script
        self.all_mrc_paths = sorted(glob(self.data_folder + "/*.mrc"))
        self.all_box_paths = sorted(glob(self.data_folder + "/sample*.box"))
        self.files_box = list(zip(self.all_mrc_paths, self.all_box_paths))
        self.src_from_box = BoxesCoordinateSource(self.files_box)
        # create file lists that will be used several times
        self.files_coord = list(
            zip(self.all_mrc_paths, sorted(glob(self.data_folder + "/sample*.coord")))
        )
        self.files_star = list(
            zip(self.all_mrc_paths, sorted(glob(self.data_folder + "/sample*.star")))
        )
        self.files_box_nonsquare = list(
            zip(self.all_mrc_paths, sorted(glob(self.data_folder + "/nonsquare*.box")))
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def testLoadFromCoord(self):
        # ensure successful loading from particle center files (.coord)
        CentersCoordinateSource(self.files_coord, particle_size=256)

    def testLoadFromStar(self):
        # ensure successful loading from particle center files (.star)
        CentersCoordinateSource(self.files_star, particle_size=256)

    def testLoadFromRelionAutoPick(self):
        # Here we test loading from a Relion project directory's STAR
        # index file.
        RelionCoordinateSource(
            os.path.join(self.autopick_star_path),
            particle_size=256,
        )

    def testNonSquareParticles(self):
        # nonsquare box sizes must fail
        with self.assertRaises(ValueError):
            BoxesCoordinateSource(self.files_box_nonsquare)

    def testOverrideParticleSize(self):
        # it is possible to override the particle size in the box file
        src_new_size = BoxesCoordinateSource(self.files_box, particle_size=100)
        src_from_centers = CentersCoordinateSource(self.files_coord, particle_size=100)
        imgs_new_size = src_new_size.images(0, 10)
        imgs_from_centers = src_from_centers.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_new_size[i], imgs_from_centers[i]))

    def testImages(self):
        # load from both the box format and the coord format
        # ensure the images obtained are the same
        src_from_coord = CentersCoordinateSource(self.files_coord, particle_size=256)
        src_from_star = CentersCoordinateSource(self.files_star, particle_size=256)
        src_from_relion = RelionCoordinateSource(
            self.autopick_star_path,
            particle_size=256,
        )
        imgs_box = self.src_from_box.images(0, 10)
        imgs_coord = src_from_coord.images(0, 10)
        imgs_star = src_from_star.images(0, 10)
        imgs_relion = src_from_relion.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_box[i], imgs_coord[i]))
            self.assertTrue(np.array_equal(imgs_coord[i], imgs_star[i]))
            self.assertTrue(np.array_equal(imgs_star[i], imgs_relion[i]))

    def testImagesRandomIndices(self):
        # ensure that we can load a specific, possibly out of order, list of
        # indices, and that the result is in the order we asked for
        images_in_order = self.src_from_box.images(0, 440)
        # test loading every other image and compare
        odd = np.array([i for i in range(1, 440, 2)])
        even = np.array([i for i in range(0, 439, 2)])
        odd_images = self.src_from_box._images(indices=odd)
        even_images = self.src_from_box._images(indices=even)
        for i in range(0, 220):
            self.assertTrue(np.array_equal(images_in_order[2 * i], even_images[i]))
            self.assertTrue(np.array_equal(images_in_order[2 * i + 1], odd_images[i]))

        # random sample of [0,440) of length 100
        random_sample = np.array(random.sample([i for i in range(440)], 100))
        random_images = self.src_from_box._images(indices=random_sample)
        for i, idx in enumerate(random_sample):
            self.assertTrue(np.array_equal(images_in_order[idx], random_images[i]))

    def testMaxRows(self):
        imgs = self.src_from_box.images(0, 440)
        # make sure max_rows loads the correct particles
        src_100 = BoxesCoordinateSource(self.files_box, max_rows=100)
        imgs_100 = src_100.images(0, src_100.n)
        for i in range(100):
            self.assertTrue(np.array_equal(imgs[i], imgs_100[i]))
        # make sure max_rows > self.n loads max_rows images
        src_500 = BoxesCoordinateSource(self.files_box, max_rows=500)
        self.assertEqual(src_500.n, 440)
        imgs_500 = src_500.images(0, 440)
        for i in range(440):
            self.assertTrue(np.array_equal(imgs[i], imgs_500[i]))
        # make sure max_rows loads correct particles
        # when some have been excluded
        imgs_newsize = BoxesCoordinateSource(self.files_box, particle_size=336).images(
            0, 50
        )
        src_maxrows = BoxesCoordinateSource(
            self.files_box, particle_size=336, max_rows=50
        )
        # max_rows still loads 50 images even if some particles were excluded
        self.assertEqual(src_maxrows.n, 50)
        imgs_maxrows = src_maxrows.images(0, 50)
        for i in range(50):
            self.assertTrue(np.array_equal(imgs_newsize[i], imgs_maxrows[i]))

    def testBoundaryParticlesRemoved(self):
        src_centers_larger_particles = CentersCoordinateSource(
            self.files_coord, particle_size=300
        )
        src_box_larger_particles = BoxesCoordinateSource(
            self.files_box, particle_size=300
        )
        # 2 particles do not fit at this particle size
        self.assertEqual(src_centers_larger_particles.n, 438)
        self.assertEqual(src_box_larger_particles.n, 438)
        # make sure we have the same particles
        imgs_centers = src_centers_larger_particles.images(0, 438)
        imgs_resized = src_box_larger_particles.images(0, 438)
        for i in range(50):
            self.assertTrue(np.array_equal(imgs_centers[i], imgs_resized[i]))

    def testEvenOddResize(self):
        # test a range of even and odd resizes
        for _size in range(252, 260):
            src_centers = CentersCoordinateSource(self.files_coord, particle_size=_size)
            src_resized = BoxesCoordinateSource(self.files_box, particle_size=_size)
            imgs_centers = src_centers.images(0, 440)
            imgs_resized = src_resized.images(0, 440)
            for i in range(440):
                self.assertTrue(np.array_equal(imgs_centers[i], imgs_resized[i]))

    def testSave(self):
        # we can save the source into an .mrcs stack with *no* metadata
        src = BoxesCoordinateSource(self.files_box, max_rows=10)
        imgs = src.images(0, 10)
        star_path = os.path.join(self.tmpdir.name, "stack.star")
        mrcs_path = os.path.join(self.tmpdir.name, "stack_0_9.mrcs")
        src.save(star_path)
        # load saved particle stack
        saved_mrcs_stack = mrcfile.open(mrcs_path).data
        saved_star = StarFile(star_path)
        # assert that the particles saved are correct
        for i in range(10):
            self.assertTrue(np.array_equal(imgs[i], saved_mrcs_stack[i]))
        # assert that the star file has no metadata: the only col is _rlnImageName
        self.assertEqual(list(saved_star[""].columns), ["_rlnImageName"])

    def testPreprocessing(self):
        # ensure that the preprocessing methods that do not require CTF do not error
        src = BoxesCoordinateSource(self.files_box, max_rows=5)
        src.downsample(60)
        src.normalize_background()
        noise_estimator = WhiteNoiseEstimator(src)
        src.whiten(noise_estimator.filter)
        src.invert_contrast()
        # call .images() to ensure the filters are applied
        # and not just added to pipeline
        src.images(0, 5)

    def testCommand(self):
        # ensure that the command line tool works as expected
        runner = CliRunner()
        result_box = runner.invoke(
            extract_particles,
            [
                f"--mrc_paths={self.data_folder}/*.mrc",
                f"--coord_paths={self.data_folder}/sample*.box",
                f"--starfile_out={self.data_folder}/saved_box.star",
            ],
        )
        result_coord = runner.invoke(
            extract_particles,
            [
                f"--mrc_paths={self.data_folder}/*.mrc",
                f"--coord_paths={self.data_folder}/sample*.coord",
                f"--starfile_out={self.data_folder}/saved_coord.star",
                "--centers",
                "--particle_size=256",
            ],
        )
        result_star = runner.invoke(
            extract_particles,
            [
                f"--mrc_paths={self.data_folder}/*.mrc",
                f"--coord_paths={self.data_folder}/sample*.star",
                f"--starfile_out={self.data_folder}/saved_star.star",
                "--centers",
                "--particle_size=256",
            ],
        )
        # check that all commands completed successfully
        self.assertTrue(result_box.exit_code == 0)
        self.assertTrue(result_coord.exit_code == 0)
        self.assertTrue(result_star.exit_code == 0)
