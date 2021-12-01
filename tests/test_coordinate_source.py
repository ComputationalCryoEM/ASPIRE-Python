import os
import pickle
import shutil
import tempfile
from glob import glob
from random import shuffle
from unittest import TestCase

import importlib_resources
import mrcfile
import numpy as np

import tests.saved_test_data
from aspire.noise import WhiteNoiseEstimator
from aspire.source.coordinates import CoordinateSource
from aspire.storage import StarFile


class ParticleCoordinateSourceTestCase(TestCase):
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
        # save test data root dir (for Relion and data_folder tests)
        self.test_dir_root = os.path.dirname(self.original_mrc_path)

        # We will construct a source with two micrographs and two coordinate
        # files by using the same micrograph, but dividing the coordinates
        # among two files
        for i in range(2):
            # copy mrc to temp location
            _new_mrc_path = os.path.join(self.data_folder, f"sample{i+1}.mrc")
            shutil.copyfile(self.original_mrc_path, _new_mrc_path)

            # get the first half of the coordinates for i=0 [0:220] and the
            # second half for i=1 [220:440]
            _centers = centers[i * 220 : (i + 1) * 220]

            # create a coord file (only particle centers listed)
            self.coord_fp = os.path.join(self.data_folder, f"sample{i+1}.coord")
            # create a box file (lower left corner and X/Y sizes)
            self.box_fp = os.path.join(self.tmpdir.name, f"sample{i+1}.box")
            # box file with nonsquare particles
            self.box_fp_nonsquare = os.path.join(
                self.tmpdir.name, f"sample_nonsquare{i+1}.box"
            )

            # populate coord file with particle centers
            with open(self.coord_fp, "w") as coord:
                for center in _centers:
                    # .coord file usually contains just the centers
                    coord.write(f"{center[0]}\t{center[1]}\n")
            # populate box file with coordinates in EMAN1 .box format
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

        # create default object from a .box file, for comparisons in tests
        # also provides an example of how one might use this in a script
        self.all_mrc_paths = sorted(glob(self.data_folder + "/*.mrc"))
        self.all_box_paths = sorted(glob(self.data_folder + "/*.box"))
        self.files_box = list(zip(self.all_mrc_paths, self.all_box_paths))
        self.src_from_box = CoordinateSource(self.files_box)
        # create file lists that will be used several times
        self.files_coord = list(
            zip(self.all_mrc_paths, sorted(glob(self.data_folder + "/*.coord")))
        )
        self.files_box_nonsquare = list(
            zip(self.all_mrc_paths, sorted(glob(self.data_folder + "/*nonsquare*.box")))
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def testLoadFromCoord(self):
        # ensure successful loading from particle center files (.coord)
        src = CoordinateSource(
            self.files_coord,
            centers=True,
            particle_size=256,
        )
        self.assertEqual(src.n, 440)

    def testLoadFromRelion(self):
        CoordinateSource(
            relion_autopick_star=os.path.join(
                self.test_dir_root, "AutoPick/job006/sample_relion_autopick.star"
            ),
            particle_size=256,
        )

    def testLoadFromCoordWithoutCentersTrue(self):
        # if loading only centers (coord file), centers must be set to true
        with self.assertRaises(ValueError):
            CoordinateSource(self.files_coord, particle_size=256)

    def testLoadFromCoordNoParticleSize(self):
        # if loading only centers (coord file), particle_size must be specified
        with self.assertRaises(ValueError):
            CoordinateSource(self.files_coord, centers=True)

    def testNonSquareParticles(self):
        # nonsquare box sizes must fail
        with self.assertRaises(ValueError):
            CoordinateSource(self.files_box_nonsquare)

    def testDataFolderMismatch(self):
        # our sample.mrc is located in a tempdir
        # if we give an absolute path data_folder, and the dirnames don't match
        # there should be an error due to the ambiguity
        with self.assertRaises(ValueError):
            CoordinateSource(self.files_coord, data_folder=self.test_dir_root)

    def testOverrideParticleSize(self):
        # it is possible to override the particle size in the box file
        src_new_size = CoordinateSource(self.files_box, particle_size=100)
        src_from_centers = CoordinateSource(
            self.files_coord,
            centers=True,
            particle_size=100,
        )
        imgs_new_size = src_new_size.images(0, 10)
        imgs_from_centers = src_from_centers.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_new_size[i], imgs_from_centers[i]))

    def testImages(self):
        # load from both the box format and the coord format
        # ensure the images obtained are the same
        src_from_coord = CoordinateSource(
            self.files_coord,
            particle_size=256,
            centers=True,
        )
        src_from_relion = CoordinateSource(
            relion_autopick_star=os.path.join(
                self.test_dir_root, "AutoPick/job006/sample_relion_autopick.star"
            ),
            particle_size=256,
        )
        imgs_box = self.src_from_box.images(0, 10)
        imgs_coord = src_from_coord.images(0, 10)
        imgs_star = src_from_relion.images(0, 10)
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_box[i], imgs_coord[i]))
            self.assertTrue(np.array_equal(imgs_coord[i], imgs_star[i]))

    def testImagesRandomIndices(self):
        # ensure that we can load a specific, possibly out of order, list of
        # indices, and that the result is in the order we asked for
        _indices = [i for i in range(440)]
        shuffle(_indices)
        images_in_order = self.src_from_box.images(0, 440)
        random_order = self.src_from_box._images(indices=np.array(_indices))
        for i, idx in enumerate(_indices):
            assert np.array_equal(images_in_order[idx], random_order[i])

    def testMaxRows(self):
        # make sure max_rows loads the correct particles
        src_only100 = CoordinateSource(self.files_box, max_rows=100)
        imgs = self.src_from_box.images(0, 440)
        only100imgs = src_only100.images(0, src_only100.n)
        for i in range(100):
            self.assertTrue(np.array_equal(imgs[i], only100imgs[i]))

    def testSave(self):
        # we can save the source into an .mrcs stack with *no* metadata
        src = CoordinateSource(self.files_box, max_rows=10)
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
        src = CoordinateSource(self.files_box, max_rows=5)
        src.downsample(60)
        src.normalize_background()
        noise_estimator = WhiteNoiseEstimator(src)
        src.whiten(noise_estimator.filter)
        src.invert_contrast()
        # call .images() to ensure the filters are applied
        # and not just added to pipeline
        src.images(0, 5)
