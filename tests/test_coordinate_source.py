import importlib.resources
import os
import random
import shutil
import tempfile
from collections import OrderedDict
from glob import glob
from unittest import TestCase

import mrcfile
import numpy as np
from click.testing import CliRunner
from pandas import DataFrame

import tests.saved_test_data
from aspire.commands.extract_particles import extract_particles
from aspire.noise import WhiteNoiseEstimator
from aspire.source import BoxesCoordinateSource, CentersCoordinateSource
from aspire.storage import StarFile


class CoordinateSourceTestCase(TestCase):
    def setUp(self):
        # temporary directory to set up a toy dataset
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_folder = self.tmpdir.name
        # get path to test .mrc file
        with importlib.resources.path(tests.saved_test_data, "sample.mrc") as test_path:
            self.original_mrc_path = str(test_path)
        # save test data root dir
        self.test_dir_root = os.path.dirname(self.original_mrc_path)

        # We will construct a source with two micrographs and two coordinate
        # files by using the same micrograph, but dividing the coordinates
        # among two files (this simulates a dataset with multiple micrographs)

        # generate random particle centers
        centers = []
        # randomly generate 300 particles that fit on the micrograph with box size
        # 256 AND box size 300 (these will be included for particle_size <= 300)
        num_particles_fit = 300
        # randomly generate an additional 100 particles that fit with box size 256
        # but not with box size 300 (these will be excluded with the particle_size
        # is set to 300)
        num_particles_excl = 100

        # get micrograph dimensions
        with mrcfile.open(self.original_mrc_path) as mrc:
            shape_y, shape_x = mrc.data.shape
        for _i in range(num_particles_fit):
            # particle center must be at least half of
            # the max particle_size into the micrograph
            # in this case, the particles must fit into
            # a box of at least size 300
            x = random.choice([j for j in range(150, shape_x - 150)])
            y = random.choice([j for j in range(150, shape_y - 150)])
            centers.append((x, y))

        for _i in range(num_particles_excl):
            # now the particles must fit into a 256x256 box
            # but NOT into a 300x300 box
            x = random.choice(
                [j for j in range(128, 150)]
                + [j for j in range(shape_x - 150, shape_x - 128)]
            )
            y = random.choice(
                [j for j in range(128, 150)]
                + [j for j in range(shape_y - 150, shape_y - 128)]
            )
            centers.append((x, y))

        # randomize order of centers for when they are written to files
        random.shuffle(centers)

        for i in range(2):
            # copy mrc to temp location
            _new_mrc_path = os.path.join(self.data_folder, f"sample{i+1}.mrc")
            shutil.copyfile(self.original_mrc_path, _new_mrc_path)

            # get the first half of the coordinates for i=0 [0:200] and the
            # second half for i=1 [200:400]
            _centers = centers[i * 200 : (i + 1) * 200]

            self.createTestBoxFiles(_centers, i)
            self.createTestCoordFiles(_centers, i)
            self.createTestStarFiles(_centers, i)

            # create sample CTF STAR files
            self.createTestCtfFiles(i)

        # Create extra coordinate files with float
        # coordinates to make sure we can process these
        # as well
        self.createFloatBoxFile(centers)
        self.createFloatCoordFile(centers)
        self.createFloatStarFile(centers)

        # create lists of files
        self.all_mrc_paths = sorted(glob(os.path.join(self.data_folder, "sample*.mrc")))
        # create file lists that will be used several times
        self.files_box = list(
            zip(
                self.all_mrc_paths,
                sorted(glob(os.path.join(self.data_folder, "sample*.box"))),
            )
        )
        self.files_coord = list(
            zip(
                self.all_mrc_paths,
                sorted(glob(os.path.join(self.data_folder, "sample*.coord"))),
            )
        )
        self.files_star = list(
            zip(
                self.all_mrc_paths,
                sorted(glob(os.path.join(self.data_folder, "sample*.star"))),
            )
        )
        self.files_box_nonsquare = list(
            zip(
                self.all_mrc_paths,
                sorted(glob(os.path.join(self.data_folder, "nonsquare*.box"))),
            )
        )

        self.float_box = os.path.join(self.data_folder, "float.box")
        self.float_coord = os.path.join(self.data_folder, "float.coord")
        self.float_star = os.path.join(self.data_folder, "float.star")

        self.ctf_files = sorted(glob(os.path.join(self.data_folder, "ctf*.star")))
        self.relion_ctf_file = self.createTestRelionCtfFile()

    def tearDown(self):
        self.tmpdir.cleanup()

    def createTestBoxFiles(self, centers, index):
        """
        Create a .box file storing particle coordinates as
        lower left corner and X/Y sizes.
        :param centers: A list of tuples containing the centers of the particles.
        :param index: The number appended to the end of the temporary file's name.
        """
        box_fp = os.path.join(self.data_folder, f"sample{index+1}.box")
        # box file with nonsquare particles
        box_fp_nonsquare = os.path.join(
            self.data_folder, f"nonsquare_sample{index+1}.box"
        )
        # populate box file with coordinates in box format
        with open(box_fp, "w") as box:
            for center in centers:
                # to make a box file, we convert the centers to lower left
                # corners by subtracting half the particle size (here: 256)
                lower_left_corners = (center[0] - 128, center[1] - 128)
                box.write(
                    f"{lower_left_corners[0]}\t{lower_left_corners[1]}\t256\t256\n"
                )
        # populate the second box file with nonsquare coordinates
        with open(box_fp_nonsquare, "w") as box_nonsquare:
            for center in centers:
                # make a bad box file with non square particles
                lower_left_corners = (center[0] - 128, center[1] - 128)
                box_nonsquare.write(
                    f"{lower_left_corners[0]}\t{lower_left_corners[1]}\t256\t100\n"
                )

    def createTestCoordFiles(self, centers, index):
        """
        Create a .coord file storing particle coordinates as X/Y centers.
        :param centers: A list of tuples containing the centers of the particles.
        :param index: The number appended to the end of the temporary file's name.
        """
        coord_fp = os.path.join(self.data_folder, f"sample{index+1}.coord")
        # populate coord file with particle centers
        with open(coord_fp, "w") as coord:
            for center in centers:
                # .coord file usually contains just the centers
                coord.write(f"{center[0]}\t{center[1]}\n")

    def createTestStarFiles(self, centers, index):
        """
        Create a .star file storing particle coordinates as X/Y centers under
        'rlnCoordinateX' and 'rlnCoordinateY' columns.
        :param centers: A list of tuples containing the centers of the particles.
        :param index: The number appended to the end of the temporary file's name.
        """
        star_fp = os.path.join(self.data_folder, f"sample{index+1}.star")
        # populate star file with particle centers
        x_coords = [center[0] for center in centers]
        y_coords = [center[1] for center in centers]
        blocks = OrderedDict(
            {
                "coordinates": DataFrame(
                    {"_rlnCoordinateX": x_coords, "_rlnCoordinateY": y_coords}
                )
            }
        )
        starfile = StarFile(blocks=blocks)
        starfile.write(star_fp)

    def createFloatBoxFile(self, centers):
        """
        Create a .box file storing particle coordinates as
        lower left corner and X/Y sizes. This file will save coordinates as
        floats to test CoordinateSource's parsing.
        :param centers: A list of tuples containing the centers of the particles.
        :param index: The number appended to the end of the temporary file's name.
        """
        box_fp = os.path.join(self.data_folder, "float.box")
        # populate box file with coordinates in box format
        with open(box_fp, "w") as box:
            for center in centers:
                # to make a box file, we convert the centers to lower left
                # corners by subtracting half the particle size (here: 256)
                lower_left_corners = (center[0] - 128, center[1] - 128)
                box.write(
                    f"{lower_left_corners[0]}.000\t{lower_left_corners[1]}.000\t256.000\t256.000\n"
                )

    def createFloatCoordFile(self, centers):
        """
        Create a .coord file storing particle coordinates as X/Y centers.
        This file will save coordinates as floats to test CoordinateSource's parsing.
        :param centers: A list of tuples containing the centers of the particles.
        :param index: The number appended to the end of the temporary file's name.
        """
        coord_fp = os.path.join(self.data_folder, "float.coord")
        # populate coord file with particle centers
        with open(coord_fp, "w") as coord:
            for center in centers:
                # .coord file usually contains just the centers
                coord.write(f"{center[0]}.000\t{center[1]}.000\n")

    def createFloatStarFile(self, centers):
        """
        Create a .star file storing particle coordinates as X/Y centers under
        'rlnCoordinateX' and 'rlnCoordinateY' columns. This file will save coordinates as
        floats to test CoordinateSource's parsing.
        :param centers: A list of tuples containing the centers of the particles.
        :param index: The number appended to the end of the temporary file's name.
        """
        star_fp = os.path.join(self.data_folder, "float.star")
        # populate star file with particle centers
        x_coords = [str(center[0]) + ".000" for center in centers]
        y_coords = [str(center[1]) + ".000" for center in centers]
        blocks = OrderedDict(
            {
                "coordinates": DataFrame(
                    {"_rlnCoordinateX": x_coords, "_rlnCoordinateY": y_coords}
                )
            }
        )
        starfile = StarFile(blocks=blocks)
        starfile.write(star_fp)

    def createTestCtfFiles(self, index):
        """
        Creates example ASPIRE-generated CTF files.
        """
        star_fp = os.path.join(self.data_folder, f"ctf{index+1}.star")
        # note that values are arbitrary and not representative of actual CTF data
        params_dict = {
            "_rlnMicrographName": f"sample{index+1}.mrc",
            "_rlnDefocusU": 1000 + index,
            "_rlnDefocusV": 900 + index,
            "_rlnDefocusAngle": 800 + index,
            "_rlnSphericalAberration": 700 + index,
            "_rlnAmplitudeContrast": 600 + index,
            "_rlnVoltage": 500 + index,
            "_rlnDetectorPixelSize": 400 + index,
        }
        blocks = OrderedDict(
            {"root": DataFrame([params_dict], columns=params_dict.keys())}
        )
        starfile = StarFile(blocks=blocks)
        starfile.write(star_fp)

    def createTestRelionCtfFile(self):
        """
        Creates example RELION-generated CTF file for a set of micrographs.
        """
        star_fp = os.path.join(self.data_folder, "micrographs_ctf.star")
        blocks = OrderedDict()

        optics_columns = [
            "_rlnOpticsGroupName",
            "_rlnOpticsGroup",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnMicrographPixelSize",
        ]
        # using same unique values as in createTestCtfFiles
        optics_block = [
            ["opticsGroup1", 1, 500.0, 700.0, 600.0, 400.0],
            ["opticsGroup2", 2, 501.0, 701.0, 601.0, 401.0],
        ]
        blocks["optics"] = DataFrame(optics_block, columns=optics_columns)

        micrographs_columns = [
            "_rlnMicrographName",
            "_rlnOpticsGroup",
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
        ]
        # using same unique values as in createTestCtfFiles
        micrographs_block = [
            [self.all_mrc_paths[0], 1, 1000.0, 900.0, 800.0],
            [self.all_mrc_paths[0], 2, 1001.0, 901.0, 801.0],
        ]
        blocks["micrographs"] = DataFrame(
            micrographs_block, columns=micrographs_columns
        )

        star = StarFile(blocks=blocks)
        star.write(star_fp)
        return star_fp

    def testLoadFromBox(self):
        # ensure successful loading from box files
        BoxesCoordinateSource(self.files_box)

    def testLoadFromCenters(self):
        # ensure successful loading from particle center files (.coord)
        CentersCoordinateSource(self.files_coord, particle_size=256)

    def testLoadFromStar(self):
        # ensure successful loading from particle center files (.star)
        CentersCoordinateSource(self.files_star, particle_size=256)

    def testLoadFromBox_Floats(self):
        # ensure successful loading from box files with float coordinates
        BoxesCoordinateSource([(self.all_mrc_paths[0], self.float_box)])

    def testLoadFromCenters_Floats(self):
        # ensure successful loading from particle center files (.coord)
        # with float coordinates
        CentersCoordinateSource(
            [(self.all_mrc_paths[0], self.float_coord)], particle_size=256
        )

    def testLoadFromStar_Floats(self):
        # ensure successful loading from particle center files (.star)
        # with float coordinates
        CentersCoordinateSource(
            [(self.all_mrc_paths[0], self.float_star)], particle_size=256
        )

    def testNonSquareParticles(self):
        # nonsquare box sizes must fail
        with self.assertRaises(ValueError):
            BoxesCoordinateSource(self.files_box_nonsquare)

    def testOverrideParticleSize(self):
        # it is possible to override the particle size in the box file
        src_new_size = BoxesCoordinateSource(self.files_box, particle_size=100)
        src_from_centers = CentersCoordinateSource(self.files_coord, particle_size=100)
        imgs_new_size = src_new_size.images[:10]
        imgs_from_centers = src_from_centers.images[:10]
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_new_size[i], imgs_from_centers[i]))

    def testImages(self):
        # load from both the box format and the coord format
        # ensure the images obtained are the same
        src_from_box = BoxesCoordinateSource(self.files_box)
        src_from_coord = CentersCoordinateSource(self.files_coord, particle_size=256)
        src_from_star = CentersCoordinateSource(self.files_star, particle_size=256)
        imgs_box = src_from_box.images[:10]
        imgs_coord = src_from_coord.images[:10]
        imgs_star = src_from_star.images[:10]
        for i in range(10):
            self.assertTrue(np.array_equal(imgs_box[i], imgs_coord[i]))
            self.assertTrue(np.array_equal(imgs_coord[i], imgs_star[i]))

    def testCached(self):
        src_cached = BoxesCoordinateSource(self.files_box)
        src_uncached = BoxesCoordinateSource(self.files_box)
        src_cached.cache()
        self.assertTrue(
            np.array_equal(
                src_cached.images[:].asnumpy(), src_uncached.images[:].asnumpy()
            )
        )

    def testImagesRandomIndices(self):
        # ensure that we can load a specific, possibly out of order, list of
        # indices, and that the result is in the order we asked for
        src_from_box = BoxesCoordinateSource(self.files_box)
        images_in_order = src_from_box.images[:400]
        # test loading every other image and compare
        odd = np.array([i for i in range(1, 400, 2)])
        even = np.array([i for i in range(0, 399, 2)])
        odd_images = src_from_box.images[odd]
        even_images = src_from_box.images[even]
        for i in range(0, 200):
            self.assertTrue(np.array_equal(images_in_order[2 * i], even_images[i]))
            self.assertTrue(np.array_equal(images_in_order[2 * i + 1], odd_images[i]))

        # random sample of [0,400) of length 100
        random_sample = np.array(random.sample([i for i in range(400)], 100))
        random_images = src_from_box.images[random_sample]
        for i, idx in enumerate(random_sample):
            self.assertTrue(np.array_equal(images_in_order[idx], random_images[i]))

        # include negative indices
        random_sample_neg = np.array(random.sample([i for i in range(-200, 200)], 100))
        random_images_neg = src_from_box.images[random_sample_neg]
        for i, idx in enumerate(random_sample_neg):
            self.assertTrue(np.array_equal(images_in_order[idx], random_images_neg[i]))

    def testMaxRows(self):
        src_from_box = BoxesCoordinateSource(self.files_box)
        imgs = src_from_box.images[:400]
        # make sure max_rows loads the correct particles
        src_100 = BoxesCoordinateSource(self.files_box, max_rows=100)
        imgs_100 = src_100.images[:]
        for i in range(100):
            self.assertTrue(np.array_equal(imgs[i], imgs_100[i]))
        # make sure max_rows > self.n loads max_rows images
        src_500 = BoxesCoordinateSource(self.files_box, max_rows=500)
        self.assertEqual(src_500.n, 400)
        imgs_500 = src_500.images[:400]
        for i in range(400):
            self.assertTrue(np.array_equal(imgs[i], imgs_500[i]))
        # make sure max_rows loads correct particles
        # when some have been excluded
        imgs_newsize = BoxesCoordinateSource(self.files_box, particle_size=336).images[
            :50
        ]
        src_maxrows = BoxesCoordinateSource(
            self.files_box, particle_size=336, max_rows=50
        )
        # max_rows still loads 50 images even if some particles were excluded
        self.assertEqual(src_maxrows.n, 50)
        imgs_maxrows = src_maxrows.images[:50]
        for i in range(50):
            self.assertTrue(np.array_equal(imgs_newsize[i], imgs_maxrows[i]))

    def testBoundaryParticlesRemoved(self):
        src_centers_larger_particles = CentersCoordinateSource(
            self.files_coord, particle_size=300
        )
        src_box_larger_particles = BoxesCoordinateSource(
            self.files_box, particle_size=300
        )
        # 100 particles do not fit at this particle size
        self.assertEqual(src_centers_larger_particles.n, 300)
        self.assertEqual(src_box_larger_particles.n, 300)
        # make sure we have the same particles
        imgs_centers = src_centers_larger_particles.images[:300]
        imgs_resized = src_box_larger_particles.images[:300]
        for i in range(50):
            self.assertTrue(np.array_equal(imgs_centers[i], imgs_resized[i]))

    def testEvenOddResize(self):
        # test a range of even and odd resizes
        for _size in range(252, 260):
            src_centers = CentersCoordinateSource(self.files_coord, particle_size=_size)
            src_resized = BoxesCoordinateSource(self.files_box, particle_size=_size)
            # some particles might be chopped off for sizes greater than
            # 256, so we just load the first 300 images for comparison
            imgs_centers = src_centers.images[:300]
            imgs_resized = src_resized.images[:300]
            for i in range(300):
                self.assertTrue(np.array_equal(imgs_centers[i], imgs_resized[i]))

    def testSave(self):
        # we can save the source into an .mrcs stack with *no* metadata
        src = BoxesCoordinateSource(self.files_box, max_rows=10)
        imgs = src.images[:10]
        star_path = os.path.join(self.data_folder, "stack.star")
        src.save(star_path)
        # load saved particle stack
        saved_star = StarFile(star_path)
        # we want to read the saved mrcs file from the STAR file
        image_name_column = saved_star.get_block_by_index(0)["_rlnImageName"]
        # we're reading a string of the form 0000X@mrcs_path.mrcs
        _particle, mrcs_path = image_name_column.iloc[0].split("@")
        saved_mrcs_stack = mrcfile.open(os.path.join(self.data_folder, mrcs_path)).data
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
        src.images[:5]

    def testWrongNumberCtfFiles(self):
        # trying to give 3 CTF files to a source with 2 micrographs should error
        with self.assertRaises(ValueError):
            src = BoxesCoordinateSource(self.files_box)
            src.import_ctf(["badfile", "badfile", "badfile"])

    def testImportCtfFromList(self):
        src = BoxesCoordinateSource(self.files_box)
        src.import_ctf(self.ctf_files)
        self._testCtfFilters(src)
        self._testCtfMetadata(src)

    def testImportCtfFromRelion(self):
        src = BoxesCoordinateSource(self.files_box)
        src.import_ctf(self.relion_ctf_file)
        self._testCtfFilters(src)
        self._testCtfMetadata(src)

    def _testCtfFilters(self, src):
        # there are two micrographs and two CTF files, so there should be two
        # unique CTF filters
        self.assertEqual(len(src.unique_filters), 2)
        # test the properties of the CTF filters
        # based on the arbitrary values we added to the CTF files
        # note these values are not realistic
        filter0 = src.unique_filters[0]
        self.assertEqual(
            (1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0),
            (
                filter0.defocus_u,
                filter0.defocus_v,
                filter0.defocus_ang,
                filter0.Cs,
                filter0.alpha,
                filter0.voltage,
                filter0.pixel_size,
            ),
        )
        filter1 = src.unique_filters[1]
        self.assertEqual(
            (1001.0, 901.0, 801.0, 701.0, 601.0, 501.0, 401.0),
            (
                filter1.defocus_u,
                filter1.defocus_v,
                filter1.defocus_ang,
                filter1.Cs,
                filter1.alpha,
                filter1.voltage,
                filter1.pixel_size,
            ),
        )
        # the first 200 particles should correspond to the first filter
        # since they came from the first micrograph
        self.assertTrue(
            np.array_equal(np.where(src.filter_indices == 0)[0], np.arange(0, 200))
        )
        # the last 200 particles should correspond to the second filter
        self.assertTrue(
            np.array_equal(np.where(src.filter_indices == 1)[0], np.arange(200, 400))
        )

    def _testCtfMetadata(self, src):
        # ensure metadata is populated correctly when adding CTF info
        # __mrc_filepath
        mrc_fp_metadata = np.array(
            [self.all_mrc_paths[0]] * 200 + [self.all_mrc_paths[1]] * 200
        ).astype(object)
        self.assertTrue(
            np.array_equal(mrc_fp_metadata, src.get_metadata("__mrc_filepath"))
        )
        # __mrc_index
        mrc_idx_metadata = np.array([0] * 200 + [1] * 200)
        self.assertTrue(
            np.array_equal(mrc_idx_metadata, src.get_metadata("__mrc_index"))
        )
        # __filter_indices
        filter_indices_metadata = np.array([0] * 200 + [1] * 200)
        self.assertTrue(
            np.array_equal(
                filter_indices_metadata, src.get_metadata("__filter_indices")
            )
        )
        # CTF metadata
        ctf_cols = [
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnVoltage",
        ]
        ctf_metadata = np.zeros((src.n, len(ctf_cols)), dtype=np.float64)
        ctf_metadata[:200] = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0])
        ctf_metadata[200:400] = np.array([1001.0, 901.0, 801.0, 701.0, 601.0, 501.0])
        self.assertTrue(np.array_equal(ctf_metadata, src.get_metadata(ctf_cols)))

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
        result_preprocess = runner.invoke(
            extract_particles,
            [
                f"--mrc_paths={self.data_folder}/*.mrc",
                f"--coord_paths={self.data_folder}/sample*.box",
                f"--starfile_out={self.data_folder}/saved_star_ds.star",
                "--downsample=33",
                "--normalize_bg",
                "--whiten",
                "--invert_contrast",
            ],
        )
        # check that all commands completed successfully
        self.assertTrue(result_box.exit_code == 0)
        self.assertTrue(result_coord.exit_code == 0)
        self.assertTrue(result_star.exit_code == 0)
        self.assertTrue(result_preprocess.exit_code == 0)
