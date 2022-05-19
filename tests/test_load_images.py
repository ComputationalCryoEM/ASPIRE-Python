import os
import tempfile
from collections import OrderedDict
from unittest import TestCase

import mrcfile
import numpy as np
from pandas import DataFrame

from aspire.image import Image
from aspire.source import RelionSource
from aspire.storage import StarFile
from aspire.utils.random import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class LoadImagesTestCase(TestCase):
    def setUp(self):
        # set up temporary directory
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_folder = self.tmpdir.name

        self.particles_per_stack = 100
        self.num_stacks = 10
        self.n = self.particles_per_stack * self.num_stacks  # 1000
        self.L = 16
        self.dtype = np.float32
        self.seed = 14

        # setting up starfile for simulated RelionSource file structure
        self.starfile_path = os.path.join(self.data_folder, "load_images_test.star")
        starfile_data = OrderedDict()
        starfile_keys = ["_rlnImageName"]
        starfile_loop = []

        for i in range(self.num_stacks):
            # fill 10 mrcs files with random data
            mrcs_fn = f"particle_stack_{i:02d}.mrcs"
            data = random(
                seed=14, size=(self.particles_per_stack, self.L, self.L)
            ).astype(self.dtype)
            with mrcfile.new(os.path.join(self.data_folder, mrcs_fn)) as mrc:
                mrc.set_data(data)

            # write the particle identifiers to our starfile data
            for j in range(self.particles_per_stack):
                starfile_loop.append(f"{j+1:06d}@{mrcs_fn}")

        # save starfile
        starfile_data[""] = DataFrame(starfile_loop, columns=starfile_keys)
        StarFile(blocks=starfile_data).write(self.starfile_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def testRelionSourceInOrder(self):
        # ensure that we can load images in order using
        # the start and num arguments
        src = RelionSource(self.starfile_path, data_folder=self.data_folder)
        imgs = src.images(range(0, 500))
        from_mrc = self.getParticlesFromIndices([i for i in range(0, 500)])
        self.assertTrue(np.array_equal(imgs.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceNonsequential(self):
        # ensure that we can load particles from different mrc files,
        # nonsequentially, and end up with the ordering we asked for
        src = RelionSource(self.starfile_path, data_folder=self.data_folder)
        indices = [501, 502, 503, 504, 505, 0, 1, 2, 3, 4, 729, 728, 730, 720]
        from_src = src.images(indices)
        from_mrc = self.getParticlesFromIndices(indices)
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceEveryOther(self):
        src = RelionSource(self.starfile_path, data_folder=self.data_folder)
        even_indices = [i for i in range(0, self.n, 2)]
        odd_indices = [i for i in range(1, self.n, 2)]
        even_from_src = src.images(even_indices)
        even_from_mrc = self.getParticlesFromIndices(even_indices)
        odd_from_src = src.images(odd_indices)
        odd_from_mrc = self.getParticlesFromIndices(odd_indices)
        self.assertTrue(
            np.array_equal(even_from_src.asnumpy(), even_from_mrc.asnumpy())
        )
        self.assertTrue(np.array_equal(odd_from_src.asnumpy(), odd_from_mrc.asnumpy()))

    def getParticlesFromIndices(self, indices):
        # The purpose of this function is to load the *true* particles from
        # the indices provided. We do this by bypassing the logic in the code we
        # want to test, instead loading the "slow" way directly from the mrcs files
        # created in setUp
        # we can do this efficiently because the particles per mrcs are fixed
        mdata = np.zeros((len(indices), self.L, self.L))
        for i, idx in enumerate(indices):
            # which stack is this particle in
            mrcs = idx // self.particles_per_stack
            # in that stack, at what index is the particle stored
            mrc_index = idx % self.particles_per_stack
            # load the right mrcs
            data = mrcfile.open(
                os.path.join(self.data_folder, f"particle_stack_{mrcs:02d}.mrcs")
            ).data
            # get the right particle
            mdata[i, :, :] = data[mrc_index, :, :]

        return Image(mdata)
