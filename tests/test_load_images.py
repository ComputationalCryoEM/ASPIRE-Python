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

        # create source
        self.src = RelionSource(self.starfile_path, data_folder=self.data_folder)

    def tearDown(self):
        self.tmpdir.cleanup()

    def testRelionSourceInOrder(self):
        # ensure that we can load images in order using
        # the start and num arguments
        imgs = self.src.images[:500]
        from_mrc = self.getParticlesFromIndices([i for i in range(0, 500)])
        self.assertTrue(np.array_equal(imgs.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceSlice(self):
        imgs_full = self.src.images[0 : self.n : 2]
        imgs_infer_start = self.src.images[: self.n : 2]
        imgs_infer_end = self.src.images[0::2]
        imgs_infer_both = self.src.images[::2]
        from_mrc = self.getParticlesFromIndices([i for i in range(0, self.n, 2)])
        self.assertTrue(
            np.array_equal(
                imgs_full.asnumpy(),
                from_mrc.asnumpy(),
            )
        )
        self.assertTrue(
            np.array_equal(
                imgs_infer_start.asnumpy(),
                from_mrc.asnumpy(),
            )
        )
        self.assertTrue(
            np.array_equal(
                imgs_infer_end.asnumpy(),
                from_mrc.asnumpy(),
            )
        )
        self.assertTrue(
            np.array_equal(
                imgs_infer_both.asnumpy(),
                from_mrc.asnumpy(),
            )
        )

    def testRelionSourceNegIndex(self):
        from_src = self.src.images[-1]
        from_mrc = self.getParticlesFromIndices([self.n - 1])
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceNegSlice(self):
        from_src = self.src.images[-100:100]
        from_mrc = self.getParticlesFromIndices(
            [i for i in range(900, 1000)] + [i for i in range(0, 100)]
        )
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceMaxStop(self):
        from_src = self.src.images[5:2000]
        from_mrc = self.getParticlesFromIndices([i for i in range(5, self.n)])
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceNonsequential(self):
        # ensure that we can load particles from different mrc files,
        # nonsequentially, and end up with the ordering we asked for
        indices = [501, 502, 503, 504, 505, 0, 1, 2, 3, 4, 729, 728, 730, 720]
        from_src = self.src.images[indices]
        from_mrc = self.getParticlesFromIndices(indices)
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceEveryOther(self):
        even_indices = [i for i in range(0, self.n, 2)]
        odd_indices = [i for i in range(1, self.n, 2)]
        even_from_src = self.src.images[even_indices]
        even_from_mrc = self.getParticlesFromIndices(even_indices)
        odd_from_src = self.src.images[odd_indices]
        odd_from_mrc = self.getParticlesFromIndices(odd_indices)
        self.assertTrue(
            np.array_equal(even_from_src.asnumpy(), even_from_mrc.asnumpy())
        )
        self.assertTrue(np.array_equal(odd_from_src.asnumpy(), odd_from_mrc.asnumpy()))

    def testRelionSourceNDArray(self):
        indices = np.arange(self.n)
        from_src = self.src.images[indices]
        from_mrc = self.getParticlesFromIndices(list(indices))
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceRange(self):
        indices = range(self.n)
        from_src = self.src.images[indices]
        from_mrc = self.getParticlesFromIndices(list(indices))
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceFilter(self):
        indices = filter(lambda x: x % 2 == 0, list(range(self.n)))
        from_src = self.src.images[indices]
        from_mrc = self.getParticlesFromIndices(
            [i for i in range(self.n) if i % 2 == 0]
        )
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testRelionSourceTuple(self):
        indices = (1, 2, 3, 4, 5)
        from_src = self.src.images[indices]
        from_mrc = self.getParticlesFromIndices([1, 2, 3, 4, 5])
        self.assertTrue(np.array_equal(from_src.asnumpy(), from_mrc.asnumpy()))

    def testBadInput(self):
        # test a non-iterable, non-NumPy, non-integer input
        with self.assertRaisesRegex(KeyError, "Key for .images"):
            _ = self.src.images[3.14]

    def testRelionSourceBadSlice(self):
        with self.assertRaisesRegex(TypeError, "Non-integer slice components."):
            _ = self.src.images[1.5:1.5:1.5]

    def testRelionSourceOutOfRange(self):
        with self.assertRaisesRegex(KeyError, "Out-of-range indices: "):
            _ = self.src.images[[0, 1, 1100]]

    def testBadNDArray(self):
        with self.assertRaisesRegex(
            KeyError, "Only one-dimensional indexing is allowed for images."
        ):
            _ = self.src.images[np.zeros((3, 3))]

    def testRelionSourceCached(self):
        src_cached = RelionSource(self.starfile_path, data_folder=self.data_folder)
        src_cached.cache()
        self.assertTrue(
            np.array_equal(src_cached.images[:].asnumpy(), self.src.images[:].asnumpy())
        )

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
