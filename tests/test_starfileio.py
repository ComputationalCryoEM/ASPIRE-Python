import os.path
import tempfile
from itertools import zip_longest
from unittest import TestCase

import importlib_resources
import numpy as np
from pandas import DataFrame
from scipy import misc

import tests.saved_test_data
from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.storage.starfile import StarFile

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


# From itertools standard recipes
def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.

    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    :param iterable: Iterable object to split into chunks
    :param n: Size of each chunk
    :param fillvalue: Value to tail fill if iterable not exact multiple of n
    :return: iterator over chunks of length n
    """

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class StarFileTestCase(TestCase):
    def setUp(self):
        with importlib_resources.path(tests.saved_test_data, "sample_relion_data.star") as path:
            self.starfile_singleblock = StarFile.read(path)
        with importlib_resources.path(tests.saved_test_data, "sample_data_model.star") as path:
            self.starfile_multiblock = StarFile.read(path)
        # Independent Image object for testing Image source methods
        L = 768
        self.im = Image(misc.face(gray=True).astype("float64")[:L, :L])
        self.img_src = ArrayImageSource(self.im)

        # We also want to flex the stack logic.
        self.n = 21
        im_stack = np.broadcast_to(self.im.data, (self.n, L, L))
        # make each image methodically different
        im_stack = np.multiply(im_stack, np.arange(self.n)[:, None, None])
        self.im_stack = Image(im_stack)
        self.img_src_stack = ArrayImageSource(self.im_stack)

        # Create a tmpdir object for this test instance
        self._tmpdir = tempfile.TemporaryDirectory()
        # Get the directory from the name attribute of the instance
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        # Destroy the tmpdir instance and contents
        self._tmpdir.cleanup()

    def testLengthSingleBlock(self):
        # starfile_oneblock is an OrderedDict containing one
        # pandas dataframe, representing the single data block
        # in this STAR file. It should have length 1
        self.assertEqual(1, len(self.starfile_singleblock))

    def testLengthMultiBlock(self):
        # starfile_multiblock is an OrderedDict containing 6
        # pandas dataframes, representing the 6 data blocks
        # in this STAR file. It should have length 6
        self.assertEqual(6, len(self.starfile_multiblock)

    def testIteration(self):
        # A StarFile can be iterated over, yielding StarFileBlocks
        for block in self.starfile:
            self.assertTrue(isinstance(block, StarFileBlock))

    def testBlockByIndex(self):
        # Indexing a StarFile with a 0-based index gives us a 'block',
        block0 = self.starfile[0]
        self.assertTrue(isinstance(block0, StarFileBlock))
        # Our first block has no 'loop's.
        self.assertEqual(0, len(block0))

    def testBlockByName(self):
        # Indexing a StarFile with a string gives us a block with that name
        #   ("data_<name>" in starfile).
        # In our case the block at index 1 has name 'planetary'
        block1 = self.starfile["planetary"]
        # This block has a two 'loops'.
        self.assertEqual(2, len(block1))

    def testBlockProperties(self):
        # A StarFileBlock may have attributes that were read from the
        #   starfile key=>value pairs.
        block0 = self.starfile["general"]
        # Note that no typecasting is performed
        self.assertEqual(block0._three, "3")

    def testLoop(self):
        loop = self.starfile[1][0]
        self.assertIsInstance(loop, DataFrame)

    def testData1(self):
        df = self.starfile["planetary"][0]
        self.assertEqual(8, len(df))
        self.assertEqual(4, len(df.columns))
        # Note that no typecasting of values is performed at io.StarFile level
        self.assertEqual("1", df[df["_name"] == "Earth"].iloc[0]["_gravity"])

    def testData2(self):
        df = self.starfile["planetary"][1]
        self.assertEqual(3, len(df))
        self.assertEqual(2, len(df.columns))
        # Missing values in a loop default to ''
        self.assertEqual("", df[df["_name"] == "Earth"].iloc[0]["_discovered_year"])

    def testSave(self):
        # Save the StarFile object to disk,
        #   read it back, and check for equality.
        # Note that __eq__ is supported for StarFile/StarFileBlock classes

        with open("sample_saved.star", "w") as f:
            self.starfile.save(f)
        self.starfile2 = StarFile("sample_saved.star")
        self.assertEqual(self.starfile, self.starfile2)

        os.remove("sample_saved.star")
