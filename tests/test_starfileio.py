import os.path
import tempfile
from collections import OrderedDict
from itertools import zip_longest
from unittest import TestCase

import importlib_resources
import numpy as np
from pandas import DataFrame
from scipy import misc

import tests.saved_test_data
from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.storage import StarFile, StarFileError

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
        with importlib_resources.path(
            tests.saved_test_data, "sample_data_model.star"
        ) as path:
            # explicitly reset blocks variable, ensures identical object is tested
            # in each test
            self.starfile = StarFile(path, blocks=OrderedDict())

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

    def testLength(self):
        # StarFile is an iterable that gives us blocks
        # blocks are pandas DataFrames
        # We have 6 blocks in our sample starfile.
        self.assertEqual(6, len(self.starfile))

    def testIteration(self):
        # A StarFile can be iterated over, yielding DataFrames
        for _, df in self.starfile:
            self.assertTrue(isinstance(df, DataFrame))

    def testBlockByIndex(self):
        # We can use get_block_by_index to retrieve the blocks in
        # the OrderedDict by index
        block0 = self.starfile.get_block_by_index(0)
        self.assertTrue(isinstance(block0, DataFrame))
        # Our first block has one row
        self.assertEqual(1, len(block0))

    def testBlockByName(self):
        # Indexing a StarFile with a string gives us a block with that name
        #   ("data_<name>" in starfile).
        # In our case the block at index 1 has name 'model_classes'
        block1 = self.starfile["model_classes"]
        # This block has one row as well
        self.assertEqual(1, len(block1))

    def testBlockProperties(self):
        # A StarFileBlock may have attributes that were read from the
        #   starfile key=>value pairs.
        block0 = self.starfile["model_general"]
        # Note that no typecasting is performed
        self.assertEqual(block0.at[0, "_rlnReferenceDimensionality"], "3")

    def testData(self):
        df = self.starfile["model_class_1"]
        self.assertEqual(76, len(df))
        self.assertEqual(8, len(df.columns))
        # Note that no typecasting of values is performed at io.StarFile level
        self.assertEqual(
            "0.000000", df[df["_rlnSpectralIndex"] == "0"].iloc[0]["_rlnResolution"]
        )

    def testFileNotFound(self):
        with self.assertRaises(FileNotFoundError):
            StarFile("badfile.star")

    def testReadWriteReadBack(self):
        # Save the StarFile object to a .star file
        # Read it back for object equality
        # Note that __eq__ is supported for the class
        # it checks the equality of the underlying OrderedDicts of DataFrames
        # using pd.DataFrame.equals()
        test_outfile = os.path.join(self.tmpdir, "sample_saved.star")
        self.starfile.write(test_outfile)
        starfile2 = StarFile(test_outfile)
        self.assertEqual(self.starfile, starfile2)

        os.remove(test_outfile)

    def testWriteReadWriteBack(self):
        # setup our temp filenames
        test_outfile = os.path.join(self.tmpdir, "sample_saved.star")
        test_outfile2 = os.path.join(self.tmpdir, "sampled_saved2.star")

        # create a new StarFile object directly via an OrderedDict of DataFrames
        # not by reading a file
        data = OrderedDict()
        # note that GEMMI requires the names of the fields to start with _
        block1_dict = {"_field1": 31, "_field2": 32, "_field3": 33}
        # initialize a single-row data block (a set of pairs in GEMMI-parlance)
        block1 = DataFrame([block1_dict], columns=block1_dict.keys())
        block2_keys = ["_field4", "_field5", "_field6"]
        block2_arr = [[f"{x}{y}" for x in range(3)] for y in range(3)]
        # initialize a loop data block with a list of lists
        block2 = DataFrame(block2_arr, columns=block2_keys)
        data["single_row"] = block1
        data["loops"] = block2
        # initialize with blocks kwarg
        # in test environment only, we have to specify filepath=''
        original = StarFile(blocks=data)
        original.write(test_outfile)
        # in test environment only, we have to specify blocks as an
        # empty OrderedDict
        read_back = StarFile(test_outfile)
        # assert that the read-back objects are equal
        self.assertEqual(original, read_back)
        # write back the second star file object
        read_back.write(test_outfile2)
        # compare the two .star files line by line
        with open(test_outfile) as f_original, open(test_outfile2) as f_read_back:
            lines_original = f_original.readlines()
            lines_read_back = f_read_back.readlines()
            self.assertEqual(lines_original, lines_read_back)

        os.remove(test_outfile)
        os.remove(test_outfile2)

    def testArgsError(self):
        with self.assertRaises(StarFileError):
            _blocks = OrderedDict()
            _blocks[""] = DataFrame(["test","data"])
            with importlib_resources.path(
                tests.saved_test_data, "sample_data_model.star"
            ) as path:
                StarFile(filepath=path, blocks=_blocks)

    def testEmptyInit(self):
        empty = StarFile()
        self.assertTrue(isinstance(empty.blocks, OrderedDict))
        self.assertEqual(len(empty.blocks),0)
