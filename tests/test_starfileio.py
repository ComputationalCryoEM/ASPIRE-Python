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

    def testDictIteration(self):
        # starfile_singleblock and starfile_multiblock are OrderedDicts 
        # of dataframes indexed by strings
        for blockname, block in self.starfile_singleblock.items():
            self.assertTrue(isinstance(blockname, str))
            self.assertTrue(isinstance(block, pd.DataFrame))
        for blockname, block in self.starfile_multiblock.items():
            self.assertTrue(isinstance(blockname, str)
            self.assertTrue(isinstance(block, pd.DataFrame)

    def testListIteration(self):
        # to access blocks by index, we have to convert the items() 
        # of the OrderedDict to a list of tuples
        list_singleblock = list(self.starfile_singleblock.items())
        list_multiblock = list(self.starfile_multiblock.items())
        for i in range(len(list_singleblock)):
            # this is a tuple containing the key and the value at index i
            kv_pair = list_singleblock[i]
            self.assertTrue(isinstance(kv_pair[0], str))
            self.assertTrue(isinstance(kv_pair[1], pd.DataFrame))
        for i in range(len(list_multiblock)):
            # this is a tuple containing the key and the value at index i
            kv_pair = list_multiblock[i]
            self.assertTrue(isinstance(kv_pair[0], str))
            self.assertTrue(isinstance(kv_pair[1], pd.DataFrame))

    def testBlockByName(self):
        # We can access each block by name.
        # the blocks are indexed by their name in the OrderedDict

        # single block case:
        datablock = self.starfile_singleblock['model_class_1']
        # make sure we're actually getting a DF
        self.assertTrue(isinstance(datablock, pd.DataFrame))
        # make sure this is the *right* DF
        self.assertEqual(17, len(datablock))
        self.assertEqual(29, len(datablock.columns))

        # multi block case:
        block1 = self.starfile_multiblock['model_general']
        self.assertTrue(isinstance(block1, pd.DataFrame)
        self.assertEqual(1, len(block1))
        self.assertEqual(22, len(block1.columns))
        block2 = self.starfile_multiblock['model_classes']
        self.assertTrue(isinstance(block2, pd.DataFrame)
        self.assertEqual(1, len(block2))
        self.assertEqual(4, len(block2.columns))
        block3 = self.starfile_multiblock['model_class_1']
        self.assertTrue(isinstance(block3, pd.DataFrame))
        self.assertEqual(76, len(block3))
        self.assertEqual(8, len(block3.columns))

    def testBlockByIndex(self):
        # We can access each block by index if we convert the OrderedDict to a list
        list_singleblock = list(self.starfile_singleblock.items())
        kv_pair_single = list_singleblock[0]
        # make sure we are accessing the correct block
        self.assertEqual('model_class_1', kv_pair_single[0])
        self.assertEqual(self.starfile_singleblock['model_class_1'], kv_pair_single[1])

        list_multiblock = list(self.starfile_multiblock.items())
        kv_pair0 = list_multiblock[0]
        self.assertEqual('model_general', kv_pair0[0])
        self.assertEqual(self.starfile_multiblock['model_general'], kv_pair0[1])
        kv_pair1 = list_multiblock[1]
        self.assertEqual('model_classes', kv_pair1[0])
        self.assertEqual(self.starfile_multiblock['model_classes'], kv_pair1[1])
        kv_pair2 = list_multiblock[2]
        self.assertEqual('model_class_1', kv_pair2[0])
        self.assertEqual(self.starfile_multiblock['model_class_1'], kv_pair2[1])
                        

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
