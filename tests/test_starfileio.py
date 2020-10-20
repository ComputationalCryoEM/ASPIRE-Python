import importlib_resources
import numpy as np
import os.path

from os.path import splitext
from pandas import DataFrame
from scipy import misc
from unittest import TestCase

import tests.saved_test_data
from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.source.mrcstack import MrcStack
from aspire.io.starfile import StarFile, StarFileBlock, save_star

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class StarFileTestCase(TestCase):
    def setUp(self):
        with importlib_resources.path(tests.saved_test_data, 'sample.star') as path:
            self.starfile = StarFile(path)

        # Independent Image object for testing Image source methods
        self.im = Image(misc.face(gray=True).astype('float64')[:768, :768])
        self.img_src = ArrayImageSource(self.im)

    def tearDown(self):
        pass

    def testLength(self):
        # StarFile is an iterable that gives us blocks. We have 2 blocks in our sample starfile.
        self.assertEqual(2, len(self.starfile))

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
        # Indexing a StarFile with a string gives us a block with that name ("data_<name>" in starfile).
        # In our case the block at index 1 has name 'planetary'
        block1 = self.starfile['planetary']
        # This block has a two 'loops'.
        self.assertEqual(2, len(block1))

    def testBlockProperties(self):
        # A StarFileBlock may have attributes that were read from the starfile key=>value pairs
        block0 = self.starfile['general']
        # Note that no typecasting is performed
        self.assertEqual(block0._three, '3')

    def testLoop(self):
        loop = self.starfile[1][0]
        self.assertIsInstance(loop, DataFrame)

    def testData1(self):
        df = self.starfile['planetary'][0]
        self.assertEqual(8, len(df))
        self.assertEqual(4, len(df.columns))
        # Note that no typecasting of values is performed at the io.StarFile level
        self.assertEqual('1', df[df['_name'] == 'Earth'].iloc[0]['_gravity'])

    def testData2(self):
        df = self.starfile['planetary'][1]
        self.assertEqual(3, len(df))
        self.assertEqual(2, len(df.columns))
        # Missing values in a loop default to ''
        self.assertEqual('', df[df['_name'] == 'Earth'].iloc[0]['_discovered_year'])

    def testSave(self):
        # Save the StarFile object to disk, re-read it back, and check for equality
        # Note that __eq__ is supported for StarFile/StarFileBlock classes

        with open('sample_saved.star', 'w') as f:
            self.starfile.save(f)
        self.starfile2 = StarFile('sample_saved.star')
        self.assertEqual(self.starfile, self.starfile2)

        os.remove('sample_saved.star')

    def testSaveStar(self):
        test_path = 'sample_save_star.star'
        mrc_path = splitext(test_path)[0] + '_0_0.mrcs'

        try:
            # Save some data using the wrapper.
            #save_star(self.img_src, test_path, save_mode='single')
            save_star(self.img_src, test_path)

            # Read it back using the class.
            starfile2 = StarFile(test_path)
            # Check we get expected filename?or?

            # Use something else to read the data file then...
            saved_data = MrcStack(mrc_path).im.data

            # Compare
            self.assertTrue(np.allclose(self.im.data, saved_data))

        finally:
            # Cleanup
            if os.path.exists(test_path):
                os.remove(test_path)

            if os.path.exists(mrc_path):
                os.remove(mrc_path)
