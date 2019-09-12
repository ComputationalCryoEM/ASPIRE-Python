from unittest import TestCase
import importlib_resources
from pandas import DataFrame

import aspire.data
from aspire.io.starfile import Starfile, StarfileBlock


import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class StarfileTestCase(TestCase):
    def setUp(self):
        with importlib_resources.path(aspire.data, 'sample.star') as path:
            self.starfile = Starfile(path)

    def tearDown(self):
        pass

    def testLength(self):
        # Starfile is an iterable that gives us blocks. We have 2 blocks in our sample starfile.
        self.assertEqual(2, len(self.starfile))

    def testIteration(self):
        # A Starfile can be iterated over, yielding StarfileBlocks
        for block in self.starfile:
            self.assertTrue(isinstance(block, StarfileBlock))

    def testBlockByIndex(self):
        # Indexing a Starfile with a 0-based index gives us a 'block',
        block0 = self.starfile[0]
        self.assertTrue(isinstance(block0, StarfileBlock))
        # Our first block has no 'loop's.
        self.assertEqual(0, len(block0))

    def testBlockByName(self):
        # Indexing a Starfile with a string gives us a block with that name ("data_<name>" in starfile).
        # In our case the block at index 1 has name 'planetary'
        block1 = self.starfile['planetary']
        # This block has a two 'loops'.
        self.assertEqual(2, len(block1))

    def testBlockProperties(self):
        # A StarfileBlock may have attributes that were read from the starfile key=>value pairs
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
        # Note that no typecasting of values is performed at the io.Starfile level
        self.assertEqual('1', df[df['_name'] == 'Earth'].iloc[0]['_gravity'])

    def testData2(self):
        df = self.starfile['planetary'][1]
        self.assertEqual(3, len(df))
        self.assertEqual(2, len(df.columns))
        # Missing values in a loop default to ''
        self.assertEqual('', df[df['_name'] == 'Earth'].iloc[0]['_discovered_year'])

    def testSave(self):
        # Save the Starfile object to disk, re-read it back, and check for equality
        # Note that __eq__ is supported for Starfile/StarfileBlock classes

        self.starfile.save('sample_saved.star')
        self.starfile2 = Starfile('sample_saved.star')
        self.assertEqual(self.starfile, self.starfile2)

        os.remove('sample_saved.star')
