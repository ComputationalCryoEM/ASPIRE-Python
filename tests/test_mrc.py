import filecmp
import os
import tempfile
from datetime import datetime
from unittest import TestCase

import mrcfile
import numpy as np

from aspire.io.mrc import MrcStats


class MrcStatsTestCase(TestCase):
    def setUp(self):
        # Create dummy data
        self.n = n = 100
        # Note, integers will overflow on windows, use floats.
        self.a = a = np.arange(n * n).reshape((n, n)).astype(np.float32)

        # Create stats instance
        self.stats = MrcStats()

        # Push chunks of data to it.
        for i in range(n):
            self.stats.push(a[i])

    def testMin(self):
        self.assertTrue(np.allclose(
            np.min(self.a),
            self.stats.amin))

        self.stats.push(-10. * abs(self.stats.amin))
        self.assertTrue(np.allclose(
            -10. * abs(np.min(self.a)),
            self.stats.amin))

    def testMax(self):
        self.assertTrue(np.allclose(
            np.max(self.a),
            self.stats.amax))

        self.stats.push(10. * self.stats.amax)

        self.assertTrue(np.allclose(
            10. * np.max(self.a),
            self.stats.amax))

    def testMean(self):
        self.assertTrue(np.allclose(
            np.mean(self.a),
            self.stats.amean))

        orig = self.stats.amean

        # Push some enough zeros to halve our original mean
        self.stats.push(np.zeros(self.n * self.n))

        self.assertTrue(np.allclose(
            self.stats.amean,
            orig/2.))

        # Push enough mean data in to restore it
        self.stats.push(np.ones(self.n * self.n) * orig * 2.)
        self.assertTrue(np.allclose(
            self.stats.amean,
            orig))

    def testRms(self):
        self.assertTrue(np.allclose(
            np.std(self.a),
            self.stats.arms))

    def testUpdate(self):
        # Create a tmpdir in a context. Cleans up on exit.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two filenames in our tmpdir
            mrcs_filepath = os.path.join(tmpdir, 'test.mrc')
            files = (f'{mrcs_filepath}.1',
                     f'{mrcs_filepath}.2')

            # Note below we will fix the time to avoid racy unit tests.
            epoch = datetime(1970, 1, 1).strftime('%Y-%m-%d %H:%M:%S')

            with mrcfile.new_mmap(
                    files[0],
                    shape=(self.n, self.n),
                    mrc_mode=2,
                    overwrite=True) as mrc:

                mrc.data[:, :] = self.a
                mrc.update_header_from_data()
                self.stats.update_header(mrc)
                mrc.header.time = epoch

            with mrcfile.new_mmap(
                    files[1],
                    shape=(self.n, self.n),
                    mrc_mode=2,
                    overwrite=True) as mrc:

                mrc.set_data(self.a.astype(np.float32))
                mrc.header.time = epoch

            # Our homebrew and mrcfile files should now match to the bit.
            filecmp.clear_cache()  # clear any previous attempts
            # Shallow=False is important to ensure we check file contents.
            self.assertTrue(filecmp.cmp(files[0], files[1], shallow=False))
