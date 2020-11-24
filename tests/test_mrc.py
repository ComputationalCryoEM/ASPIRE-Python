import logging
import os
import tempfile
from datetime import datetime
from unittest import TestCase

import mrcfile
import numpy as np

from aspire.storage import MrcStats
from aspire.utils.misc import sha256sum

logger = logging.getLogger(__name__)


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
        self.assertTrue(np.allclose(np.min(self.a), self.stats.amin))

        self.stats.push(-10.0 * abs(self.stats.amin))
        self.assertTrue(np.allclose(-10.0 * abs(np.min(self.a)), self.stats.amin))

    def testMax(self):
        self.assertTrue(np.allclose(np.max(self.a), self.stats.amax))

        self.stats.push(10.0 * self.stats.amax)

        self.assertTrue(np.allclose(10.0 * np.max(self.a), self.stats.amax))

    def testMean(self):
        self.assertTrue(np.allclose(np.mean(self.a), self.stats.amean))

        orig = self.stats.amean

        # Push some enough zeros to halve our original mean
        self.stats.push(np.zeros(self.n * self.n))

        self.assertTrue(np.allclose(self.stats.amean, orig / 2.0))

        # Push enough mean data in to restore it
        self.stats.push(np.ones(self.n * self.n) * orig * 2.0)
        self.assertTrue(np.allclose(self.stats.amean, orig))

    def testRms(self):
        self.assertTrue(np.allclose(np.std(self.a), self.stats.arms))

    def testUpdate(self):
        # Create a tmpdir in a context. Cleans up on exit.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two filenames in our tmpdir
            mrcs_filepath = os.path.join(tmpdir, "test.mrc")
            files = (f"{mrcs_filepath}.1", f"{mrcs_filepath}.2")

            # Note below we will fix the time to avoid racy unit tests.
            epoch = datetime(1970, 1, 1).strftime("%Y-%m-%d %H:%M:%S")
            # The time is also packed into the label by mrcfile package.
            label = "{0:40s}{1:>39s} ".format("Created by aspire unit test", epoch)

            with mrcfile.new_mmap(
                files[0], shape=(self.n, self.n), mrc_mode=2, overwrite=True
            ) as mrc:

                mrc.data[:, :] = self.a
                mrc.update_header_from_data()
                self.stats.update_header(mrc)
                mrc.header.time = epoch
                mrc.header.label[0] = label

            with mrcfile.new_mmap(
                files[1], shape=(self.n, self.n), mrc_mode=2, overwrite=True
            ) as mrc:

                mrc.set_data(self.a.astype(np.float32))
                mrc.header.time = epoch
                mrc.header.label[0] = label

            # Our homebrew and mrcfile files should now match to the bit.
            comparison = sha256sum(files[0]) == sha256sum(files[1])
            # Expected hash:
            # 71355fa0bcd5b989ff88166962ea5d2b78ea032933bd6fda41fbdcc1c6d1a009
            logging.debug(f"sha256(file0): {sha256sum(files[0])}")
            logging.debug(f"sha256(file1): {sha256sum(files[1])}")

            self.assertTrue(comparison)
